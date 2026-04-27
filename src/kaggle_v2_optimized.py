"""Optimized long-horizon hybrid forecasting pipeline for Datathon 2026.

Design goals:
1) No leakage in validation/final forecasting.
2) Stable long-horizon behavior for 548-day forecasts.
3) Separate calibration for Revenue and COGS.

Pipeline per target:
- Baseline: blend of 364-day seasonal naive + seasonal neighborhood median.
- Prophet: structural trend/seasonality/holiday model.
- LightGBM residual: predicts y - prophet_yhat from leakage-safe date features.
- Ensemble: rolling-year weight search (2020, 2021, 2022), averaged weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


SEED = 42
np.random.seed(SEED)


@dataclass
class TargetResult:
    target: str
    weights: Tuple[float, float, float]
    fold_mae: Dict[str, float]
    diagnostics: Dict[str, object]


def find_data_dir(explicit_data_dir: str | None) -> str:
    if explicit_data_dir:
        return explicit_data_dir

    kaggle_matches = glob.glob('/kaggle/input/**/sales.csv', recursive=True)
    if kaggle_matches:
        return os.path.dirname(kaggle_matches[0])

    return 'data/raw'


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sales_path = os.path.join(data_dir, 'sales.csv')
    sub_path = os.path.join(data_dir, 'sample_submission.csv')

    if not os.path.exists(sales_path):
        raise FileNotFoundError(f'Missing sales.csv at {sales_path}')
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f'Missing sample_submission.csv at {sub_path}')

    sales = pd.read_csv(sales_path)
    sub = pd.read_csv(sub_path)

    sales['Date'] = pd.to_datetime(sales['Date'], errors='coerce')
    sub['Date'] = pd.to_datetime(sub['Date'], errors='coerce')

    sales = sales[sales['Date'].notna()].copy()
    sub = sub[sub['Date'].notna()].copy()

    sales = sales.sort_values('Date').reset_index(drop=True)
    sub = sub.sort_values('Date').reset_index(drop=True)

    for col in ['Revenue', 'COGS']:
        if col not in sales.columns:
            raise ValueError(f'Column {col} missing in sales.csv')

    return sales, sub


def build_holidays(last_forecast_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # Include historical + forecast Tet dates used in this competition horizon.
    tet_dates = pd.to_datetime(
        [
            '2012-01-23', '2013-02-10', '2014-01-31', '2015-02-19',
            '2016-02-08', '2017-01-28', '2018-02-16', '2019-02-05',
            '2020-01-25', '2021-02-12', '2022-02-01', '2023-01-22',
            '2024-02-10',
        ]
    )
    tet_dates = tet_dates[tet_dates <= (last_forecast_date + pd.Timedelta(days=30))]

    tet_holidays = pd.DataFrame(
        {
            'holiday': 'tet',
            'ds': tet_dates,
            # Pre Tet shopping surge, Tet holiday, and post-Tet recovery.
            'lower_window': -21,
            'upper_window': 14,
        }
    )

    mega_sale_raw: List[pd.Timestamp] = []
    for year in range(2012, last_forecast_date.year + 1):
        for month, day in [(9, 9), (10, 10), (11, 11), (12, 12), (4, 30), (5, 1), (9, 2)]:
            dt = pd.Timestamp(year=year, month=month, day=day)
            if dt <= last_forecast_date:
                mega_sale_raw.append(dt)

    sale_dates = pd.Series(sorted(set(mega_sale_raw)))
    sale_holidays = pd.DataFrame(
        {
            'holiday': 'mega_sale',
            'ds': sale_dates,
            'lower_window': -2,
            'upper_window': 2,
        }
    )

    holidays = pd.concat([tet_holidays, sale_holidays], ignore_index=True)
    return holidays, pd.Series(tet_dates), sale_dates


def circular_doy_distance(d1: np.ndarray, d2: int, period: int = 366) -> np.ndarray:
    diff = np.abs(d1 - d2)
    return np.minimum(diff, period - diff)


def seasonal_364_forecast(train_series: pd.Series, future_dates: Sequence[pd.Timestamp]) -> np.ndarray:
    series = train_series.sort_index().astype(float)
    history: Dict[pd.Timestamp, float] = {pd.Timestamp(i): float(v) for i, v in series.items()}

    fallback = float(series.tail(28).median()) if len(series) >= 28 else float(series.median())
    preds: List[float] = []

    for dt in pd.to_datetime(future_dates):
        dt = pd.Timestamp(dt)
        candidates = [dt - pd.Timedelta(days=364), dt - pd.Timedelta(days=371), dt - pd.Timedelta(days=357)]

        value = None
        for c in candidates:
            if c in history:
                value = history[c]
                break

        if value is None:
            value = fallback

        history[dt] = float(value)
        preds.append(float(value))

    return np.asarray(preds, dtype=float)


def seasonal_window_forecast(
    train_df: pd.DataFrame,
    target_col: str,
    future_dates: Sequence[pd.Timestamp],
    window: int = 7,
) -> np.ndarray:
    work = train_df[['Date', target_col]].copy()
    work['doy'] = work['Date'].dt.dayofyear
    work['dow'] = work['Date'].dt.dayofweek

    doys = work['doy'].to_numpy(dtype=int)
    dows = work['dow'].to_numpy(dtype=int)
    vals = work[target_col].to_numpy(dtype=float)

    global_fallback = float(np.nanmedian(vals))
    preds: List[float] = []

    for dt in pd.to_datetime(future_dates):
        doy = int(dt.dayofyear)
        dow = int(dt.dayofweek)

        dist = circular_doy_distance(doys, doy)
        mask = (dows == dow) & (dist <= window)
        if mask.sum() == 0:
            mask = (dows == dow) & (dist <= (window + 7))
        if mask.sum() == 0:
            mask = dist <= window

        if mask.sum() == 0:
            preds.append(global_fallback)
        else:
            preds.append(float(np.nanmedian(vals[mask])))

    return np.asarray(preds, dtype=float)


def fit_prophet_model(train_df: pd.DataFrame, target_col: str, holidays: pd.DataFrame) -> Tuple[Prophet, float]:
    df = train_df[['Date', target_col]].rename(columns={'Date': 'ds', target_col: 'y'}).copy()
    cap = float(df['y'].quantile(0.995) * 1.20)
    floor = max(0.0, float(df['y'].quantile(0.01) * 0.80))

    df['cap'] = cap
    df['floor'] = floor

    model = Prophet(
        growth='logistic',
        holidays=holidays,
        yearly_seasonality=20,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.08,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=8.0,
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df)
    return model, cap


def prophet_predict_components(model: Prophet, dates: Sequence[pd.Timestamp], cap: float) -> pd.DataFrame:
    future = pd.DataFrame({'ds': pd.to_datetime(list(dates))})
    future['cap'] = cap
    future['floor'] = 0.0
    fc = model.predict(future)

    out = pd.DataFrame({'Date': future['ds']})
    for col in ['yhat', 'trend', 'weekly', 'yearly', 'holidays']:
        out[f'prophet_{col}'] = fc[col] if col in fc.columns else 0.0
    return out


def _days_to_next_event(dates: pd.Series, events: np.ndarray, default_value: int) -> np.ndarray:
    out = np.full(len(dates), default_value, dtype=int)
    idx = np.searchsorted(events, dates.to_numpy(dtype='datetime64[ns]'), side='left')
    for i, (d, j) in enumerate(zip(dates, idx)):
        if j < len(events):
            out[i] = int((pd.Timestamp(events[j]) - d).days)
    return out


def _days_since_last_event(dates: pd.Series, events: np.ndarray, default_value: int) -> np.ndarray:
    out = np.full(len(dates), default_value, dtype=int)
    idx = np.searchsorted(events, dates.to_numpy(dtype='datetime64[ns]'), side='right') - 1
    for i, (d, j) in enumerate(zip(dates, idx)):
        if j >= 0:
            out[i] = int((d - pd.Timestamp(events[j])).days)
    return out


def build_residual_features(
    dates: Sequence[pd.Timestamp],
    target_history: pd.DataFrame,
    target_col: str,
    tet_dates: pd.Series,
    sale_dates: pd.Series,
    origin_date: pd.Timestamp,
    prophet_components: pd.DataFrame,
) -> pd.DataFrame:
    feats = pd.DataFrame({'Date': pd.to_datetime(list(dates))})
    d = feats['Date']

    feats['year'] = d.dt.year
    feats['month'] = d.dt.month
    feats['day'] = d.dt.day
    feats['dayofweek'] = d.dt.dayofweek
    feats['dayofyear'] = d.dt.dayofyear
    feats['weekofyear'] = d.dt.isocalendar().week.astype(int)
    feats['quarter'] = d.dt.quarter
    feats['is_weekend'] = (feats['dayofweek'] >= 5).astype(int)
    feats['is_month_start'] = d.dt.is_month_start.astype(int)
    feats['is_month_end'] = d.dt.is_month_end.astype(int)
    feats['is_payday'] = feats['day'].isin([1, 15]).astype(int)

    feats['days_since_start'] = (d - pd.Timestamp(target_history['Date'].min())).dt.days.astype(int)
    feats['forecast_horizon'] = (d - pd.Timestamp(origin_date)).dt.days.clip(lower=1).astype(int)

    for k in [1, 2, 3, 4]:
        feats[f'sin_y{k}'] = np.sin(2 * np.pi * k * feats['dayofyear'] / 365.25)
        feats[f'cos_y{k}'] = np.cos(2 * np.pi * k * feats['dayofyear'] / 365.25)
    for k in [1, 2]:
        feats[f'sin_w{k}'] = np.sin(2 * np.pi * k * feats['dayofweek'] / 7.0)
        feats[f'cos_w{k}'] = np.cos(2 * np.pi * k * feats['dayofweek'] / 7.0)

    tet_arr = tet_dates.sort_values().to_numpy(dtype='datetime64[ns]')
    sale_arr = sale_dates.sort_values().to_numpy(dtype='datetime64[ns]')

    feats['days_to_tet'] = _days_to_next_event(d, tet_arr, default_value=365)
    feats['days_since_tet'] = _days_since_last_event(d, tet_arr, default_value=365)
    feats['is_pre_tet'] = feats['days_to_tet'].between(1, 21).astype(int)
    feats['is_during_tet'] = feats['days_to_tet'].between(-7, 0).astype(int)
    feats['is_post_tet'] = feats['days_since_tet'].between(1, 14).astype(int)

    feats['days_to_sale'] = _days_to_next_event(d, sale_arr, default_value=365)
    feats['days_since_sale'] = _days_since_last_event(d, sale_arr, default_value=365)
    feats['is_sale_window'] = (feats['days_to_sale'].abs() <= 2).astype(int)
    feats['is_pre_sale'] = feats['days_to_sale'].between(1, 3).astype(int)
    feats['is_post_sale'] = feats['days_since_sale'].between(1, 3).astype(int)

    hist = target_history[['Date', target_col]].copy()
    hist['doy'] = hist['Date'].dt.dayofyear
    doy_map = hist.groupby('doy')[target_col].median()
    feats['hist_doy_target'] = feats['dayofyear'].map(doy_map).fillna(float(hist[target_col].median()))

    feats = feats.merge(prophet_components, on='Date', how='left')
    for col in feats.columns:
        if col == 'Date':
            continue
        if feats[col].isna().any():
            feats[col] = feats[col].fillna(0)

    return feats.drop(columns=['Date'])


def fit_residual_model(X: pd.DataFrame, residual: np.ndarray) -> LGBMRegressor:
    # Emphasize recent data to reduce historical regime drift.
    rank = np.arange(len(X), dtype=float)
    sample_weight = 1.0 + 1.5 * (rank / max(1.0, rank.max()))

    model = LGBMRegressor(
        objective='mae',
        n_estimators=900,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_alpha=0.3,
        reg_lambda=3.0,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(X, residual, sample_weight=sample_weight)
    return model


def optimize_weights(y_true: np.ndarray, p_naive: np.ndarray, p_prophet: np.ndarray, p_hybrid: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
    best_w = (0.3, 0.3, 0.4)
    best_mae = float('inf')

    for w_n in np.arange(0.0, 0.61, 0.05):
        for w_p in np.arange(0.0, 0.81, 0.05):
            w_h = 1.0 - w_n - w_p
            if w_h < 0:
                continue
            pred = w_n * p_naive + w_p * p_prophet + w_h * p_hybrid
            mae = mean_absolute_error(y_true, pred)
            if mae < best_mae:
                best_mae = float(mae)
                best_w = (float(w_n), float(w_p), float(w_h))

    return best_w, best_mae


def evaluate_target(
    sales: pd.DataFrame,
    target_col: str,
    holidays: pd.DataFrame,
    tet_dates: pd.Series,
    sale_dates: pd.Series,
    fold_years: Sequence[int],
) -> TargetResult:
    diagnostics: Dict[str, object] = {'target': target_col, 'folds': []}
    fold_weights: List[Tuple[float, float, float]] = []
    fold_mae_models: Dict[str, List[float]] = {'naive': [], 'prophet': [], 'hybrid': [], 'blend': []}

    for year in fold_years:
        train = sales[sales['Date'].dt.year < year].copy()
        valid = sales[sales['Date'].dt.year == year].copy()

        if train.empty or valid.empty:
            continue

        val_dates = valid['Date'].tolist()
        y_val = valid[target_col].to_numpy(dtype=float)

        naive_364 = seasonal_364_forecast(train.set_index('Date')[target_col], val_dates)
        naive_win = seasonal_window_forecast(train, target_col, val_dates, window=7)
        pred_naive = 0.5 * naive_364 + 0.5 * naive_win

        prophet_model, cap = fit_prophet_model(train, target_col, holidays)
        prophet_val_df = prophet_predict_components(prophet_model, val_dates, cap)
        prophet_train_df = prophet_predict_components(prophet_model, train['Date'].tolist(), cap)

        pred_prophet = prophet_val_df['prophet_yhat'].to_numpy(dtype=float)

        residual_train = train[target_col].to_numpy(dtype=float) - prophet_train_df['prophet_yhat'].to_numpy(dtype=float)
        X_train = build_residual_features(
            dates=train['Date'].tolist(),
            target_history=train[['Date', target_col]],
            target_col=target_col,
            tet_dates=tet_dates,
            sale_dates=sale_dates,
            origin_date=train['Date'].max(),
            prophet_components=prophet_train_df,
        )
        X_val = build_residual_features(
            dates=val_dates,
            target_history=train[['Date', target_col]],
            target_col=target_col,
            tet_dates=tet_dates,
            sale_dates=sale_dates,
            origin_date=train['Date'].max(),
            prophet_components=prophet_val_df,
        )
        residual_model = fit_residual_model(X_train, residual_train)
        residual_pred = residual_model.predict(X_val)
        pred_hybrid = np.clip(pred_prophet + residual_pred, a_min=0.0, a_max=None)

        best_w, best_mae = optimize_weights(y_val, pred_naive, pred_prophet, pred_hybrid)

        fold_weights.append(best_w)
        fold_mae_models['naive'].append(float(mean_absolute_error(y_val, pred_naive)))
        fold_mae_models['prophet'].append(float(mean_absolute_error(y_val, pred_prophet)))
        fold_mae_models['hybrid'].append(float(mean_absolute_error(y_val, pred_hybrid)))
        fold_mae_models['blend'].append(float(best_mae))

        diagnostics['folds'].append(
            {
                'year': int(year),
                'rows_train': int(len(train)),
                'rows_valid': int(len(valid)),
                'mae_naive': fold_mae_models['naive'][-1],
                'mae_prophet': fold_mae_models['prophet'][-1],
                'mae_hybrid': fold_mae_models['hybrid'][-1],
                'mae_blend': fold_mae_models['blend'][-1],
                'weights': {'naive': best_w[0], 'prophet': best_w[1], 'hybrid': best_w[2]},
            }
        )

    if not fold_weights:
        # Safe fallback if folds are empty in any custom setting.
        avg_w = (0.30, 0.35, 0.35)
    else:
        avg_w = tuple(np.mean(np.asarray(fold_weights), axis=0).tolist())

    diagnostics['avg_weights'] = {'naive': avg_w[0], 'prophet': avg_w[1], 'hybrid': avg_w[2]}

    mae_summary = {
        k: float(np.mean(v)) if v else float('nan')
        for k, v in fold_mae_models.items()
    }

    return TargetResult(target=target_col, weights=avg_w, fold_mae=mae_summary, diagnostics=diagnostics)


def apply_dynamic_blend(
    p_naive: np.ndarray,
    p_prophet: np.ndarray,
    p_hybrid: np.ndarray,
    weights: Tuple[float, float, float],
    dynamic: bool,
) -> np.ndarray:
    w_n, w_p, w_h = weights

    if not dynamic or len(p_naive) <= 1:
        pred = w_n * p_naive + w_p * p_prophet + w_h * p_hybrid
        return np.clip(pred, a_min=0.0, a_max=None)

    t = np.linspace(0.0, 1.0, len(p_naive))
    w_h_t = w_h * (1.00 - 0.45 * t)
    w_p_t = w_p * (1.00 + 0.55 * t)
    w_n_t = w_n * (1.00 + 0.20 * t)
    denom = w_h_t + w_p_t + w_n_t

    pred = (w_n_t * p_naive + w_p_t * p_prophet + w_h_t * p_hybrid) / denom
    return np.clip(pred, a_min=0.0, a_max=None)


def train_and_forecast_target(
    sales: pd.DataFrame,
    target_col: str,
    holidays: pd.DataFrame,
    tet_dates: pd.Series,
    sale_dates: pd.Series,
    forecast_dates: Sequence[pd.Timestamp],
    avg_weights: Tuple[float, float, float],
    dynamic_blend: bool,
) -> np.ndarray:
    train = sales[['Date', target_col]].copy()

    naive_364 = seasonal_364_forecast(train.set_index('Date')[target_col], forecast_dates)
    naive_win = seasonal_window_forecast(train, target_col, forecast_dates, window=7)
    pred_naive = 0.5 * naive_364 + 0.5 * naive_win

    prophet_model, cap = fit_prophet_model(train, target_col, holidays)
    prophet_train_df = prophet_predict_components(prophet_model, train['Date'].tolist(), cap)
    prophet_future_df = prophet_predict_components(prophet_model, forecast_dates, cap)
    pred_prophet = prophet_future_df['prophet_yhat'].to_numpy(dtype=float)

    residual_train = train[target_col].to_numpy(dtype=float) - prophet_train_df['prophet_yhat'].to_numpy(dtype=float)
    X_train = build_residual_features(
        dates=train['Date'].tolist(),
        target_history=train,
        target_col=target_col,
        tet_dates=tet_dates,
        sale_dates=sale_dates,
        origin_date=train['Date'].max(),
        prophet_components=prophet_train_df,
    )
    X_future = build_residual_features(
        dates=forecast_dates,
        target_history=train,
        target_col=target_col,
        tet_dates=tet_dates,
        sale_dates=sale_dates,
        origin_date=train['Date'].max(),
        prophet_components=prophet_future_df,
    )
    residual_model = fit_residual_model(X_train, residual_train)
    residual_pred = residual_model.predict(X_future)
    pred_hybrid = np.clip(pred_prophet + residual_pred, a_min=0.0, a_max=None)

    final_pred = apply_dynamic_blend(pred_naive, pred_prophet, pred_hybrid, avg_weights, dynamic_blend)
    return np.clip(final_pred, a_min=0.0, a_max=None)


def parse_years(raw: str) -> List[int]:
    years = []
    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        years.append(int(token))
    if not years:
        raise ValueError('fold years cannot be empty')
    return years


def main() -> None:
    parser = argparse.ArgumentParser(description='Optimized Prophet + LGBM residual pipeline for Datathon 2026')
    parser.add_argument('--data-dir', type=str, default=None, help='Path containing sales.csv and sample_submission.csv')
    parser.add_argument('--out-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--submission-name', type=str, default='submission_v2_optimized.csv', help='Submission filename')
    parser.add_argument('--fold-years', type=str, default='2020,2021,2022', help='Comma-separated validation years')
    parser.add_argument('--disable-dynamic-blend', action='store_true', help='Disable horizon-dependent ensemble blending')
    args = parser.parse_args()

    data_dir = find_data_dir(args.data_dir)
    sales, sub = load_data(data_dir)
    forecast_dates = sub['Date'].tolist()

    holidays, tet_dates, sale_dates = build_holidays(sub['Date'].max())
    fold_years = parse_years(args.fold_years)

    print(f'DATA_DIR: {data_dir}')
    print(f'Train range: {sales.Date.min().date()} -> {sales.Date.max().date()} ({len(sales)} rows)')
    print(f'Forecast horizon: {forecast_dates[0].date()} -> {forecast_dates[-1].date()} ({len(forecast_dates)} rows)')
    print(f'Fold years: {fold_years}')

    results: Dict[str, TargetResult] = {}
    for target in ['Revenue', 'COGS']:
        print('\n' + '=' * 72)
        print(f'Calibrating target: {target}')
        print('=' * 72)

        res = evaluate_target(
            sales=sales,
            target_col=target,
            holidays=holidays,
            tet_dates=tet_dates,
            sale_dates=sale_dates,
            fold_years=fold_years,
        )
        results[target] = res

        print(f'Average weights ({target}): naive={res.weights[0]:.3f}, prophet={res.weights[1]:.3f}, hybrid={res.weights[2]:.3f}')
        print(f'Mean fold MAE ({target}): naive={res.fold_mae["naive"]:,.0f}, prophet={res.fold_mae["prophet"]:,.0f}, hybrid={res.fold_mae["hybrid"]:,.0f}, blend={res.fold_mae["blend"]:,.0f}')

    dynamic_blend = not args.disable_dynamic_blend
    rev_pred = train_and_forecast_target(
        sales=sales,
        target_col='Revenue',
        holidays=holidays,
        tet_dates=tet_dates,
        sale_dates=sale_dates,
        forecast_dates=forecast_dates,
        avg_weights=results['Revenue'].weights,
        dynamic_blend=dynamic_blend,
    )
    cogs_pred = train_and_forecast_target(
        sales=sales,
        target_col='COGS',
        holidays=holidays,
        tet_dates=tet_dates,
        sale_dates=sale_dates,
        forecast_dates=forecast_dates,
        avg_weights=results['COGS'].weights,
        dynamic_blend=dynamic_blend,
    )

    submission = pd.DataFrame(
        {
            'Date': pd.to_datetime(forecast_dates).strftime('%Y-%m-%d'),
            'Revenue': np.round(np.clip(rev_pred, a_min=0.0, a_max=None), 2),
            'COGS': np.round(np.clip(cogs_pred, a_min=0.0, a_max=None), 2),
        }
    )

    if len(submission) != len(sub):
        raise RuntimeError(f'Submission length mismatch: expected {len(sub)}, got {len(submission)}')
    if submission[['Revenue', 'COGS']].isna().any().any():
        raise RuntimeError('Submission contains NaN values')

    os.makedirs(args.out_dir, exist_ok=True)
    submission_path = os.path.join(args.out_dir, args.submission_name)
    submission.to_csv(submission_path, index=False, float_format='%.2f')

    diag_payload = {
        'dynamic_blend': dynamic_blend,
        'fold_years': fold_years,
        'results': {
            k: {
                'weights': {'naive': v.weights[0], 'prophet': v.weights[1], 'hybrid': v.weights[2]},
                'fold_mae': v.fold_mae,
                'diagnostics': v.diagnostics,
            }
            for k, v in results.items()
        },
    }
    diag_path = os.path.join(args.out_dir, 'v2_optimized_diagnostics.json')
    with open(diag_path, 'w', encoding='utf-8') as f:
        json.dump(diag_payload, f, indent=2, ensure_ascii=True)

    print('\n' + '=' * 72)
    print('DONE')
    print('=' * 72)
    print(f'Submission: {submission_path}')
    print(f'Diagnostics: {diag_path}')
    print(f'Revenue mean/std: {submission.Revenue.mean():,.0f} / {submission.Revenue.std():,.0f}')
    print(f'COGS mean/std:    {submission.COGS.mean():,.0f} / {submission.COGS.std():,.0f}')


if __name__ == '__main__':
    main()
