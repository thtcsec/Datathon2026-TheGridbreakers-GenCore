"""
Tabular GBDT (XGBoost + LightGBM) daily forecast with TimeSeriesSplit CV.

Self-contained: only pandas, numpy, sklearn, xgboost, lightgbm — no ``src.*`` imports,
so the file can be materialized on Kaggle under ``/kaggle/working/`` and imported via importlib.

Used as an algorithmic signal blended into the neural b39 anchor before LB-guided post-processing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def _to_datetime_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = _to_datetime_col(df, date_col)
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["dayofweek"] = out[date_col].dt.dayofweek
    out["dayofyear"] = out[date_col].dt.dayofyear
    out["weekofyear"] = out[date_col].dt.isocalendar().week.astype("Int64")
    out["quarter"] = out[date_col].dt.quarter
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_month_start"] = out[date_col].dt.is_month_start.astype(int)
    out["is_month_end"] = out[date_col].dt.is_month_end.astype(int)
    out["is_quarter_end"] = out[date_col].dt.is_quarter_end.astype(int)
    out["is_tet_season"] = out["month"].isin([1, 2]).astype(int)
    md = out[date_col].dt.strftime("%m-%d")
    vn_holidays = {"01-01", "04-30", "05-01", "09-02"}
    out["is_vn_holiday"] = md.isin(vn_holidays).astype(int)
    return out


def _merge_web_traffic(df: pd.DataFrame, data_path: str | Path, date_col: str) -> pd.DataFrame:
    out = df.copy()
    traffic_path = os.path.join(str(data_path), "web_traffic.csv")
    if not os.path.exists(traffic_path):
        return out
    traffic = pd.read_csv(traffic_path)
    if "date" not in traffic.columns:
        return out
    traffic["date"] = pd.to_datetime(traffic["date"], errors="coerce")
    numeric_cols = [
        c
        for c in [
            "sessions",
            "unique_visitors",
            "page_views",
            "bounce_rate",
            "avg_session_duration_sec",
        ]
        if c in traffic.columns
    ]
    if not numeric_cols:
        return out
    existing = [c for c in numeric_cols if c in out.columns]
    if existing:
        return out
    traffic_daily = traffic.groupby("date", as_index=False)[numeric_cols].mean()
    out = out.merge(traffic_daily, left_on=date_col, right_on="date", how="left")
    if "date" in out.columns:
        out = out.drop(columns=["date"])
    return out


def _merge_inventory(df: pd.DataFrame, data_path: str | Path) -> pd.DataFrame:
    out = df.copy()
    inv_path = os.path.join(str(data_path), "inventory.csv")
    if not os.path.exists(inv_path):
        return out
    inv = pd.read_csv(inv_path)
    if "snapshot_date" not in inv.columns:
        return out
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"], errors="coerce")
    inv = inv[inv["snapshot_date"].notna()].copy()
    inv["year"] = inv["snapshot_date"].dt.year
    inv["month"] = inv["snapshot_date"].dt.month
    inventory_metrics = [
        c
        for c in [
            "stock_on_hand",
            "units_received",
            "units_sold",
            "stockout_days",
            "days_of_supply",
            "fill_rate",
            "stockout_flag",
            "overstock_flag",
            "reorder_flag",
            "sell_through_rate",
        ]
        if c in inv.columns
    ]
    if not inventory_metrics:
        return out
    inv_monthly = inv.groupby(["year", "month"], as_index=False)[inventory_metrics].mean()
    out = out.merge(inv_monthly, on=["year", "month"], how="left")
    return out


def add_external_features(df: pd.DataFrame, data_path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    out = _merge_web_traffic(df, data_path, date_col)
    out = _merge_inventory(out, data_path)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].ffill()
    return out


def add_lag_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    target_cols: tuple[str, ...] = ("Revenue", "COGS"),
    lags: tuple[int, ...] = (7, 14),
    rolling_windows: tuple[int, ...] = (7, 14),
) -> pd.DataFrame:
    out = df.sort_values(date_col).copy()
    for target in target_cols:
        if target not in out.columns:
            continue
        for lag in lags:
            out[f"{target}_lag_{lag}"] = out[target].shift(lag)
        for window in rolling_windows:
            out[f"{target}_roll_mean_{window}"] = out[target].shift(1).rolling(window=window).mean()
    return out


def _get_xgb(seed: int = 42):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
    )


def _get_lgb(seed: int = 42):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def per_fold_cv_metrics_df(
    X: pd.DataFrame,
    y: pd.Series,
    model_ctor,
    *,
    model_label: str,
    target_label: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """Multi-fold **expanding-window** split (``sklearn.model_selection.TimeSeriesSplit``): one row per fold."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    rows: list[dict[str, Any]] = []
    fold_id = 0
    for train_idx, valid_idx in tss.split(X):
        fold_id += 1
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        m = model_ctor(seed)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        rows.append(
            {
                "fold": fold_id,
                "target": target_label,
                "model": model_label,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "valid_start_row": int(valid_idx.min()),
                "valid_end_row": int(valid_idx.max()),
                "mae": mean_absolute_error(y_va, pred),
            }
        )
    return pd.DataFrame(rows)


def gbdt_per_fold_cv_report(
    data_dir: Path | str,
    *,
    date_col: str = "Date",
    cutoff_date: str | pd.Timestamp = "2022-12-31",
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Stacked per-fold MAE for Revenue/COGS × XGB/LGB (same train matrix as final forecast)."""
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales.csv")
    sales = sales.copy()
    sales[date_col] = pd.to_datetime(sales[date_col], errors="coerce")
    sales = sales[sales[date_col].notna()].sort_values(date_col).reset_index(drop=True)
    sample_anchor = sales.loc[sales[date_col] <= pd.to_datetime(cutoff_date), [date_col]].tail(1).copy()
    train_df, _, feature_cols = build_train_and_forecast_frames(
        sales,
        sample_anchor,
        data_dir,
        date_col=date_col,
        cutoff_date=cutoff_date,
    )
    X_train = train_df[feature_cols]
    parts: list[pd.DataFrame] = []
    for target in ("Revenue", "COGS"):
        y = train_df[target]
        parts.append(
            per_fold_cv_metrics_df(
                X_train,
                y,
                _get_xgb,
                model_label="xgboost",
                target_label=target.lower(),
                n_splits=n_splits,
                seed=seed,
            )
        )
        parts.append(
            per_fold_cv_metrics_df(
                X_train,
                y,
                _get_lgb,
                model_label="lightgbm",
                target_label=target.lower(),
                n_splits=n_splits,
                seed=seed,
            )
        )
    return pd.concat(parts, ignore_index=True)


def _ensemble_fit_predict_train_forecast(
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    ensemble_xgb_weight: float,
) -> dict[str, np.ndarray]:
    """Train full-history GBDT pair per target; predict forecast rows (ensemble mean)."""
    X_train = train_df[feature_cols]
    X_fore = forecast_df[feature_cols]
    preds: dict[str, np.ndarray] = {}
    for target in ("Revenue", "COGS"):
        y = train_df[target]
        mx = _get_xgb(seed)
        mx.fit(X_train, y)
        ml = _get_lgb(seed)
        ml.fit(X_train, y)
        px = mx.predict(X_fore)
        pl = ml.predict(X_fore)
        preds[target] = np.maximum(
            ensemble_xgb_weight * px + (1.0 - ensemble_xgb_weight) * pl,
            0.0,
        )
    return preds


def walk_forward_gbdt_evaluation(
    data_dir: Path | str,
    *,
    windows: list[tuple[str, str, str]],
    date_col: str = "Date",
    seed: int = 42,
    ensemble_xgb_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Rolling-origin **pseudo backtest** on historical dates where ``sales.csv`` has labels.

    Each row: train on ``Date <= train_cutoff``, predict daily rows in ``[test_start, test_end]``,
    compare to actual Revenue/COGS (same lag/traffic logic as final submission builder).

    ``windows`` — list of ``(train_cutoff, test_start, test_end)`` as ISO date strings (inclusive bounds).
    """
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales.csv")
    sales[date_col] = pd.to_datetime(sales[date_col], errors="coerce")
    sales = sales[sales[date_col].notna()].sort_values(date_col).reset_index(drop=True)

    rows_out: list[dict[str, Any]] = []
    for train_cutoff, test_start, test_end in windows:
        tc = pd.to_datetime(train_cutoff)
        ts = pd.to_datetime(test_start)
        te = pd.to_datetime(test_end)
        sample_dates = sales.loc[(sales[date_col] >= ts) & (sales[date_col] <= te), [date_col]]
        if sample_dates.empty:
            rows_out.append(
                {
                    "train_cutoff": str(train_cutoff),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "n_test_days": 0,
                    "mae_revenue": np.nan,
                    "mae_cogs": np.nan,
                    "note": "no labeled rows in range",
                }
            )
            continue

        train_df, forecast_df, feature_cols = build_train_and_forecast_frames(
            sales,
            sample_dates,
            data_dir,
            date_col=date_col,
            cutoff_date=tc,
        )
        preds = _ensemble_fit_predict_train_forecast(
            train_df,
            forecast_df,
            feature_cols,
            seed=seed,
            ensemble_xgb_weight=ensemble_xgb_weight,
        )
        actual = sales.loc[
            (sales[date_col] >= ts) & (sales[date_col] <= te),
            [date_col, "Revenue", "COGS"],
        ].sort_values(date_col)
        pred_frame = pd.DataFrame(
            {
                date_col: forecast_df[date_col].values,
                "Revenue": preds["Revenue"],
                "COGS": preds["COGS"],
            }
        ).sort_values(date_col)
        merged = actual.merge(pred_frame, on=date_col, suffixes=("_act", "_pred"))
        mae_rev = mean_absolute_error(merged["Revenue_act"], merged["Revenue_pred"])
        mae_cogs = mean_absolute_error(merged["COGS_act"], merged["COGS_pred"])
        rows_out.append(
            {
                "train_cutoff": str(train_cutoff),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "n_test_days": len(merged),
                "mae_revenue": float(mae_rev),
                "mae_cogs": float(mae_cogs),
                "note": "walk-forward GBDT ensemble vs actuals",
            }
        )
    return pd.DataFrame(rows_out)


def default_walk_forward_windows_2022() -> list[tuple[str, str, str]]:
    """Three late-2022 blocks (train expands, test is strictly after train cutoff)."""
    return [
        ("2022-06-30", "2022-07-01", "2022-08-31"),
        ("2022-08-31", "2022-09-01", "2022-10-31"),
        ("2022-10-31", "2022-11-01", "2022-12-31"),
    ]


def build_train_and_forecast_frames(
    sales: pd.DataFrame,
    sample_dates: pd.DataFrame,
    data_dir: Path | str,
    *,
    date_col: str = "Date",
    cutoff_date: str | pd.Timestamp = "2022-12-31",
    target_cols: tuple[str, ...] = ("Revenue", "COGS"),
    lags: tuple[int, ...] = (7, 14),
    rolling_windows: tuple[int, ...] = (7, 14),
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = sales.copy()
    if date_col not in train.columns:
        raise KeyError(f"Expected column {date_col} in sales")
    train[date_col] = pd.to_datetime(train[date_col], errors="coerce")
    train = train[train[date_col].notna()].sort_values(date_col).reset_index(drop=True)
    cutoff_ts = pd.to_datetime(cutoff_date)
    train = train[train[date_col] <= cutoff_ts].copy()

    train = add_time_features(train, date_col=date_col)
    train = add_external_features(train, data_path=data_dir, date_col=date_col)
    train = add_lag_features(
        train,
        date_col=date_col,
        target_cols=target_cols,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    lag_cols = [
        c
        for c in train.columns
        if any(c.startswith(f"{t}_lag_") or c.startswith(f"{t}_roll_mean_") for t in target_cols)
    ]
    if lag_cols:
        train = train.dropna(subset=lag_cols).reset_index(drop=True)

    forecast = sample_dates.copy()
    if date_col not in forecast.columns:
        raise KeyError(f"sample_dates must contain {date_col}")
    forecast[date_col] = pd.to_datetime(forecast[date_col], errors="coerce")
    forecast = forecast[forecast[date_col].notna()].sort_values(date_col).reset_index(drop=True)

    forecast = add_time_features(forecast, date_col=date_col)
    forecast = add_external_features(forecast, data_path=data_dir, date_col=date_col)

    history = train.sort_values(date_col)
    for target in target_cols:
        if target not in history.columns:
            continue
        for lag in lags:
            col = f"{target}_lag_{lag}"
            value = history[target].iloc[-lag] if len(history) >= lag else history[target].iloc[-1]
            forecast[col] = value
        for window in rolling_windows:
            col = f"{target}_roll_mean_{window}"
            value = history[target].tail(window).mean()
            forecast[col] = value

    feature_cols = [c for c in train.columns if c not in {date_col, *target_cols}]
    feature_cols = [c for c in feature_cols if c in forecast.columns]

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            med = train[col].median()
            if pd.isna(med):
                med = 0.0
            train[col] = train[col].fillna(med)
            forecast[col] = forecast[col].fillna(med)

    return train, forecast, feature_cols


def tabular_gbdt_forecast(
    data_dir: Path | str,
    sample_submission: pd.DataFrame,
    *,
    date_col: str = "Date",
    cutoff_date: str | pd.Timestamp = "2022-12-31",
    n_splits: int = 5,
    seed: int = 42,
    ensemble_xgb_weight: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Train XGB + LGBM on lag/calendar/external features; report TimeSeriesSplit CV MAE;
    return ensemble predictions on ``sample_submission`` dates (constant lag trick for long horizon).
    """
    data_dir = Path(data_dir)
    sales_path = data_dir / "sales.csv"
    if not sales_path.exists():
        raise FileNotFoundError(f"sales.csv not found under {data_dir}")

    sales = pd.read_csv(sales_path)
    sample = sample_submission[[date_col]].copy()

    train_df, forecast_df, feature_cols = build_train_and_forecast_frames(
        sales,
        sample,
        data_dir,
        date_col=date_col,
        cutoff_date=cutoff_date,
    )

    X_train = train_df[feature_cols]

    diagnostics: dict[str, Any] = {"feature_cols": feature_cols, "n_train": len(train_df)}
    targets = ("Revenue", "COGS")
    cv_fold_parts: list[pd.DataFrame] = []
    for target in targets:
        if target not in train_df.columns:
            raise KeyError(f"Missing target {target} in sales.csv")
        y = train_df[target]
        for ctor, short in ((_get_xgb, "xgb"), (_get_lgb, "lgb")):
            df_fold = per_fold_cv_metrics_df(
                X_train,
                y,
                ctor,
                model_label=short,
                target_label=target.lower(),
                n_splits=n_splits,
                seed=seed,
            )
            cv_fold_parts.append(df_fold)
            diagnostics[f"cv_mae_{target.lower()}_{short}"] = float(df_fold["mae"].mean())
    diagnostics["cv_fold_detail"] = pd.concat(cv_fold_parts, ignore_index=True)

    preds = _ensemble_fit_predict_train_forecast(
        train_df,
        forecast_df,
        feature_cols,
        seed=seed,
        ensemble_xgb_weight=ensemble_xgb_weight,
    )

    out = pd.DataFrame(
        {
            date_col: forecast_df[date_col].values,
            "Revenue": preds["Revenue"],
            "COGS": preds["COGS"],
        }
    )
    out = out.sort_values(date_col).reset_index(drop=True)
    return out, diagnostics


def blend_anchor_with_ml(
    anchor: pd.DataFrame,
    ml_forecast: pd.DataFrame,
    *,
    date_col: str = "Date",
    ml_weight: float = 0.18,
) -> pd.DataFrame:
    """Convex blend: ``(1-w)*anchor + w*ml``. Keeps anchor dominant by default."""
    if not (0.0 <= ml_weight <= 1.0):
        raise ValueError("ml_weight must be in [0, 1]")
    a = anchor.sort_values(date_col).reset_index(drop=True)
    m = ml_forecast.sort_values(date_col).reset_index(drop=True)
    if not a[date_col].equals(m[date_col]):
        m = m.set_index(date_col).loc[a[date_col]].reset_index()
    w = ml_weight
    out = a.copy()
    out["Revenue"] = (1.0 - w) * a["Revenue"].to_numpy(dtype=float) + w * m["Revenue"].to_numpy(dtype=float)
    out["COGS"] = (1.0 - w) * a["COGS"].to_numpy(dtype=float) + w * m["COGS"].to_numpy(dtype=float)
    out["Revenue"] = np.maximum(out["Revenue"], 0.0)
    out["COGS"] = np.maximum(out["COGS"], 0.0)
    return out


def run_ml_blend_into_anchor(
    data_dir: Path,
    out_dir: Path,
    anchor_b39: pd.DataFrame,
    sample: pd.DataFrame,
    *,
    ml_weight: float = 0.18,
    cutoff_date: str | pd.Timestamp = "2022-12-31",
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convenience: train tabular GBDT, blend into anchor, optionally save CSV under ``out_dir``.
    """
    ml_forecast, diag = tabular_gbdt_forecast(
        data_dir,
        sample,
        cutoff_date=cutoff_date,
        n_splits=n_splits,
        seed=seed,
    )
    blended = blend_anchor_with_ml(anchor_b39, ml_forecast, ml_weight=ml_weight)
    diag["ml_weight"] = ml_weight
    return blended, diag
