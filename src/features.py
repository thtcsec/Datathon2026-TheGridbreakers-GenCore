import os
import numpy as np
import pandas as pd


def _detect_date_col(df):
    if 'order_date' in df.columns:
        return 'order_date'
    if 'Date' in df.columns:
        return 'Date'
    raise KeyError("Expected a date column: 'order_date' or 'Date'.")


def _to_datetime_col(df, col):
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors='coerce')
    return out


def add_time_features(df, date_col='order_date'):
    out = _to_datetime_col(df, date_col)

    out['year'] = out[date_col].dt.year
    out['month'] = out[date_col].dt.month
    out['day'] = out[date_col].dt.day
    out['dayofweek'] = out[date_col].dt.dayofweek
    out['dayofyear'] = out[date_col].dt.dayofyear
    out['weekofyear'] = out[date_col].dt.isocalendar().week.astype('Int64')
    out['quarter'] = out[date_col].dt.quarter

    out['is_weekend'] = (out['dayofweek'] >= 5).astype(int)
    out['is_month_start'] = out[date_col].dt.is_month_start.astype(int)
    out['is_month_end'] = out[date_col].dt.is_month_end.astype(int)
    out['is_quarter_end'] = out[date_col].dt.is_quarter_end.astype(int)

    # Approximate Tet window used by many VN retail models.
    out['is_tet_season'] = out['month'].isin([1, 2]).astype(int)

    # Fixed-date holidays (solar calendar) as a lightweight baseline.
    md = out[date_col].dt.strftime('%m-%d')
    vn_holidays = {'01-01', '04-30', '05-01', '09-02'}
    out['is_vn_holiday'] = md.isin(vn_holidays).astype(int)

    return out


def _merge_web_traffic(df, data_path, date_col='order_date'):
    out = df.copy()
    traffic_path = os.path.join(data_path, 'web_traffic.csv')
    if not os.path.exists(traffic_path):
        return out

    traffic = pd.read_csv(traffic_path)
    if 'date' not in traffic.columns:
        return out

    traffic['date'] = pd.to_datetime(traffic['date'], errors='coerce')

    numeric_cols = [
        c for c in ['sessions', 'unique_visitors', 'page_views', 'bounce_rate', 'avg_session_duration_sec']
        if c in traffic.columns
    ]
    if not numeric_cols:
        return out

    # Avoid duplicate suffix columns when the base frame already contains traffic fields.
    existing = [c for c in numeric_cols if c in out.columns]
    if existing:
        return out

    traffic_daily = traffic.groupby('date', as_index=False)[numeric_cols].mean()
    out = out.merge(traffic_daily, left_on=date_col, right_on='date', how='left')
    if 'date' in out.columns:
        out = out.drop(columns=['date'])

    return out


def _merge_inventory(df, data_path):
    out = df.copy()
    inv_path = os.path.join(data_path, 'inventory.csv')
    if not os.path.exists(inv_path):
        return out

    inv = pd.read_csv(inv_path)
    if 'snapshot_date' not in inv.columns:
        return out

    inv['snapshot_date'] = pd.to_datetime(inv['snapshot_date'], errors='coerce')
    inv = inv[inv['snapshot_date'].notna()].copy()
    inv['year'] = inv['snapshot_date'].dt.year
    inv['month'] = inv['snapshot_date'].dt.month

    inventory_metrics = [
        c for c in [
            'stock_on_hand', 'units_received', 'units_sold', 'stockout_days',
            'days_of_supply', 'fill_rate', 'stockout_flag', 'overstock_flag',
            'reorder_flag', 'sell_through_rate'
        ] if c in inv.columns
    ]
    if not inventory_metrics:
        return out

    inv_monthly = inv.groupby(['year', 'month'], as_index=False)[inventory_metrics].mean()
    out = out.merge(inv_monthly, on=['year', 'month'], how='left')

    return out


def add_external_features(df, data_path='data/raw', date_col='order_date'):
    out = _merge_web_traffic(df, data_path=data_path, date_col=date_col)
    out = _merge_inventory(out, data_path=data_path)

    # Use forward-fill only to avoid introducing future information.
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].ffill()

    return out


def add_lag_features(df, date_col='order_date', target_cols=('Revenue', 'COGS'), lags=(7, 14), rolling_windows=(7, 14)):
    out = df.sort_values(date_col).copy()

    for target in target_cols:
        if target not in out.columns:
            continue
        for lag in lags:
            out[f'{target}_lag_{lag}'] = out[target].shift(lag)
        for window in rolling_windows:
            out[f'{target}_roll_mean_{window}'] = out[target].shift(1).rolling(window=window).mean()

    return out


def build_features(
    train_df,
    forecast_df=None,
    data_path='data/raw',
    date_col='order_date',
    cutoff_date=None,
    target_cols=('Revenue', 'COGS'),
    lags=(7, 14),
    rolling_windows=(7, 14),
):
    train = train_df.copy()
    if date_col not in train.columns:
        detected = _detect_date_col(train)
        train = train.rename(columns={detected: date_col})

    train[date_col] = pd.to_datetime(train[date_col], errors='coerce')
    train = train[train[date_col].notna()].copy()

    if cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        train = train[train[date_col] <= cutoff_ts].copy()

    train = train.sort_values(date_col).reset_index(drop=True)

    train = add_time_features(train, date_col=date_col)
    train = add_external_features(train, data_path=data_path, date_col=date_col)
    train = add_lag_features(
        train,
        date_col=date_col,
        target_cols=target_cols,
        lags=lags,
        rolling_windows=rolling_windows,
    )

    # Drop early rows without lag context.
    lag_cols = [
        c for c in train.columns
        if any(c.startswith(f'{t}_lag_') or c.startswith(f'{t}_roll_mean_') for t in target_cols)
    ]
    if lag_cols:
        train = train.dropna(subset=lag_cols).reset_index(drop=True)

    forecast_features = None
    if forecast_df is not None:
        forecast = forecast_df.copy()
        if date_col not in forecast.columns:
            detected = _detect_date_col(forecast)
            forecast = forecast.rename(columns={detected: date_col})

        forecast[date_col] = pd.to_datetime(forecast[date_col], errors='coerce')
        forecast = forecast[forecast[date_col].notna()].sort_values(date_col).reset_index(drop=True)

        forecast = add_time_features(forecast, date_col=date_col)
        forecast = add_external_features(forecast, data_path=data_path, date_col=date_col)

        # Populate lag/rolling features for inference from most recent history.
        history = train.sort_values(date_col)
        for target in target_cols:
            if target not in history.columns:
                continue
            for lag in lags:
                col = f'{target}_lag_{lag}'
                value = history[target].iloc[-lag] if len(history) >= lag else history[target].iloc[-1]
                forecast[col] = value
            for window in rolling_windows:
                col = f'{target}_roll_mean_{window}'
                value = history[target].tail(window).mean()
                forecast[col] = value

        forecast_features = forecast

    feature_cols = [
        c for c in train.columns
        if c not in {date_col, *target_cols}
    ]

    if forecast_features is not None:
        feature_cols = [c for c in feature_cols if c in forecast_features.columns]

    # Median imputation for any leftover numeric nulls.
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            med = train[col].median()
            if pd.isna(med):
                med = 0
            train[col] = train[col].fillna(med)
            if forecast_features is not None and col in forecast_features.columns:
                forecast_features[col] = forecast_features[col].fillna(med)

    return {
        'train_features': train,
        'forecast_features': forecast_features,
        'feature_cols': feature_cols,
    }
