"""
DATATHON 2026 - GenCore v3 Ultimate AutoSearch
===============================================
Kaggle-ready script. Upload as notebook, attach dataset, Run All.

Architecture (from research):
  1. Prophet as feature extractor (trend, seasonality, holidays)
  2. LightGBM predicts residuals on top of Prophet
  3. Seasonal Naive 364 + Seasonal Window Median as anchors
  4. Auxiliary data as static historical aggregations (zero leakage)
  5. Fourier cyclical features (no ordinal boundary issues)
  6. Vietnamese Tet + Mega-sale calendar engineering
  7. Rolling year CV (2020, 2021, 2022) for profile selection
  8. Dynamic horizon-weighted blending (trust LGBM near, Prophet far)
  9. Portfolio: conservative / balanced / aggressive submissions
"""

import os
import sys
import glob
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ── Environment detection ────────────────────────────────────────────
KAGGLE = os.path.exists("/kaggle/input")
if KAGGLE:
    matches = glob.glob("/kaggle/input/**/sales.csv", recursive=True)
    if not matches:
        raise FileNotFoundError("sales.csv not found under /kaggle/input")
    DATA_DIR = os.path.dirname(matches[0])
    OUT_DIR = "/kaggle/working"
else:
    # Local: try repo structure
    for candidate in ["data/raw", "../data/raw"]:
        if os.path.isfile(os.path.join(candidate, "sales.csv")):
            DATA_DIR = candidate
            break
    else:
        DATA_DIR = "data/raw"
    OUT_DIR = "output"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"ENV: {'Kaggle' if KAGGLE else 'Local'}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"OUT_DIR: {OUT_DIR}")

# ── Dependencies ─────────────────────────────────────────────────────
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    if KAGGLE:
        os.system("pip install -q prophet")
        from prophet import Prophet
        HAS_PROPHET = True
    else:
        HAS_PROPHET = False
        print("WARNING: Prophet not available. Will use fallback.")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    if KAGGLE:
        os.system("pip install -q lightgbm")
        from lightgbm import LGBMRegressor
        HAS_LGB = True
    else:
        HAS_LGB = False
        print("WARNING: LightGBM not available.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING & AUXILIARY STATIC FEATURES
# ══════════════════════════════════════════════════════════════════════

def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load sales + sample_submission + auxiliary static aggregations."""
    sales = pd.read_csv(os.path.join(data_dir, "sales.csv"))
    sub_tpl = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    sub_tpl["Date"] = pd.to_datetime(sub_tpl["Date"], errors="coerce")
    sales = sales.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    sub_tpl = sub_tpl.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for c in ["Revenue", "COGS"]:
        if c not in sales.columns:
            raise ValueError(f"Missing column: {c}")

    # ── Auxiliary static aggregations (zero leakage) ──
    aux = {}

    # Web traffic: monthly avg sessions/visitors by month-of-year
    wt_path = os.path.join(data_dir, "web_traffic.csv")
    if os.path.isfile(wt_path):
        wt = pd.read_csv(wt_path)
        wt["date"] = pd.to_datetime(wt["date"], errors="coerce")
        wt = wt.dropna(subset=["date"])
        wt["month"] = wt["date"].dt.month
        # Static profile: avg sessions per calendar month across all years
        aux["web_monthly_sessions"] = wt.groupby("month")["sessions"].median().to_dict()
        aux["web_monthly_visitors"] = wt.groupby("month")["unique_visitors"].median().to_dict()
        aux["web_monthly_pageviews"] = wt.groupby("month")["page_views"].median().to_dict()
        # Day-of-week traffic profile
        wt["dow"] = wt["date"].dt.dayofweek
        aux["web_dow_sessions"] = wt.groupby("dow")["sessions"].median().to_dict()

    # Orders: daily order count aggregated by month-of-year
    ord_path = os.path.join(data_dir, "orders.csv")
    if os.path.isfile(ord_path):
        orders = pd.read_csv(ord_path)
        orders["order_date"] = pd.to_datetime(orders["order_date"], errors="coerce")
        orders = orders.dropna(subset=["order_date"])
        daily_orders = orders.groupby("order_date").size().reset_index(name="n_orders")
        daily_orders["month"] = daily_orders["order_date"].dt.month
        daily_orders["dow"] = daily_orders["order_date"].dt.dayofweek
        aux["orders_monthly"] = daily_orders.groupby("month")["n_orders"].median().to_dict()
        aux["orders_dow"] = daily_orders.groupby("dow")["n_orders"].median().to_dict()

    # Returns: return rate profile by month
    ret_path = os.path.join(data_dir, "returns.csv")
    if os.path.isfile(ret_path):
        returns = pd.read_csv(ret_path)
        returns["return_date"] = pd.to_datetime(returns["return_date"], errors="coerce")
        returns = returns.dropna(subset=["return_date"])
        returns["month"] = returns["return_date"].dt.month
        aux["returns_monthly_count"] = returns.groupby("month").size().to_dict()

    # Promotions: active promo density by month
    promo_path = os.path.join(data_dir, "promotions.csv")
    if os.path.isfile(promo_path):
        promos = pd.read_csv(promo_path)
        promos["start_date"] = pd.to_datetime(promos["start_date"], errors="coerce")
        promos = promos.dropna(subset=["start_date"])
        promos["month"] = promos["start_date"].dt.month
        aux["promo_monthly_count"] = promos.groupby("month").size().to_dict()
        aux["promo_avg_discount"] = promos.groupby("month")["discount_value"].mean().to_dict()

    print(f"Sales: {sales['Date'].min().date()} -> {sales['Date'].max().date()}, rows={len(sales)}")
    print(f"Forecast: {sub_tpl['Date'].min().date()} -> {sub_tpl['Date'].max().date()}, rows={len(sub_tpl)}")
    print(f"Auxiliary features loaded: {list(aux.keys())}")

    return sales, sub_tpl, aux


sales, sub_tpl, aux_profiles = load_data(DATA_DIR)
forecast_dates = sub_tpl["Date"].tolist()
N_FORECAST = len(forecast_dates)
print(f"N_FORECAST = {N_FORECAST}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: VIETNAMESE CALENDAR & HOLIDAY ENGINE
# ══════════════════════════════════════════════════════════════════════

# Tet dates (Lunar New Year Day 1) - verified from research
TET_DATES_STR = [
    "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19",
    "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
    "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
]
TET_DATES = pd.to_datetime(TET_DATES_STR)

# Vietnamese public holidays (fixed Gregorian dates)
VN_FIXED_HOLIDAYS = [
    (1, 1),   # New Year
    (4, 30),  # Reunification Day
    (5, 1),   # International Workers' Day
    (9, 2),   # National Day
]

# E-commerce mega-sale days (SEA pattern)
MEGA_SALE_DAYS = [
    (1, 1),   # New Year sale
    (3, 3),   # 3.3
    (4, 4),   # 4.4
    (5, 5),   # 5.5
    (6, 6),   # 6.6
    (7, 7),   # 7.7
    (8, 8),   # 8.8
    (9, 9),   # 9.9
    (10, 10), # 10.10
    (11, 11), # 11.11 Singles Day - biggest
    (12, 12), # 12.12
]


def build_prophet_holidays(
    last_date: pd.Timestamp,
    tet_lower: int = -21,
    tet_upper: int = 7,
) -> pd.DataFrame:
    """Build comprehensive holiday dataframe for Prophet."""
    rows = []

    # Tet holidays with pre/post windows
    for td in TET_DATES:
        if td <= last_date + pd.Timedelta(days=60):
            rows.append({
                "holiday": "tet",
                "ds": td,
                "lower_window": tet_lower,
                "upper_window": tet_upper,
            })

    # Mega-sale days
    for year in range(2012, last_date.year + 1):
        for m, d in MEGA_SALE_DAYS:
            dt = pd.Timestamp(year=year, month=m, day=d)
            if dt <= last_date:
                rows.append({
                    "holiday": f"mega_sale_{m}_{d}",
                    "ds": dt,
                    "lower_window": -3,
                    "upper_window": 2,
                })

        # Vietnamese fixed holidays
        for m, d in VN_FIXED_HOLIDAYS:
            dt = pd.Timestamp(year=year, month=m, day=d)
            if dt <= last_date:
                rows.append({
                    "holiday": "vn_holiday",
                    "ds": dt,
                    "lower_window": -1,
                    "upper_window": 1,
                })

    return pd.DataFrame(rows)


def _days_to_next(dates: pd.Series, events: np.ndarray, default: int = 365) -> np.ndarray:
    """Vectorized days-to-next-event calculation."""
    out = np.full(len(dates), default, dtype=int)
    ev = np.sort(events.astype("datetime64[ns]"))
    d_arr = dates.to_numpy().astype("datetime64[ns]")
    idx = np.searchsorted(ev, d_arr, side="left")
    for i in range(len(d_arr)):
        if idx[i] < len(ev):
            out[i] = int((ev[idx[i]] - d_arr[i]) / np.timedelta64(1, "D"))
    return out


def _days_since_last(dates: pd.Series, events: np.ndarray, default: int = 365) -> np.ndarray:
    """Vectorized days-since-last-event calculation."""
    out = np.full(len(dates), default, dtype=int)
    ev = np.sort(events.astype("datetime64[ns]"))
    d_arr = dates.to_numpy().astype("datetime64[ns]")
    idx = np.searchsorted(ev, d_arr, side="right") - 1
    for i in range(len(d_arr)):
        if idx[i] >= 0:
            out[i] = int((d_arr[i] - ev[idx[i]]) / np.timedelta64(1, "D"))
    return out


print("Holiday engine ready.")
print(f"Tet dates: {len(TET_DATES)}, Mega-sale patterns: {len(MEGA_SALE_DAYS)}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: SEASONAL NAIVE BASELINES
# ══════════════════════════════════════════════════════════════════════

def seasonal_naive_364(
    train_series: pd.Series,
    future_dates: Sequence[pd.Timestamp],
) -> np.ndarray:
    """
    Seasonal Naive with 364-day lookback (preserves day-of-week alignment).
    For days beyond 364, recursively uses own predictions.
    """
    series = train_series.sort_index().astype(float)
    hist = {pd.Timestamp(i): float(v) for i, v in series.items()}
    fallback = float(series.tail(28).median()) if len(series) >= 28 else float(series.median())
    preds = []

    for dt in pd.to_datetime(future_dates):
        dt = pd.Timestamp(dt)
        val = None
        # Try 364, then 371, then 357 (all preserve DOW)
        for offset in [364, 371, 357, 728]:
            c = dt - pd.Timedelta(days=offset)
            if c in hist:
                val = hist[c]
                break
        if val is None:
            val = fallback
        hist[dt] = float(val)
        preds.append(float(val))

    return np.asarray(preds, dtype=float)


def seasonal_window_median(
    train_df: pd.DataFrame,
    target_col: str,
    future_dates: Sequence[pd.Timestamp],
    window: int = 7,
) -> np.ndarray:
    """
    Rolling median of same-DOW, same-DOY-window historical values.
    Smooths out one-off anomalies from Seasonal Naive.
    """
    work = train_df[["Date", target_col]].copy()
    work["doy"] = work["Date"].dt.dayofyear
    work["dow"] = work["Date"].dt.dayofweek
    vals = work[target_col].to_numpy(dtype=float)
    doys = work["doy"].to_numpy(dtype=int)
    dows = work["dow"].to_numpy(dtype=int)
    fallback = float(np.nanmedian(vals))

    preds = []
    for dt in pd.to_datetime(future_dates):
        doy = int(dt.dayofyear)
        dow = int(dt.dayofweek)
        # Circular DOY distance
        diff = np.abs(doys - doy)
        dist = np.minimum(diff, 366 - diff)

        mask = (dows == dow) & (dist <= window)
        if mask.sum() < 3:
            mask = (dows == dow) & (dist <= window + 7)
        if mask.sum() < 3:
            mask = dist <= window

        preds.append(float(np.nanmedian(vals[mask])) if mask.sum() > 0 else fallback)

    return np.asarray(preds, dtype=float)


def trend_adjusted_naive(
    train_df: pd.DataFrame,
    target_col: str,
    naive_preds: np.ndarray,
    future_dates: Sequence[pd.Timestamp],
) -> np.ndarray:
    """
    Adjust naive predictions by year-over-year growth rate.
    If 2022 revenue was X% higher than 2021, scale naive by that factor.
    """
    yearly = train_df.groupby(train_df["Date"].dt.year)[target_col].sum()
    if len(yearly) >= 2:
        last_two = yearly.iloc[-2:]
        growth = last_two.iloc[-1] / max(last_two.iloc[-2], 1.0)
        # Dampen extreme growth rates
        growth = np.clip(growth, 0.85, 1.20)
    else:
        growth = 1.0

    return naive_preds * growth


print("Seasonal baselines ready.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: PROPHET MODEL (FEATURE EXTRACTOR)
# ══════════════════════════════════════════════════════════════════════

def fit_prophet(
    train_df: pd.DataFrame,
    target_col: str,
    holidays: pd.DataFrame,
    cps: float = 0.05,
    seasonality_mode: str = "multiplicative",
    yearly_fourier: int = 20,
) -> Tuple:
    """Fit Prophet model and return (model, cap, floor)."""
    if not HAS_PROPHET:
        return None, 0.0, 0.0

    df = train_df[["Date", target_col]].rename(
        columns={"Date": "ds", target_col: "y"}
    ).copy()

    cap = float(df["y"].quantile(0.995) * 1.25)
    floor = max(0.0, float(df["y"].quantile(0.005) * 0.75))
    df["cap"] = cap
    df["floor"] = floor

    m = Prophet(
        growth="logistic",
        holidays=holidays,
        yearly_seasonality=yearly_fourier,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_range=0.9,
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.add_seasonality(name="quarterly", period=91.25, fourier_order=3)

    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    m.fit(df)
    return m, cap, floor


def prophet_predict(
    model, dates: Sequence[pd.Timestamp], cap: float, floor: float
) -> pd.DataFrame:
    """Get Prophet predictions + components for given dates."""
    if model is None:
        # Fallback: return zeros
        out = pd.DataFrame({"Date": pd.to_datetime(list(dates))})
        for col in ["prophet_yhat", "prophet_trend", "prophet_weekly",
                     "prophet_yearly", "prophet_holidays", "prophet_monthly"]:
            out[col] = 0.0
        return out

    future = pd.DataFrame({"ds": pd.to_datetime(list(dates))})
    future["cap"] = cap
    future["floor"] = floor
    fc = model.predict(future)

    out = pd.DataFrame({"Date": future["ds"]})
    component_map = {
        "yhat": "prophet_yhat",
        "trend": "prophet_trend",
        "weekly": "prophet_weekly",
        "yearly": "prophet_yearly",
        "holidays": "prophet_holidays",
    }
    for src, dst in component_map.items():
        out[dst] = fc[src].values if src in fc.columns else 0.0

    if "monthly" in fc.columns:
        out["prophet_monthly"] = fc["monthly"].values
    else:
        out["prophet_monthly"] = 0.0

    return out


print("Prophet module ready.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: FEATURE ENGINEERING (ZERO LEAKAGE)
# ══════════════════════════════════════════════════════════════════════

def build_features(
    dates: Sequence[pd.Timestamp],
    target_history: pd.DataFrame,
    target_col: str,
    origin_date: pd.Timestamp,
    prophet_df: pd.DataFrame,
    aux: dict,
) -> pd.DataFrame:
    """
    Build feature matrix for LightGBM residual model.
    ALL features are deterministically known at forecast time.
    """
    feats = pd.DataFrame({"Date": pd.to_datetime(list(dates))})
    d = feats["Date"]

    # ── Calendar features ──
    feats["year"] = d.dt.year
    feats["month"] = d.dt.month
    feats["day"] = d.dt.day
    feats["dayofweek"] = d.dt.dayofweek
    feats["dayofyear"] = d.dt.dayofyear
    feats["weekofyear"] = d.dt.isocalendar().week.astype(int).values
    feats["quarter"] = d.dt.quarter
    feats["is_weekend"] = (feats["dayofweek"] >= 5).astype(int)
    feats["is_month_start"] = d.dt.is_month_start.astype(int)
    feats["is_month_end"] = d.dt.is_month_end.astype(int)
    feats["is_payday"] = feats["day"].isin([1, 15, 25]).astype(int)

    # ── Horizon & trend features ──
    hist_start = target_history["Date"].min()
    feats["days_since_start"] = (d - pd.Timestamp(hist_start)).dt.days.astype(int)
    feats["forecast_horizon"] = np.clip(
        (d - pd.Timestamp(origin_date)).dt.days.astype(int), 0, 600
    )

    # Piecewise linear trend (allows tree models to extrapolate)
    from sklearn.linear_model import LinearRegression
    hist_copy = target_history[["Date", target_col]].copy()
    hist_copy["t"] = (hist_copy["Date"] - hist_copy["Date"].min()).dt.days.astype(float)
    lr = LinearRegression()
    lr.fit(hist_copy[["t"]], hist_copy[target_col])
    feats_t = (d - pd.Timestamp(hist_start)).dt.days.astype(float).values.reshape(-1, 1)
    feats["linear_trend"] = lr.predict(feats_t)

    # ── Fourier cyclical features (no boundary issues) ──
    doy = feats["dayofyear"].values.astype(float)
    dow = feats["dayofweek"].values.astype(float)
    for k in range(1, 6):
        feats[f"sin_y{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        feats[f"cos_y{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for k in range(1, 4):
        feats[f"sin_w{k}"] = np.sin(2 * np.pi * k * dow / 7.0)
        feats[f"cos_w{k}"] = np.cos(2 * np.pi * k * dow / 7.0)
    # Monthly Fourier
    month_frac = feats["month"].values.astype(float)
    for k in range(1, 4):
        feats[f"sin_m{k}"] = np.sin(2 * np.pi * k * month_frac / 12.0)
        feats[f"cos_m{k}"] = np.cos(2 * np.pi * k * month_frac / 12.0)

    # ── Tet features ──
    tet_arr = TET_DATES.to_numpy()
    feats["days_to_tet"] = _days_to_next(d, tet_arr, default=365)
    feats["days_since_tet"] = _days_since_last(d, tet_arr, default=365)
    feats["is_pre_tet_30"] = feats["days_to_tet"].between(1, 30).astype(int)
    feats["is_pre_tet_14"] = feats["days_to_tet"].between(1, 14).astype(int)
    feats["is_pre_tet_7"] = feats["days_to_tet"].between(1, 7).astype(int)
    feats["is_tet_week"] = feats["days_to_tet"].between(-7, 0).astype(int)
    feats["is_post_tet_7"] = feats["days_since_tet"].between(1, 7).astype(int)
    feats["is_post_tet_14"] = feats["days_since_tet"].between(1, 14).astype(int)
    feats["tet_proximity"] = np.exp(-0.1 * np.minimum(
        feats["days_to_tet"].abs(), feats["days_since_tet"].abs()
    ))

    # ── Mega-sale features ──
    sale_dates_list = []
    max_year = max(d.dt.year.max(), 2024)
    for y in range(2012, max_year + 1):
        for m, dd in MEGA_SALE_DAYS:
            sale_dates_list.append(pd.Timestamp(year=y, month=m, day=dd))
    sale_arr = np.array(sorted(sale_dates_list), dtype="datetime64[ns]")

    feats["days_to_sale"] = _days_to_next(d, sale_arr, default=365)
    feats["days_since_sale"] = _days_since_last(d, sale_arr, default=365)
    feats["is_sale_window"] = (feats["days_to_sale"].abs() <= 3).astype(int)
    feats["is_pre_sale"] = feats["days_to_sale"].between(1, 3).astype(int)
    feats["is_post_sale"] = feats["days_since_sale"].between(1, 3).astype(int)
    feats["is_11_11"] = ((feats["month"] == 11) & (feats["day"] == 11)).astype(int)
    feats["is_12_12"] = ((feats["month"] == 12) & (feats["day"] == 12)).astype(int)

    # ── Historical DOY/DOW target profiles ──
    hist = target_history[["Date", target_col]].copy()
    hist["doy"] = hist["Date"].dt.dayofyear
    hist["dow"] = hist["Date"].dt.dayofweek
    hist["month"] = hist["Date"].dt.month

    doy_med = hist.groupby("doy")[target_col].median()
    dow_med = hist.groupby("dow")[target_col].median()
    month_med = hist.groupby("month")[target_col].median()

    global_med = float(hist[target_col].median())
    feats["hist_doy_target"] = feats["dayofyear"].map(doy_med).fillna(global_med)
    feats["hist_dow_target"] = feats["dayofweek"].map(dow_med).fillna(global_med)
    feats["hist_month_target"] = feats["month"].map(month_med).fillna(global_med)

    # Recent years profile (last 3 years)
    recent = hist[hist["Date"].dt.year >= (origin_date.year - 3)]
    if len(recent) > 100:
        doy_recent = recent.groupby("doy")[target_col].median()
        feats["hist_doy_recent"] = feats["dayofyear"].map(doy_recent).fillna(global_med)
    else:
        feats["hist_doy_recent"] = feats["hist_doy_target"]

    # ── Auxiliary static features (from load_data) ──
    if "web_monthly_sessions" in aux:
        feats["aux_web_sessions"] = feats["month"].map(aux["web_monthly_sessions"]).fillna(0)
    if "web_monthly_visitors" in aux:
        feats["aux_web_visitors"] = feats["month"].map(aux["web_monthly_visitors"]).fillna(0)
    if "web_monthly_pageviews" in aux:
        feats["aux_web_pageviews"] = feats["month"].map(aux["web_monthly_pageviews"]).fillna(0)
    if "web_dow_sessions" in aux:
        feats["aux_web_dow"] = feats["dayofweek"].map(aux["web_dow_sessions"]).fillna(0)
    if "orders_monthly" in aux:
        feats["aux_orders_monthly"] = feats["month"].map(aux["orders_monthly"]).fillna(0)
    if "orders_dow" in aux:
        feats["aux_orders_dow"] = feats["dayofweek"].map(aux["orders_dow"]).fillna(0)
    if "returns_monthly_count" in aux:
        feats["aux_returns_monthly"] = feats["month"].map(aux["returns_monthly_count"]).fillna(0)
    if "promo_monthly_count" in aux:
        feats["aux_promo_count"] = feats["month"].map(aux["promo_monthly_count"]).fillna(0)
    if "promo_avg_discount" in aux:
        feats["aux_promo_discount"] = feats["month"].map(aux["promo_avg_discount"]).fillna(0)

    # ── Prophet components as features ──
    feats = feats.merge(prophet_df, on="Date", how="left")

    # Fill NaN
    for col in feats.columns:
        if col == "Date":
            continue
        if feats[col].isna().any():
            feats[col] = feats[col].fillna(0)

    return feats.drop(columns=["Date"])


print("Feature engineering module ready.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: LIGHTGBM RESIDUAL MODEL
# ══════════════════════════════════════════════════════════════════════

def fit_lgbm_residual(
    X: pd.DataFrame,
    y_residual: np.ndarray,
    objective: str = "mae",
    n_estimators: int = 1200,
    learning_rate: float = 0.02,
    num_leaves: int = 31,
    reg_alpha: float = 0.5,
    reg_lambda: float = 5.0,
    recency_weight: bool = True,
) -> "LGBMRegressor":
    """
    Fit LightGBM to predict residuals (actual - prophet_yhat).
    Uses recency weighting so recent years matter more.
    """
    if not HAS_LGB:
        return None

    # Recency weighting: exponentially increasing weight for recent data
    if recency_weight:
        rank = np.arange(len(X), dtype=float)
        sample_weight = 1.0 + 2.0 * (rank / max(1.0, rank.max())) ** 1.5
    else:
        sample_weight = np.ones(len(X))

    params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=30,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )

    if objective == "quantile":
        params["objective"] = "quantile"
        params["alpha"] = 0.5
    elif objective == "huber":
        params["objective"] = "huber"
        params["alpha"] = 0.9
    else:
        params["objective"] = "mae"

    m = LGBMRegressor(**params)
    m.fit(X, y_residual, sample_weight=sample_weight)
    return m


print("LightGBM residual module ready.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: BLENDING & WEIGHT OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════

def optimize_blend_weights(
    y_true: np.ndarray,
    p_naive: np.ndarray,
    p_prophet: np.ndarray,
    p_hybrid: np.ndarray,
    step: float = 0.05,
) -> Tuple[Tuple[float, float, float], float]:
    """Grid search for optimal blend weights minimizing MAE."""
    best_w = (0.33, 0.33, 0.34)
    best_mae = 1e18

    for wn in np.arange(0.0, 0.65, step):
        for wp in np.arange(0.0, 0.85, step):
            wh = round(1.0 - wn - wp, 4)
            if wh < -0.01:
                continue
            wh = max(wh, 0.0)
            pred = wn * p_naive + wp * p_prophet + wh * p_hybrid
            mae = mean_absolute_error(y_true, pred)
            if mae < best_mae:
                best_mae = float(mae)
                best_w = (float(wn), float(wp), float(wh))

    return best_w, best_mae


def dynamic_horizon_blend(
    p_naive: np.ndarray,
    p_prophet: np.ndarray,
    p_hybrid: np.ndarray,
    weights: Tuple[float, float, float],
    dynamic: bool = True,
    decay_strength: float = 0.5,
) -> np.ndarray:
    """
    Horizon-weighted blending:
    - Near horizon: trust LightGBM hybrid more
    - Far horizon: shift trust to Prophet + Naive (more stable)
    """
    wn, wp, wh = weights

    if not dynamic or len(p_naive) < 10:
        pred = wn * p_naive + wp * p_prophet + wh * p_hybrid
        return np.clip(pred, 0.0, None)

    t = np.linspace(0.0, 1.0, len(p_naive))

    # Hybrid weight decays, Prophet weight grows
    wh_t = wh * (1.0 - decay_strength * t)
    wp_t = wp * (1.0 + decay_strength * 0.8 * t)
    wn_t = wn * (1.0 + decay_strength * 0.3 * t)

    denom = wh_t + wp_t + wn_t
    denom = np.where(denom < 0.01, 1.0, denom)

    pred = (wn_t * p_naive + wp_t * p_prophet + wh_t * p_hybrid) / denom
    return np.clip(pred, 0.0, None)


print("Blending module ready.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: SEARCH PROFILES & EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SearchProfile:
    name: str
    train_start_year: Optional[int]  # None = use all history
    tet_lower: int = -21
    tet_upper: int = 7
    cps: float = 0.05
    seasonality_mode: str = "multiplicative"
    yearly_fourier: int = 20
    lgb_objective: str = "mae"
    lgb_n_estimators: int = 1200
    lgb_lr: float = 0.02
    lgb_num_leaves: int = 31
    lgb_reg_alpha: float = 0.5
    lgb_reg_lambda: float = 5.0
    dynamic_blend: bool = True
    decay_strength: float = 0.5


# ── Define search space ──
PROFILES = [
    # Full history profiles
    SearchProfile("full_mult_cps005", None, -21, 7, 0.05, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),
    SearchProfile("full_mult_cps01", None, -21, 7, 0.10, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),
    SearchProfile("full_mult_cps003", None, -21, 7, 0.03, "multiplicative", 25, "mae", 1500, 0.015, 25, 0.3, 3.0, True, 0.6),
    SearchProfile("full_add_cps005", None, -21, 7, 0.05, "additive", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),
    SearchProfile("full_mult_huber", None, -21, 7, 0.05, "multiplicative", 20, "huber", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),
    SearchProfile("full_mult_quantile", None, -21, 7, 0.05, "multiplicative", 20, "quantile", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),

    # Recent history (2016+) profiles
    SearchProfile("recent16_mult", 2016, -21, 7, 0.05, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),
    SearchProfile("recent16_mult_cps01", 2016, -21, 7, 0.10, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),

    # Very recent (2018+) profiles
    SearchProfile("recent18_mult", 2018, -21, 7, 0.05, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),
    SearchProfile("recent18_mult_cps01", 2018, -21, 7, 0.10, "multiplicative", 20, "mae", 1000, 0.025, 31, 0.3, 3.0, True, 0.5),
    SearchProfile("recent18_add", 2018, -21, 7, 0.05, "additive", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),

    # Wider Tet window
    SearchProfile("full_tet_wide", None, -28, 10, 0.05, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),
    SearchProfile("recent18_tet_wide", 2018, -28, 10, 0.05, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, True, 0.5),

    # Static blend (no dynamic decay)
    SearchProfile("full_static", None, -21, 7, 0.05, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, False, 0.0),
    SearchProfile("recent18_static", 2018, -21, 7, 0.05, "multiplicative", 20, "mae", 1200, 0.02, 31, 0.5, 5.0, False, 0.0),

    # High regularization
    SearchProfile("full_highreg", None, -21, 7, 0.03, "multiplicative", 15, "mae", 800, 0.03, 20, 1.0, 10.0, True, 0.6),
    SearchProfile("recent18_highreg", 2018, -21, 7, 0.03, "multiplicative", 15, "mae", 800, 0.03, 20, 1.0, 10.0, True, 0.6),
]

print(f"Search space: {len(PROFILES)} profiles defined.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 9: CROSS-VALIDATION ENGINE
# ══════════════════════════════════════════════════════════════════════

def evaluate_profile(
    sales_df: pd.DataFrame,
    target_col: str,
    profile: SearchProfile,
    fold_years: Sequence[int],
    aux: dict,
) -> dict:
    """
    Evaluate a profile using rolling year CV.
    Returns dict with avg MAE for each component and best blend weights.
    """
    fold_results = []

    for year in fold_years:
        # Train: everything before this year
        train = sales_df[sales_df["Date"].dt.year < year].copy()
        # Validation: this entire year
        valid = sales_df[sales_df["Date"].dt.year == year].copy()

        if profile.train_start_year is not None:
            train = train[train["Date"].dt.year >= profile.train_start_year].copy()

        if len(train) < 365 or len(valid) < 30:
            continue

        val_dates = valid["Date"].tolist()
        y_val = valid[target_col].to_numpy(dtype=float)
        origin = train["Date"].max()

        # ── Seasonal Naive ──
        p364 = seasonal_naive_364(train.set_index("Date")[target_col], val_dates)
        pwin = seasonal_window_median(train, target_col, val_dates, window=7)
        p_naive = 0.5 * p364 + 0.5 * pwin
        p_naive = trend_adjusted_naive(train, target_col, p_naive, val_dates)

        # ── Prophet ──
        holidays = build_prophet_holidays(valid["Date"].max(), profile.tet_lower, profile.tet_upper)
        m_prophet, cap, floor = fit_prophet(
            train, target_col, holidays,
            cps=profile.cps,
            seasonality_mode=profile.seasonality_mode,
            yearly_fourier=profile.yearly_fourier,
        )
        pr_train = prophet_predict(m_prophet, train["Date"].tolist(), cap, floor)
        pr_val = prophet_predict(m_prophet, val_dates, cap, floor)
        p_prophet = pr_val["prophet_yhat"].to_numpy(dtype=float)

        # ── LightGBM Residual ──
        residual_train = train[target_col].to_numpy(dtype=float) - pr_train["prophet_yhat"].to_numpy(dtype=float)

        X_train = build_features(
            train["Date"].tolist(), train[["Date", target_col]],
            target_col, origin, pr_train, aux,
        )
        X_val = build_features(
            val_dates, train[["Date", target_col]],
            target_col, origin, pr_val, aux,
        )

        lgb_model = fit_lgbm_residual(
            X_train, residual_train,
            objective=profile.lgb_objective,
            n_estimators=profile.lgb_n_estimators,
            learning_rate=profile.lgb_lr,
            num_leaves=profile.lgb_num_leaves,
            reg_alpha=profile.lgb_reg_alpha,
            reg_lambda=profile.lgb_reg_lambda,
        )

        if lgb_model is not None:
            residual_pred = lgb_model.predict(X_val)
            p_hybrid = np.clip(p_prophet + residual_pred, 0.0, None)
        else:
            p_hybrid = p_prophet.copy()

        # ── Optimize blend ──
        best_w, best_mae = optimize_blend_weights(y_val, p_naive, p_prophet, p_hybrid)

        # Also evaluate dynamic blend
        p_dynamic = dynamic_horizon_blend(
            p_naive, p_prophet, p_hybrid, best_w,
            dynamic=profile.dynamic_blend,
            decay_strength=profile.decay_strength,
        )
        dynamic_mae = mean_absolute_error(y_val, p_dynamic)

        fold_results.append({
            "year": year,
            "naive_mae": float(mean_absolute_error(y_val, p_naive)),
            "prophet_mae": float(mean_absolute_error(y_val, p_prophet)),
            "hybrid_mae": float(mean_absolute_error(y_val, p_hybrid)),
            "blend_mae": float(best_mae),
            "dynamic_mae": float(dynamic_mae),
            "weights": best_w,
        })

    if not fold_results:
        return {"profile": profile.name, "avg_mae": 1e18, "folds": []}

    # Average across folds (weighted: more recent folds matter more)
    n = len(fold_results)
    fold_w = np.array([1.0 + i * 0.5 for i in range(n)])
    fold_w /= fold_w.sum()

    avg_blend = sum(r["blend_mae"] * w for r, w in zip(fold_results, fold_w))
    avg_dynamic = sum(r["dynamic_mae"] * w for r, w in zip(fold_results, fold_w))
    best_overall = min(avg_blend, avg_dynamic)

    # Average weights
    avg_weights = tuple(
        float(np.mean([r["weights"][i] for r in fold_results]))
        for i in range(3)
    )

    return {
        "profile": profile.name,
        "avg_mae": float(best_overall),
        "avg_blend_mae": float(avg_blend),
        "avg_dynamic_mae": float(avg_dynamic),
        "avg_weights": avg_weights,
        "folds": fold_results,
    }


print("CV engine ready.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 10: AUTO-SEARCH EXECUTION
# ══════════════════════════════════════════════════════════════════════

FOLD_YEARS = [2020, 2021, 2022]

print("=" * 70)
print("STARTING AUTO-SEARCH")
print(f"Profiles: {len(PROFILES)}, Fold years: {FOLD_YEARS}")
print(f"Targets: Revenue, COGS")
print("=" * 70)

all_results = {}

for target_col in ["Revenue", "COGS"]:
    print(f"\n{'─' * 50}")
    print(f"TARGET: {target_col}")
    print(f"{'─' * 50}")

    target_results = []

    for i, profile in enumerate(PROFILES):
        t0 = time.time()
        print(f"  [{i+1}/{len(PROFILES)}] {profile.name} ...", end=" ", flush=True)

        try:
            result = evaluate_profile(sales, target_col, profile, FOLD_YEARS, aux_profiles)
            result["time_sec"] = time.time() - t0
            target_results.append(result)
            print(f"MAE={result['avg_mae']:,.0f} ({result['time_sec']:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            target_results.append({
                "profile": profile.name,
                "avg_mae": 1e18,
                "error": str(e),
            })

    # Sort by MAE
    target_results.sort(key=lambda x: x["avg_mae"])
    all_results[target_col] = target_results

    print(f"\n  TOP 5 for {target_col}:")
    for j, r in enumerate(target_results[:5]):
        w_str = ""
        if "avg_weights" in r:
            w = r["avg_weights"]
            w_str = f" [naive={w[0]:.2f}, prophet={w[1]:.2f}, hybrid={w[2]:.2f}]"
        print(f"    {j+1}. {r['profile']}: MAE={r['avg_mae']:,.0f}{w_str}")

print("\n" + "=" * 70)
print("AUTO-SEARCH COMPLETE")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# SECTION 11: FINAL FORECAST GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_final_forecast(
    sales_df: pd.DataFrame,
    target_col: str,
    profile: SearchProfile,
    forecast_dates: Sequence[pd.Timestamp],
    weights: Tuple[float, float, float],
    aux: dict,
    variant: str = "balanced",
) -> np.ndarray:
    """Generate final 548-day forecast using best profile."""
    train = sales_df.copy()
    if profile.train_start_year is not None:
        train = train[train["Date"].dt.year >= profile.train_start_year].copy()

    origin = train["Date"].max()

    # ── Seasonal Naive ──
    p364 = seasonal_naive_364(train.set_index("Date")[target_col], forecast_dates)
    pwin = seasonal_window_median(train, target_col, forecast_dates, window=7)
    p_naive = 0.5 * p364 + 0.5 * pwin
    p_naive = trend_adjusted_naive(train, target_col, p_naive, forecast_dates)

    # ── Prophet ──
    last_fc = pd.Timestamp(max(forecast_dates))
    holidays = build_prophet_holidays(last_fc, profile.tet_lower, profile.tet_upper)
    m_prophet, cap, floor = fit_prophet(
        train, target_col, holidays,
        cps=profile.cps,
        seasonality_mode=profile.seasonality_mode,
        yearly_fourier=profile.yearly_fourier,
    )
    pr_train = prophet_predict(m_prophet, train["Date"].tolist(), cap, floor)
    pr_fc = prophet_predict(m_prophet, forecast_dates, cap, floor)
    p_prophet = pr_fc["prophet_yhat"].to_numpy(dtype=float)

    # ── LightGBM Residual ──
    residual_train = train[target_col].to_numpy(dtype=float) - pr_train["prophet_yhat"].to_numpy(dtype=float)
    X_train = build_features(
        train["Date"].tolist(), train[["Date", target_col]],
        target_col, origin, pr_train, aux,
    )
    X_fc = build_features(
        forecast_dates, train[["Date", target_col]],
        target_col, origin, pr_fc, aux,
    )

    lgb_model = fit_lgbm_residual(
        X_train, residual_train,
        objective=profile.lgb_objective,
        n_estimators=profile.lgb_n_estimators,
        learning_rate=profile.lgb_lr,
        num_leaves=profile.lgb_num_leaves,
        reg_alpha=profile.lgb_reg_alpha,
        reg_lambda=profile.lgb_reg_lambda,
    )

    if lgb_model is not None:
        residual_pred = lgb_model.predict(X_fc)
        p_hybrid = np.clip(p_prophet + residual_pred, 0.0, None)
    else:
        p_hybrid = p_prophet.copy()

    # ── Variant-specific blending ──
    wn, wp, wh = weights

    if variant == "conservative":
        # More Prophet + Naive, less hybrid, stronger decay
        wn_adj = wn * 1.15
        wp_adj = wp * 1.10
        wh_adj = wh * 0.75
        total = wn_adj + wp_adj + wh_adj
        weights_adj = (wn_adj / total, wp_adj / total, wh_adj / total)
        decay = profile.decay_strength * 1.3
    elif variant == "aggressive":
        # More hybrid, less decay
        wn_adj = wn * 0.85
        wp_adj = wp * 0.90
        wh_adj = wh * 1.25
        total = wn_adj + wp_adj + wh_adj
        weights_adj = (wn_adj / total, wp_adj / total, wh_adj / total)
        decay = profile.decay_strength * 0.6
    else:  # balanced
        weights_adj = weights
        decay = profile.decay_strength

    pred = dynamic_horizon_blend(
        p_naive, p_prophet, p_hybrid, weights_adj,
        dynamic=profile.dynamic_blend,
        decay_strength=decay,
    )

    return np.clip(pred, 0.0, None)


# ── Select best profiles and generate forecasts ──
print("\n" + "=" * 70)
print("GENERATING FINAL FORECASTS")
print("=" * 70)

# Find best profile for each target
best_profiles = {}
best_weights = {}

for target_col in ["Revenue", "COGS"]:
    results = all_results[target_col]
    best = results[0]  # Already sorted by MAE
    profile_name = best["profile"]

    # Find the actual profile object
    profile_obj = next(p for p in PROFILES if p.name == profile_name)
    best_profiles[target_col] = profile_obj
    best_weights[target_col] = best.get("avg_weights", (0.3, 0.4, 0.3))

    print(f"\n{target_col}: Best profile = {profile_name}")
    print(f"  CV MAE = {best['avg_mae']:,.0f}")
    print(f"  Weights = {best_weights[target_col]}")

# Generate 3 variants
variants = ["conservative", "balanced", "aggressive"]
submissions = {}

for variant in variants:
    print(f"\n── Generating {variant} submission ──")
    sub = sub_tpl[["Date"]].copy()

    for target_col in ["Revenue", "COGS"]:
        print(f"  {target_col}...", end=" ", flush=True)
        t0 = time.time()

        preds = generate_final_forecast(
            sales, target_col,
            best_profiles[target_col],
            forecast_dates,
            best_weights[target_col],
            aux_profiles,
            variant=variant,
        )
        sub[target_col] = preds
        print(f"done ({time.time()-t0:.1f}s)")

    submissions[variant] = sub

print("\nAll variants generated.")


# ══════════════════════════════════════════════════════════════════════
# SECTION 12: VALIDATION, DIAGNOSTICS & OUTPUT
# ══════════════════════════════════════════════════════════════════════

def validate_submission(sub: pd.DataFrame, expected_rows: int) -> bool:
    """Validate submission format."""
    ok = True
    if len(sub) != expected_rows:
        print(f"  ❌ Row count: {len(sub)} (expected {expected_rows})")
        ok = False
    for col in ["Date", "Revenue", "COGS"]:
        if col not in sub.columns:
            print(f"  ❌ Missing column: {col}")
            ok = False
    if sub[["Revenue", "COGS"]].isna().any().any():
        print(f"  ❌ NaN values found")
        ok = False
    if (sub[["Revenue", "COGS"]] < 0).any().any():
        print(f"  ❌ Negative values found")
        ok = False
    if ok:
        print(f"  ✅ Valid ({expected_rows} rows, no NaN, no negatives)")
    return ok


print("\n" + "=" * 70)
print("VALIDATION & OUTPUT")
print("=" * 70)

output_files = {}

for variant in variants:
    sub = submissions[variant]
    fname = f"submission_v3_{variant}.csv"
    fpath = os.path.join(OUT_DIR, fname)

    print(f"\n{variant.upper()}:")
    validate_submission(sub, N_FORECAST)

    # Stats
    for col in ["Revenue", "COGS"]:
        vals = sub[col]
        print(f"  {col}: mean={vals.mean():,.0f}, median={vals.median():,.0f}, "
              f"min={vals.min():,.0f}, max={vals.max():,.0f}, std={vals.std():,.0f}")

    # Save
    out_df = sub.copy()
    out_df["Date"] = out_df["Date"].dt.strftime("%Y-%m-%d")
    out_df.to_csv(fpath, index=False)
    output_files[variant] = fpath
    print(f"  Saved: {fpath}")

# ── Diagnostics JSON ──
diagnostics = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "n_profiles_searched": len(PROFILES),
    "fold_years": FOLD_YEARS,
    "best_profiles": {},
    "all_results_summary": {},
}

for target_col in ["Revenue", "COGS"]:
    bp = best_profiles[target_col]
    diagnostics["best_profiles"][target_col] = {
        "name": bp.name,
        "train_start_year": bp.train_start_year,
        "cps": bp.cps,
        "seasonality_mode": bp.seasonality_mode,
        "lgb_objective": bp.lgb_objective,
        "cv_mae": all_results[target_col][0]["avg_mae"],
        "weights": list(best_weights[target_col]),
    }
    diagnostics["all_results_summary"][target_col] = [
        {"profile": r["profile"], "mae": r["avg_mae"]}
        for r in all_results[target_col][:10]
    ]

diag_path = os.path.join(OUT_DIR, "v3_ultimate_diagnostics.json")
with open(diag_path, "w") as f:
    json.dump(diagnostics, f, indent=2, default=str)
print(f"\nDiagnostics saved: {diag_path}")

# ── Summary ──
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
for target_col in ["Revenue", "COGS"]:
    bp = best_profiles[target_col]
    mae = all_results[target_col][0]["avg_mae"]
    print(f"  {target_col}: {bp.name} (CV MAE = {mae:,.0f})")

print(f"\nOutput files:")
for variant, fpath in output_files.items():
    print(f"  {variant}: {fpath}")

print(f"\nDiagnostics: {diag_path}")
print("\n🏁 DONE. Upload submission_v3_balanced.csv to Kaggle first.")
print("   If public LB improves, try aggressive. If drops, use conservative.")
