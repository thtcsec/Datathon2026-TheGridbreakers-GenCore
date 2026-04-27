"""
DATATHON 2026 - GenCore v4 TopKill
====================================
Key changes from v3 (addressing CV-LB gap of 400k):

1. AGGRESSIVE HORIZON DECAY: LightGBM hybrid weight drops to near-zero
   after day 90. Research says tree models degrade catastrophically
   beyond 300 days. v3 still gave hybrid 55% weight uniformly.

2. THETA METHOD as 4th ensemble member (research: "remarkably robust,
   computationally cheap, prevents wild divergence")

3. MULTIPLE PROPHET CONFIGS blended (not just one) - reduces Prophet
   variance which dominates long-horizon predictions

4. PURE PROPHET SUBMISSION as a separate variant - research says
   Prophet alone often beats complex hybrids on 500+ day horizons

5. SMARTER TET CALIBRATION using actual historical Tet patterns
   (v3 used generic windows, v4 uses empirical Tet multipliers)

6. TRAIN ON 2016+ ONLY for LightGBM (2012-2015 data is from a
   structurally different business era - 2018 revenue was 2x 2019+)

7. REDUCED PROFILE COUNT - only run proven-good configs from v3
   search results, spend compute on ensemble diversity instead
"""

import os
import sys
import glob
import json
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ── Environment ──────────────────────────────────────────────────────
KAGGLE = os.path.exists("/kaggle/input")
if KAGGLE:
    matches = glob.glob("/kaggle/input/**/sales.csv", recursive=True)
    if not matches:
        raise FileNotFoundError("sales.csv not found under /kaggle/input")
    DATA_DIR = os.path.dirname(matches[0])
    OUT_DIR = "/kaggle/working"
else:
    for candidate in ["data/raw", "../data/raw"]:
        if os.path.isfile(os.path.join(candidate, "sales.csv")):
            DATA_DIR = candidate
            break
    else:
        DATA_DIR = "data/raw"
    OUT_DIR = "output"

os.makedirs(OUT_DIR, exist_ok=True)
print(f"ENV: {'Kaggle' if KAGGLE else 'Local'}")
print(f"DATA_DIR: {DATA_DIR} | OUT_DIR: {OUT_DIR}")

# ── Dependencies ─────────────────────────────────────────────────────
try:
    from prophet import Prophet
except ImportError:
    if KAGGLE:
        os.system("pip install -q prophet")
        from prophet import Prophet
    else:
        raise

try:
    from lightgbm import LGBMRegressor
except ImportError:
    if KAGGLE:
        os.system("pip install -q lightgbm")
        from lightgbm import LGBMRegressor
    else:
        raise

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
sub_tpl["Date"] = pd.to_datetime(sub_tpl["Date"], errors="coerce")
sales = sales.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
sub_tpl = sub_tpl.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

forecast_dates = sub_tpl["Date"].tolist()
N_FC = len(forecast_dates)

print(f"Train: {sales['Date'].min().date()} -> {sales['Date'].max().date()} ({len(sales)} rows)")
print(f"Forecast: {forecast_dates[0].date()} -> {forecast_dates[-1].date()} ({N_FC} rows)")

# ── Auxiliary static profiles ──
aux = {}
for fname, date_col, agg_cols in [
    ("web_traffic.csv", "date", ["sessions", "unique_visitors", "page_views"]),
    ("orders.csv", "order_date", []),
]:
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.isfile(fpath):
        continue
    df = pd.read_csv(fpath)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["month"] = df[date_col].dt.month
    df["dow"] = df[date_col].dt.dayofweek
    if agg_cols:
        for col in agg_cols:
            if col in df.columns:
                aux[f"{col}_month"] = df.groupby("month")[col].median().to_dict()
                aux[f"{col}_dow"] = df.groupby("dow")[col].median().to_dict()
    else:
        daily = df.groupby(date_col).size().reset_index(name="n")
        daily["month"] = daily[date_col].dt.month
        daily["dow"] = daily[date_col].dt.dayofweek
        aux["orders_month"] = daily.groupby("month")["n"].median().to_dict()
        aux["orders_dow"] = daily.groupby("dow")["n"].median().to_dict()

print(f"Auxiliary profiles: {len(aux)} features")


# ══════════════════════════════════════════════════════════════════════
# VIETNAMESE CALENDAR ENGINE
# ══════════════════════════════════════════════════════════════════════

TET_DATES = pd.to_datetime([
    "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19",
    "2016-02-08", "2017-01-28", "2018-02-16", "2019-02-05",
    "2020-01-25", "2021-02-12", "2022-02-01", "2023-01-22", "2024-02-10",
])

MEGA_SALES = [(1,1),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12)]
VN_HOLIDAYS = [(1,1),(4,30),(5,1),(9,2)]


def build_holidays_df(last_date, tet_lower=-21, tet_upper=7):
    rows = []
    for td in TET_DATES:
        if td <= last_date + pd.Timedelta(days=60):
            rows.append({"holiday": "tet", "ds": td,
                         "lower_window": tet_lower, "upper_window": tet_upper})
    for y in range(2012, last_date.year + 1):
        for m, d in MEGA_SALES:
            dt = pd.Timestamp(year=y, month=m, day=d)
            if dt <= last_date:
                rows.append({"holiday": f"sale_{m}_{d}", "ds": dt,
                             "lower_window": -3, "upper_window": 2})
        for m, d in VN_HOLIDAYS:
            dt = pd.Timestamp(year=y, month=m, day=d)
            if dt <= last_date:
                rows.append({"holiday": "vn_hol", "ds": dt,
                             "lower_window": -1, "upper_window": 1})
    return pd.DataFrame(rows)


def days_to_next(dates, events, default=365):
    ev = np.sort(np.array(events, dtype="datetime64[ns]"))
    d = dates.to_numpy().astype("datetime64[ns]")
    out = np.full(len(d), default, dtype=int)
    idx = np.searchsorted(ev, d, side="left")
    for i in range(len(d)):
        if idx[i] < len(ev):
            out[i] = int((ev[idx[i]] - d[i]) / np.timedelta64(1, "D"))
    return out


def days_since_last(dates, events, default=365):
    ev = np.sort(np.array(events, dtype="datetime64[ns]"))
    d = dates.to_numpy().astype("datetime64[ns]")
    out = np.full(len(d), default, dtype=int)
    idx = np.searchsorted(ev, d, side="right") - 1
    for i in range(len(d)):
        if idx[i] >= 0:
            out[i] = int((d[i] - ev[idx[i]]) / np.timedelta64(1, "D"))
    return out


# ── Empirical Tet multipliers from actual data ──
# Compute how revenue behaves relative to yearly median around each Tet
def compute_tet_multipliers(sales_df, target_col):
    """Compute empirical multiplier curve around Tet from historical data."""
    multipliers = {}  # days_relative_to_tet -> list of multipliers
    for tet in TET_DATES:
        year_data = sales_df[sales_df["Date"].dt.year == tet.year]
        if len(year_data) < 100:
            continue
        yearly_med = year_data[target_col].median()
        if yearly_med <= 0:
            continue
        for delta in range(-30, 21):
            d = tet + pd.Timedelta(days=delta)
            row = sales_df[sales_df["Date"] == d]
            if len(row) > 0:
                mult = row[target_col].iloc[0] / yearly_med
                multipliers.setdefault(delta, []).append(mult)

    # Median multiplier for each relative day
    return {k: float(np.median(v)) for k, v in multipliers.items()}


tet_mult_rev = compute_tet_multipliers(sales, "Revenue")
tet_mult_cogs = compute_tet_multipliers(sales, "COGS")
print(f"Tet multipliers computed: {len(tet_mult_rev)} days for Revenue")


# ══════════════════════════════════════════════════════════════════════
# BASELINE MODELS
# ══════════════════════════════════════════════════════════════════════

def seasonal_naive_364(train_series, future_dates):
    """Seasonal Naive with 364-day lookback (DOW-preserving)."""
    series = train_series.sort_index().astype(float)
    hist = {pd.Timestamp(i): float(v) for i, v in series.items()}
    fb = float(series.tail(28).median())
    preds = []
    for dt in pd.to_datetime(future_dates):
        dt = pd.Timestamp(dt)
        val = None
        for off in [364, 371, 357, 728]:
            c = dt - pd.Timedelta(days=off)
            if c in hist:
                val = hist[c]
                break
        if val is None:
            val = fb
        hist[dt] = float(val)
        preds.append(float(val))
    return np.array(preds)


def seasonal_window_median(train_df, target_col, future_dates, window=7):
    """Rolling median of same-DOW, same-DOY-window values."""
    work = train_df[["Date", target_col]].copy()
    work["doy"] = work["Date"].dt.dayofyear
    work["dow"] = work["Date"].dt.dayofweek
    vals = work[target_col].values.astype(float)
    doys = work["doy"].values
    dows = work["dow"].values
    fb = float(np.nanmedian(vals))
    preds = []
    for dt in pd.to_datetime(future_dates):
        doy, dow = int(dt.dayofyear), int(dt.dayofweek)
        diff = np.abs(doys - doy)
        dist = np.minimum(diff, 366 - diff)
        mask = (dows == dow) & (dist <= window)
        if mask.sum() < 3:
            mask = (dows == dow) & (dist <= window + 7)
        if mask.sum() < 3:
            mask = dist <= window
        preds.append(float(np.nanmedian(vals[mask])) if mask.sum() > 0 else fb)
    return np.array(preds)


def theta_forecast(train_series, n_periods):
    """
    Theta method: blend linear trend + SES on deseasonalized data.
    Research: "remarkably robust, computationally cheap"
    """
    y = train_series.values.astype(float)
    n = len(y)

    # Deseasonalize with 7-day moving average
    if n > 14:
        ma7 = pd.Series(y).rolling(7, center=True, min_periods=1).mean().values
        seasonal = np.zeros(7)
        for d in range(7):
            idx = np.arange(d, n, 7)
            seasonal[d] = np.median(y[idx] / np.maximum(ma7[idx], 1.0))
        seasonal_full = np.tile(seasonal, (n // 7) + 2)[:n]
        deseas = y / np.maximum(seasonal_full, 0.01)
    else:
        deseas = y.copy()
        seasonal = np.ones(7)

    # Theta line 1: Linear regression (long-term trend)
    t = np.arange(n, dtype=float).reshape(-1, 1)
    lr = LinearRegression().fit(t, deseas)
    t_future = np.arange(n, n + n_periods, dtype=float).reshape(-1, 1)
    theta1 = lr.predict(t_future)

    # Theta line 2: SES on deseasonalized (alpha=0.2 for stability)
    alpha = 0.2
    ses = deseas[-1]
    theta2 = np.zeros(n_periods)
    for i in range(n_periods):
        theta2[i] = ses
        # SES doesn't update in forecast mode, stays flat

    # Blend: 50/50
    deseas_fc = 0.5 * theta1.flatten() + 0.5 * theta2

    # Re-seasonalize
    seasonal_fc = np.tile(seasonal, (n_periods // 7) + 2)[:n_periods]
    forecast = deseas_fc * seasonal_fc

    return np.clip(forecast, 0.0, None)


print("Baselines ready (Naive364, WindowMedian, Theta)")


# ══════════════════════════════════════════════════════════════════════
# PROPHET ENSEMBLE (MULTIPLE CONFIGS)
# ══════════════════════════════════════════════════════════════════════

def fit_prophet_config(train_df, target_col, holidays, cps, smode, yearly_fo):
    """Fit one Prophet configuration."""
    df = train_df[["Date", target_col]].rename(
        columns={"Date": "ds", target_col: "y"}).copy()
    cap = float(df["y"].quantile(0.995) * 1.25)
    floor = max(0.0, float(df["y"].quantile(0.005) * 0.75))
    df["cap"] = cap
    df["floor"] = floor

    m = Prophet(
        growth="logistic", holidays=holidays,
        yearly_seasonality=yearly_fo, weekly_seasonality=True,
        daily_seasonality=False, seasonality_mode=smode,
        changepoint_prior_scale=cps, seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0, changepoint_range=0.9,
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
    m.fit(df)
    return m, cap, floor


def prophet_predict(model, dates, cap, floor):
    """Get Prophet yhat + components."""
    future = pd.DataFrame({"ds": pd.to_datetime(list(dates))})
    future["cap"] = cap
    future["floor"] = floor
    fc = model.predict(future)
    out = pd.DataFrame({"Date": future["ds"]})
    for col in ["yhat", "trend", "weekly", "yearly", "holidays"]:
        out[f"prophet_{col}"] = fc[col].values if col in fc.columns else 0.0
    if "monthly" in fc.columns:
        out["prophet_monthly"] = fc["monthly"].values
    else:
        out["prophet_monthly"] = 0.0
    return out


# Prophet configs to ensemble (diverse but proven)
PROPHET_CONFIGS = [
    {"cps": 0.03, "smode": "multiplicative", "yearly_fo": 15},  # conservative
    {"cps": 0.05, "smode": "multiplicative", "yearly_fo": 20},  # balanced
    {"cps": 0.10, "smode": "multiplicative", "yearly_fo": 20},  # flexible
    {"cps": 0.05, "smode": "additive", "yearly_fo": 20},        # additive variant
]

print(f"Prophet ensemble: {len(PROPHET_CONFIGS)} configs")


# ══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & LIGHTGBM (CONSERVATIVE ROLE)
# ══════════════════════════════════════════════════════════════════════

def build_features(dates, target_history, target_col, origin, prophet_df, aux_dict):
    """Build zero-leakage feature matrix."""
    feats = pd.DataFrame({"Date": pd.to_datetime(list(dates))})
    d = feats["Date"]

    feats["month"] = d.dt.month
    feats["day"] = d.dt.day
    feats["dayofweek"] = d.dt.dayofweek
    feats["dayofyear"] = d.dt.dayofyear
    feats["weekofyear"] = d.dt.isocalendar().week.astype(int).values
    feats["quarter"] = d.dt.quarter
    feats["is_weekend"] = (feats["dayofweek"] >= 5).astype(int)
    feats["is_month_start"] = d.dt.is_month_start.astype(int)
    feats["is_month_end"] = d.dt.is_month_end.astype(int)

    hist_start = target_history["Date"].min()
    feats["days_since_start"] = (d - pd.Timestamp(hist_start)).dt.days
    feats["forecast_horizon"] = np.clip((d - pd.Timestamp(origin)).dt.days, 0, 600)

    # Linear trend
    h = target_history[["Date", target_col]].copy()
    h["t"] = (h["Date"] - h["Date"].min()).dt.days.astype(float)
    lr = LinearRegression().fit(h[["t"]], h[target_col])
    ft = (d - pd.Timestamp(hist_start)).dt.days.astype(float).values.reshape(-1, 1)
    feats["linear_trend"] = lr.predict(ft)

    # Fourier
    doy = feats["dayofyear"].values.astype(float)
    dow = feats["dayofweek"].values.astype(float)
    for k in range(1, 6):
        feats[f"sin_y{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        feats[f"cos_y{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for k in range(1, 4):
        feats[f"sin_w{k}"] = np.sin(2 * np.pi * k * dow / 7.0)
        feats[f"cos_w{k}"] = np.cos(2 * np.pi * k * dow / 7.0)

    # Tet
    tet_arr = TET_DATES.to_numpy()
    feats["days_to_tet"] = days_to_next(d, tet_arr)
    feats["days_since_tet"] = days_since_last(d, tet_arr)
    feats["is_pre_tet_21"] = feats["days_to_tet"].between(1, 21).astype(int)
    feats["is_pre_tet_7"] = feats["days_to_tet"].between(1, 7).astype(int)
    feats["is_tet_week"] = feats["days_to_tet"].between(-7, 0).astype(int)
    feats["is_post_tet_14"] = feats["days_since_tet"].between(1, 14).astype(int)
    feats["tet_proximity"] = np.exp(-0.1 * np.minimum(
        feats["days_to_tet"].abs(), feats["days_since_tet"].abs()))

    # Mega-sale
    sale_list = []
    for y in range(2012, 2025):
        for m, dd in MEGA_SALES:
            sale_list.append(pd.Timestamp(year=y, month=m, day=dd))
    sale_arr = np.array(sorted(sale_list), dtype="datetime64[ns]")
    feats["days_to_sale"] = days_to_next(d, sale_arr)
    feats["is_sale_window"] = (feats["days_to_sale"].abs() <= 3).astype(int)
    feats["is_11_11"] = ((feats["month"] == 11) & (feats["day"] == 11)).astype(int)
    feats["is_12_12"] = ((feats["month"] == 12) & (feats["day"] == 12)).astype(int)

    # Historical profiles
    hist = target_history[["Date", target_col]].copy()
    hist["doy"] = hist["Date"].dt.dayofyear
    hist["dow"] = hist["Date"].dt.dayofweek
    hist["month"] = hist["Date"].dt.month
    gmed = float(hist[target_col].median())
    feats["hist_doy"] = feats["dayofyear"].map(hist.groupby("doy")[target_col].median()).fillna(gmed)
    feats["hist_dow"] = feats["dayofweek"].map(hist.groupby("dow")[target_col].median()).fillna(gmed)
    feats["hist_month"] = feats["month"].map(hist.groupby("month")[target_col].median()).fillna(gmed)

    # Aux
    for key, mapping in aux_dict.items():
        if key.endswith("_month"):
            feats[f"aux_{key}"] = feats["month"].map(mapping).fillna(0)
        elif key.endswith("_dow"):
            feats[f"aux_{key}"] = feats["dayofweek"].map(mapping).fillna(0)

    # Prophet components
    feats = feats.merge(prophet_df, on="Date", how="left")
    for col in feats.columns:
        if col != "Date" and feats[col].isna().any():
            feats[col] = feats[col].fillna(0)

    return feats.drop(columns=["Date"])


def fit_lgbm(X, y_resid):
    """Fit LightGBM with high regularization (conservative role in v4)."""
    rank = np.arange(len(X), dtype=float)
    sw = 1.0 + 2.0 * (rank / max(1.0, rank.max())) ** 1.5

    m = LGBMRegressor(
        n_estimators=800, learning_rate=0.03, num_leaves=20,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=30, reg_alpha=1.0, reg_lambda=10.0,
        objective="mae", random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    m.fit(X, y_resid, sample_weight=sw)
    return m


print("Features & LightGBM ready")


# ══════════════════════════════════════════════════════════════════════
# V4 CORE: MULTI-MODEL PIPELINE WITH AGGRESSIVE HORIZON DECAY
# ══════════════════════════════════════════════════════════════════════

def run_pipeline(
    train_df, target_col, fc_dates, holidays,
    lgb_train_start=2016,
):
    """
    Full pipeline returning individual model predictions.
    Fits each Prophet config only ONCE, caches model + predictions.
    """
    origin = train_df["Date"].max()
    n = len(fc_dates)

    # ── 1. Seasonal Naive (full history) ──
    p364 = seasonal_naive_364(train_df.set_index("Date")[target_col], fc_dates)
    pwin = seasonal_window_median(train_df, target_col, fc_dates, window=7)
    p_naive = 0.5 * p364 + 0.5 * pwin

    # ── 2. Theta (full history) ──
    p_theta = theta_forecast(train_df.set_index("Date")[target_col], n)

    # ── 3. Prophet ensemble (fit each config once, cache) ──
    prophet_models = []  # (model, cap, floor)
    prophet_fc_preds = []
    prophet_fc_comps = []

    for i, cfg in enumerate(PROPHET_CONFIGS):
        print(f"    Prophet config {i+1}/{len(PROPHET_CONFIGS)}: "
              f"cps={cfg['cps']}, {cfg['smode']}, fo={cfg['yearly_fo']}")
        m, cap, fl = fit_prophet_config(
            train_df, target_col, holidays,
            cfg["cps"], cfg["smode"], cfg["yearly_fo"])
        prophet_models.append((m, cap, fl))

        pr_fc = prophet_predict(m, fc_dates, cap, fl)
        prophet_fc_preds.append(pr_fc["prophet_yhat"].values)
        prophet_fc_comps.append(pr_fc)

    # Median of Prophet ensemble (more robust than mean)
    p_prophet = np.median(np.array(prophet_fc_preds), axis=0)

    # ── 4. LightGBM residual (recent history, uses first Prophet config) ──
    m0, cap0, fl0 = prophet_models[0]
    pr_train_comp = prophet_predict(m0, train_df["Date"].tolist(), cap0, fl0)
    pr_fc_comp = prophet_fc_comps[0]

    lgb_train = train_df[train_df["Date"].dt.year >= lgb_train_start].copy()
    # Get Prophet predictions for LGB training period
    pr_lgb = prophet_predict(m0, lgb_train["Date"].tolist(), cap0, fl0)

    resid = lgb_train[target_col].values - pr_lgb["prophet_yhat"].values
    X_train = build_features(lgb_train["Date"].tolist(),
                             lgb_train[["Date", target_col]],
                             target_col, origin, pr_lgb, aux)
    X_fc = build_features(fc_dates, lgb_train[["Date", target_col]],
                          target_col, origin, pr_fc_comp, aux)

    lgb_model = fit_lgbm(X_train, resid)
    resid_pred = lgb_model.predict(X_fc)
    p_hybrid = np.clip(p_prophet + resid_pred, 0.0, None)

    return {
        "naive": p_naive,
        "theta": p_theta,
        "prophet": p_prophet,
        "hybrid": p_hybrid,
        "prophet_preds": prophet_fc_preds,
    }


print("Pipeline ready")


# ══════════════════════════════════════════════════════════════════════
# BLENDING WITH AGGRESSIVE HORIZON DECAY
# ══════════════════════════════════════════════════════════════════════

def blend_v4(preds, n, variant="balanced"):
    """
    V4 blending strategy:
    - Days 1-60: hybrid gets meaningful weight (tree models still useful)
    - Days 60-180: hybrid weight decays rapidly
    - Days 180+: almost pure Prophet+Naive+Theta (tree models unreliable)

    This is the KEY difference from v3 which gave hybrid 55% uniformly.
    Research: "shift trust toward LightGBM for first 30-60 days,
    smoothly transition to 100% Prophet reliance for days 300-548"
    """
    p_naive = preds["naive"]
    p_theta = preds["theta"]
    p_prophet = preds["prophet"]
    p_hybrid = preds["hybrid"]

    t = np.arange(n, dtype=float)

    # Hybrid weight: starts at w_h_max, decays exponentially
    if variant == "aggressive":
        # More hybrid trust
        w_h = 0.35 * np.exp(-t / 120)  # half-life ~83 days
        w_naive = 0.20 * np.ones(n)
        w_theta = 0.10 * np.ones(n)
    elif variant == "conservative":
        # Less hybrid, more stable models
        w_h = 0.20 * np.exp(-t / 60)   # half-life ~42 days
        w_naive = 0.25 * np.ones(n)
        w_theta = 0.15 * np.ones(n)
    elif variant == "pure_prophet":
        # Zero hybrid, pure statistical
        w_h = np.zeros(n)
        w_naive = 0.25 * np.ones(n)
        w_theta = 0.15 * np.ones(n)
    else:  # balanced
        w_h = 0.30 * np.exp(-t / 90)   # half-life ~62 days
        w_naive = 0.22 * np.ones(n)
        w_theta = 0.12 * np.ones(n)

    # Prophet gets the remainder
    w_prophet = 1.0 - w_h - w_naive - w_theta
    w_prophet = np.clip(w_prophet, 0.1, None)

    # Renormalize
    total = w_h + w_naive + w_theta + w_prophet
    w_h /= total
    w_naive /= total
    w_theta /= total
    w_prophet /= total

    pred = w_h * p_hybrid + w_naive * p_naive + w_theta * p_theta + w_prophet * p_prophet
    return np.clip(pred, 0.0, None)


def apply_tet_calibration(pred, fc_dates, tet_multipliers, base_median):
    """
    Apply empirical Tet multipliers to adjust predictions around Tet.
    This corrects for systematic under/over-prediction during Tet.
    """
    pred = pred.copy()
    for tet in TET_DATES:
        for delta, mult in tet_multipliers.items():
            d = tet + pd.Timedelta(days=delta)
            if d in fc_dates:
                idx = fc_dates.index(d)
                # Blend: 70% model prediction, 30% empirical calibration
                empirical = base_median * mult
                pred[idx] = 0.7 * pred[idx] + 0.3 * empirical
    return pred


print("V4 blending ready (4 variants: conservative, balanced, aggressive, pure_prophet)")


# ══════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════

FOLD_YEARS = [2020, 2021, 2022]
VARIANTS = ["conservative", "balanced", "aggressive", "pure_prophet"]

print("\n" + "=" * 70)
print("V4 CROSS-VALIDATION")
print("=" * 70)

cv_results = {}

for target_col in ["Revenue", "COGS"]:
    print(f"\n── {target_col} ──")
    tet_mults = tet_mult_rev if target_col == "Revenue" else tet_mult_cogs

    variant_maes = {v: [] for v in VARIANTS}
    fold_weights = []

    for year in FOLD_YEARS:
        print(f"  Fold {year}:")
        train = sales[sales["Date"].dt.year < year].copy()
        valid = sales[sales["Date"].dt.year == year].copy()

        if len(train) < 365 or len(valid) < 30:
            continue

        val_dates = valid["Date"].tolist()
        y_val = valid[target_col].values.astype(float)
        holidays = build_holidays_df(valid["Date"].max())

        t0 = time.time()
        preds = run_pipeline(train, target_col, val_dates, holidays)
        elapsed = time.time() - t0

        base_med = float(train[target_col].median())

        for variant in VARIANTS:
            p = blend_v4(preds, len(val_dates), variant)
            p = apply_tet_calibration(p, val_dates, tet_mults, base_med)
            mae = mean_absolute_error(y_val, p)
            variant_maes[variant].append(mae)

        # Also test optimized static weights for reference
        best_static_mae = 1e18
        for wn in np.arange(0.0, 0.5, 0.05):
            for wt in np.arange(0.0, 0.3, 0.05):
                for wp in np.arange(0.0, 0.8, 0.05):
                    wh = 1.0 - wn - wt - wp
                    if wh < 0:
                        continue
                    p = wn * preds["naive"] + wt * preds["theta"] + wp * preds["prophet"] + wh * preds["hybrid"]
                    mae = mean_absolute_error(y_val, np.clip(p, 0, None))
                    if mae < best_static_mae:
                        best_static_mae = mae
                        best_static_w = (wn, wt, wp, wh)

        print(f"    Time: {elapsed:.0f}s")
        for v in VARIANTS:
            print(f"    {v}: MAE={variant_maes[v][-1]:,.0f}")
        print(f"    optimal_static: MAE={best_static_mae:,.0f} w={best_static_w}")

    # Weighted average (recent folds matter more)
    for v in VARIANTS:
        maes = variant_maes[v]
        if maes:
            w = np.array([1.0 + i * 0.5 for i in range(len(maes))])
            w /= w.sum()
            cv_results[f"{target_col}_{v}"] = float(np.dot(maes, w))

    print(f"\n  {target_col} CV Summary:")
    for v in VARIANTS:
        key = f"{target_col}_{v}"
        if key in cv_results:
            print(f"    {v}: {cv_results[key]:,.0f}")

print("\n" + "=" * 70)
print("CV COMPLETE")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# FINAL FORECAST GENERATION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING FINAL FORECASTS")
print("=" * 70)

# Find best variant per target
best_variant = {}
for target_col in ["Revenue", "COGS"]:
    best_v = None
    best_mae = 1e18
    for v in VARIANTS:
        key = f"{target_col}_{v}"
        if key in cv_results and cv_results[key] < best_mae:
            best_mae = cv_results[key]
            best_v = v
    best_variant[target_col] = best_v
    print(f"{target_col}: best variant = {best_v} (CV MAE = {best_mae:,.0f})")

# Generate final predictions
holidays_final = build_holidays_df(pd.Timestamp(max(forecast_dates)))

submissions = {}
for variant in VARIANTS:
    print(f"\n── {variant} ──")
    sub = sub_tpl[["Date"]].copy()

    for target_col in ["Revenue", "COGS"]:
        print(f"  {target_col}...", end=" ", flush=True)
        t0 = time.time()

        preds = run_pipeline(sales, target_col, forecast_dates, holidays_final)
        p = blend_v4(preds, N_FC, variant)

        # Apply Tet calibration
        tet_mults = tet_mult_rev if target_col == "Revenue" else tet_mult_cogs
        base_med = float(sales[target_col].median())
        p = apply_tet_calibration(p, forecast_dates, tet_mults, base_med)

        sub[target_col] = p
        print(f"done ({time.time()-t0:.0f}s)")

    submissions[variant] = sub

# ── Validation & Output ──
print("\n" + "=" * 70)
print("VALIDATION & OUTPUT")
print("=" * 70)

output_files = {}
for variant in VARIANTS:
    sub = submissions[variant]
    fname = f"submission_v4_{variant}.csv"
    fpath = os.path.join(OUT_DIR, fname)

    ok = True
    if len(sub) != N_FC:
        print(f"  ❌ {variant}: {len(sub)} rows (expected {N_FC})")
        ok = False
    if sub[["Revenue", "COGS"]].isna().any().any():
        print(f"  ❌ {variant}: NaN values")
        ok = False
    if (sub[["Revenue", "COGS"]] < 0).any().any():
        print(f"  ❌ {variant}: negative values")
        ok = False

    if ok:
        is_best = ""
        for tc in ["Revenue", "COGS"]:
            if best_variant[tc] == variant:
                is_best += f" ★BEST-{tc}"
        print(f"  ✅ {variant}: {N_FC} rows{is_best}")

    for col in ["Revenue", "COGS"]:
        vals = sub[col]
        print(f"     {col}: mean={vals.mean():,.0f} med={vals.median():,.0f} "
              f"min={vals.min():,.0f} max={vals.max():,.0f}")

    out = sub.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(fpath, index=False)
    output_files[variant] = fpath

# ── Diagnostics ──
diag = {
    "version": "v4_topkill",
    "timestamp": pd.Timestamp.now().isoformat(),
    "cv_results": cv_results,
    "best_variant": best_variant,
    "prophet_configs": PROPHET_CONFIGS,
    "fold_years": FOLD_YEARS,
}
diag_path = os.path.join(OUT_DIR, "v4_topkill_diagnostics.json")
with open(diag_path, "w") as f:
    json.dump(diag, f, indent=2, default=str)

print(f"\nDiagnostics: {diag_path}")
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
for tc in ["Revenue", "COGS"]:
    bv = best_variant[tc]
    print(f"  {tc}: {bv} (CV MAE = {cv_results[f'{tc}_{bv}']:,.0f})")
print(f"\nOutput files:")
for v, fp in output_files.items():
    print(f"  {v}: {fp}")
print(f"\n🏁 Upload submission_v4_{best_variant['Revenue']}.csv first.")
print("   Key change: hybrid weight decays exponentially -> pure Prophet at horizon 180+")
