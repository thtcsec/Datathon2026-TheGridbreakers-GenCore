"""
DATATHON 2026 - GenCore v5 Naive-First
========================================
CRITICAL INSIGHT from 548-day holdout analysis:
  - Prophet ALONE = 2.3M MAE (catastrophic)
  - Naive364 ALONE = 830k MAE (best single model)
  - Best 2-blend = 100% Naive, 0% Prophet
  - v3/v4 gave hybrid (Prophet-based) 55% weight -> WRONG

V5 Strategy:
  1. Naive364 as PRIMARY backbone (not Prophet)
  2. LightGBM corrects Naive residuals (not Prophet residuals)
  3. Prophet used ONLY as minor feature, not as base prediction
  4. Multiple Naive variants blended (364, window, trend-adjusted)
  5. Validate on TRUE 548-day holdout (not per-year)
  6. Tet-specific correction from empirical data
"""

import os, glob, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)

KAGGLE = os.path.exists("/kaggle/input")
if KAGGLE:
    matches = glob.glob("/kaggle/input/**/sales.csv", recursive=True)
    DATA_DIR = os.path.dirname(matches[0]) if matches else "/kaggle/input"
    OUT_DIR = "/kaggle/working"
else:
    DATA_DIR = "data/raw"
    for c in ["data/raw", "../data/raw"]:
        if os.path.isfile(os.path.join(c, "sales.csv")): DATA_DIR = c; break
    OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from prophet import Prophet; HAS_PROPHET = True
except ImportError:
    try: os.system("pip install -q prophet"); from prophet import Prophet; HAS_PROPHET = True
    except: HAS_PROPHET = False

try:
    from lightgbm import LGBMRegressor
except ImportError:
    os.system("pip install -q lightgbm"); from lightgbm import LGBMRegressor

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

print(f"ENV: {'Kaggle' if KAGGLE else 'Local'} | DATA: {DATA_DIR}")

# ── Load data ──
sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sales["Date"] = pd.to_datetime(sales["Date"]); sub_tpl["Date"] = pd.to_datetime(sub_tpl["Date"])
sales = sales.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
sub_tpl = sub_tpl.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
forecast_dates = sub_tpl["Date"].tolist()
N_FC = len(forecast_dates)
print(f"Train: {sales.Date.min().date()}->{sales.Date.max().date()} ({len(sales)} rows)")
print(f"Forecast: {N_FC} days")

# ── Calendar ──
TET = pd.to_datetime(["2012-01-23","2013-02-10","2014-01-31","2015-02-19","2016-02-08",
    "2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12","2022-02-01",
    "2023-01-22","2024-02-10"])
MEGA = [(1,1),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12)]

def d2next(dates, events, default=365):
    ev = np.sort(np.array(events, dtype="datetime64[ns]"))
    d = dates.to_numpy().astype("datetime64[ns]")
    out = np.full(len(d), default, dtype=int)
    idx = np.searchsorted(ev, d, side="left")
    for i in range(len(d)):
        if idx[i] < len(ev): out[i] = int((ev[idx[i]] - d[i]) / np.timedelta64(1, "D"))
    return out

def d2last(dates, events, default=365):
    ev = np.sort(np.array(events, dtype="datetime64[ns]"))
    d = dates.to_numpy().astype("datetime64[ns]")
    out = np.full(len(d), default, dtype=int)
    idx = np.searchsorted(ev, d, side="right") - 1
    for i in range(len(d)):
        if idx[i] >= 0: out[i] = int((d[i] - ev[idx[i]]) / np.timedelta64(1, "D"))
    return out

# ── Aux ──
aux = {}
for fn, dc, cols in [("web_traffic.csv","date",["sessions","unique_visitors","page_views"]),
                      ("orders.csv","order_date",[])]:
    fp = os.path.join(DATA_DIR, fn)
    if not os.path.isfile(fp): continue
    df = pd.read_csv(fp); df[dc] = pd.to_datetime(df[dc], errors="coerce"); df = df.dropna(subset=[dc])
    df["month"] = df[dc].dt.month; df["dow"] = df[dc].dt.dayofweek
    if cols:
        for c in cols:
            if c in df.columns:
                aux[f"{c}_month"] = df.groupby("month")[c].median().to_dict()
                aux[f"{c}_dow"] = df.groupby("dow")[c].median().to_dict()
    else:
        daily = df.groupby(dc).size().reset_index(name="n")
        daily["month"] = daily[dc].dt.month; daily["dow"] = daily[dc].dt.dayofweek
        aux["orders_month"] = daily.groupby("month")["n"].median().to_dict()
        aux["orders_dow"] = daily.groupby("dow")["n"].median().to_dict()


# ══════════════════════════════════════════════════════════════════════
# NAIVE BASELINES (THE BACKBONE)
# ══════════════════════════════════════════════════════════════════════

def naive364(train_s, fc_dates):
    """Seasonal Naive 364 (DOW-preserving)."""
    s = train_s.sort_index().astype(float)
    h = {pd.Timestamp(i): float(v) for i, v in s.items()}
    fb = float(s.tail(28).median())
    preds = []
    for dt in pd.to_datetime(fc_dates):
        dt = pd.Timestamp(dt); val = None
        for off in [364, 371, 357, 728]:
            c = dt - pd.Timedelta(days=off)
            if c in h: val = h[c]; break
        if val is None: val = fb
        h[dt] = float(val); preds.append(float(val))
    return np.array(preds)


def naive_multi_year_median(train_df, col, fc_dates):
    """
    For each forecast date, take MEDIAN of same-DOY values from
    multiple historical years (weighted toward recent).
    More robust than single-year naive.
    """
    work = train_df[["Date", col]].copy()
    work["doy"] = work.Date.dt.dayofyear
    work["dow"] = work.Date.dt.dayofweek
    work["year"] = work.Date.dt.year

    # Weight recent years more
    max_year = work["year"].max()
    work["weight"] = 1.0 + (work["year"] - work["year"].min()) / max(1, max_year - work["year"].min())

    vals = work[col].values.astype(float)
    doys = work["doy"].values
    dows = work["dow"].values
    weights = work["weight"].values
    fb = float(np.average(vals, weights=weights))

    preds = []
    for dt in pd.to_datetime(fc_dates):
        doy, dow = int(dt.dayofyear), int(dt.dayofweek)
        diff = np.abs(doys - doy)
        dist = np.minimum(diff, 366 - diff)

        # Same DOW, within 3 days of DOY
        mask = (dows == dow) & (dist <= 3)
        if mask.sum() < 3:
            mask = (dows == dow) & (dist <= 7)
        if mask.sum() < 3:
            mask = dist <= 7

        if mask.sum() > 0:
            w = weights[mask]
            v = vals[mask]
            # Weighted median approximation: sort by value, find weighted midpoint
            idx = np.argsort(v)
            v_sorted = v[idx]
            w_sorted = w[idx]
            cum_w = np.cumsum(w_sorted)
            mid = cum_w[-1] / 2.0
            median_idx = np.searchsorted(cum_w, mid)
            preds.append(float(v_sorted[min(median_idx, len(v_sorted)-1)]))
        else:
            preds.append(fb)

    return np.array(preds)


def naive_blend(train_df, col, fc_dates):
    """Blend of naive variants."""
    p364 = naive364(train_df.set_index("Date")[col], fc_dates)
    p_multi = naive_multi_year_median(train_df, col, fc_dates)

    # YoY growth adjustment
    yearly = train_df.groupby(train_df.Date.dt.year)[col].sum()
    if len(yearly) >= 2:
        g = yearly.iloc[-1] / max(yearly.iloc[-2], 1.0)
        g = np.clip(g, 0.90, 1.15)
    else:
        g = 1.0

    p364_adj = p364 * g
    # Blend: 60% naive364 (adjusted), 40% multi-year median
    return 0.6 * p364_adj + 0.4 * p_multi


print("Naive baselines ready")


# ══════════════════════════════════════════════════════════════════════
# LIGHTGBM CORRECTS NAIVE RESIDUALS (NOT PROPHET)
# ══════════════════════════════════════════════════════════════════════

def build_features_v5(dates, train_df, col, origin, aux_d, prophet_yhat=None):
    """Features for correcting naive residuals. Prophet only as minor feature."""
    f = pd.DataFrame({"Date": pd.to_datetime(list(dates))})
    d = f["Date"]

    f["month"] = d.dt.month; f["day"] = d.dt.day
    f["dayofweek"] = d.dt.dayofweek; f["dayofyear"] = d.dt.dayofyear
    f["weekofyear"] = d.dt.isocalendar().week.astype(int).values
    f["quarter"] = d.dt.quarter
    f["is_weekend"] = (f["dayofweek"] >= 5).astype(int)
    f["is_month_start"] = d.dt.is_month_start.astype(int)
    f["is_month_end"] = d.dt.is_month_end.astype(int)
    f["is_payday"] = f["day"].isin([1, 15, 25]).astype(int)

    f["forecast_horizon"] = np.clip((d - pd.Timestamp(origin)).dt.days, 0, 600)

    # Fourier
    doy = f["dayofyear"].values.astype(float)
    dow = f["dayofweek"].values.astype(float)
    for k in range(1, 6):
        f[f"sin_y{k}"] = np.sin(2*np.pi*k*doy/365.25)
        f[f"cos_y{k}"] = np.cos(2*np.pi*k*doy/365.25)
    for k in range(1, 4):
        f[f"sin_w{k}"] = np.sin(2*np.pi*k*dow/7.0)
        f[f"cos_w{k}"] = np.cos(2*np.pi*k*dow/7.0)

    # Tet
    ta = TET.to_numpy()
    f["days_to_tet"] = d2next(d, ta); f["days_since_tet"] = d2last(d, ta)
    f["is_pre_tet_21"] = f["days_to_tet"].between(1, 21).astype(int)
    f["is_pre_tet_7"] = f["days_to_tet"].between(1, 7).astype(int)
    f["is_tet_week"] = f["days_to_tet"].between(-7, 0).astype(int)
    f["is_post_tet_14"] = f["days_since_tet"].between(1, 14).astype(int)
    f["tet_proximity"] = np.exp(-0.1 * np.minimum(
        f["days_to_tet"].abs(), f["days_since_tet"].abs()))

    # Mega-sale
    sl = []
    for y in range(2012, 2025):
        for m, dd in MEGA: sl.append(pd.Timestamp(year=y, month=m, day=dd))
    sa = np.array(sorted(sl), dtype="datetime64[ns]")
    f["days_to_sale"] = d2next(d, sa)
    f["is_sale_window"] = (f["days_to_sale"].abs() <= 3).astype(int)
    f["is_11_11"] = ((f["month"]==11) & (f["day"]==11)).astype(int)
    f["is_12_12"] = ((f["month"]==12) & (f["day"]==12)).astype(int)

    # Historical profiles
    hi = train_df[["Date", col]].copy()
    hi["doy"] = hi.Date.dt.dayofyear; hi["dow"] = hi.Date.dt.dayofweek; hi["month"] = hi.Date.dt.month
    gm = float(hi[col].median())
    f["hist_doy"] = f["dayofyear"].map(hi.groupby("doy")[col].median()).fillna(gm)
    f["hist_dow"] = f["dayofweek"].map(hi.groupby("dow")[col].median()).fillna(gm)
    f["hist_month"] = f["month"].map(hi.groupby("month")[col].median()).fillna(gm)

    # Aux
    for key, mapping in aux_d.items():
        if key.endswith("_month"): f[f"aux_{key}"] = f["month"].map(mapping).fillna(0)
        elif key.endswith("_dow"): f[f"aux_{key}"] = f["dayofweek"].map(mapping).fillna(0)

    # Prophet as MINOR feature (not base prediction)
    if prophet_yhat is not None:
        f["prophet_hint"] = prophet_yhat

    for c in f.columns:
        if c != "Date" and f[c].isna().any(): f[c] = f[c].fillna(0)

    return f.drop(columns=["Date"])


def get_prophet_hint(train_df, col, dates):
    """Get Prophet prediction as a feature (not as base model)."""
    if not HAS_PROPHET:
        return np.zeros(len(dates))

    rows = []
    for td in TET:
        if td <= pd.Timestamp(max(dates)) + pd.Timedelta(days=60):
            rows.append({"holiday": "tet", "ds": td, "lower_window": -21, "upper_window": 7})
    holidays = pd.DataFrame(rows) if rows else None

    df = train_df[["Date", col]].rename(columns={"Date": "ds", col: "y"})
    cap = float(df.y.quantile(0.995) * 1.25)
    fl = max(0, float(df.y.quantile(0.005) * 0.75))
    df["cap"] = cap; df["floor"] = fl

    m = Prophet(growth="logistic", holidays=holidays, yearly_seasonality=15,
                weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="multiplicative", changepoint_prior_scale=0.03,
                changepoint_range=0.9)
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m.fit(df)

    fut = pd.DataFrame({"ds": pd.to_datetime(list(dates))})
    fut["cap"] = cap; fut["floor"] = fl
    return m.predict(fut)["yhat"].values


def fit_lgbm_v5(X, y_resid, n_est=600, lr=0.04, nl=15, ra=2.0, rl=15.0):
    """Ultra-regularized LightGBM for naive residual correction."""
    rank = np.arange(len(X), dtype=float)
    sw = 1.0 + 2.0 * (rank / max(1.0, rank.max())) ** 1.5
    m = LGBMRegressor(n_estimators=n_est, learning_rate=lr, num_leaves=nl,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        reg_alpha=ra, reg_lambda=rl, objective="mae", random_state=SEED,
        n_jobs=-1, verbosity=-1)
    m.fit(X, y_resid, sample_weight=sw)
    return m


print("LightGBM residual corrector ready")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE & VALIDATION
# ══════════════════════════════════════════════════════════════════════

def run_v5_pipeline(train_df, col, fc_dates, use_prophet_hint=True):
    """V5 pipeline: Naive backbone + LightGBM correction."""
    origin = train_df.Date.max()

    # 1. Naive backbone
    p_naive = naive_blend(train_df, col, fc_dates)

    # 2. Prophet hint (minor feature)
    if use_prophet_hint and HAS_PROPHET:
        prophet_train = get_prophet_hint(train_df, col, train_df.Date.tolist())
        prophet_fc = get_prophet_hint(train_df, col, fc_dates)
    else:
        prophet_train = None
        prophet_fc = None

    # 3. Compute naive predictions for training period (for residual learning)
    train_naive = naive_blend(train_df, col, train_df.Date.tolist())
    resid = train_df[col].values - train_naive

    # 4. Build features
    X_train = build_features_v5(train_df.Date.tolist(), train_df, col, origin, aux, prophet_train)
    X_fc = build_features_v5(fc_dates, train_df, col, origin, aux, prophet_fc)

    # 5. LightGBM corrects naive residuals
    lgb = fit_lgbm_v5(X_train, resid)
    correction = lgb.predict(X_fc)

    # 6. Final = naive + damped correction
    # Damp correction at far horizons (tree models less reliable)
    n = len(fc_dates)
    t = np.arange(n, dtype=float)
    damp = np.exp(-t / 300)  # gentle decay, half-life ~208 days
    p_corrected = p_naive + correction * damp

    return {
        "naive": p_naive,
        "corrected": np.clip(p_corrected, 0, None),
        "correction": correction,
    }


# ── 548-day holdout validation ──
print("\n" + "=" * 70)
print("548-DAY HOLDOUT VALIDATION")
print("=" * 70)

# Multiple holdout origins to get robust estimate
HOLDOUT_ORIGINS = [
    ("2021-06-30", 548),  # val: 2021-07-01 to 2022-12-30
    ("2021-01-01", 548),  # val: 2021-01-02 to 2022-07-03
    ("2020-06-30", 548),  # val: 2020-07-01 to 2021-12-30
]

cv_results = {}

for col in ["Revenue", "COGS"]:
    print(f"\n── {col} ──")
    fold_maes = {"naive": [], "corrected": [], "pure_naive364": []}

    for origin_str, n_days in HOLDOUT_ORIGINS:
        origin = pd.Timestamp(origin_str)
        train = sales[sales.Date <= origin].copy()
        val = sales[sales.Date > origin].head(n_days).copy()

        if len(val) < 100:
            continue

        val_dates = val.Date.tolist()
        y = val[col].values

        # Pure naive364 baseline
        p364 = naive364(train.set_index("Date")[col], val_dates)
        fold_maes["pure_naive364"].append(mean_absolute_error(y, p364))

        # V5 pipeline
        preds = run_v5_pipeline(train, col, val_dates)
        fold_maes["naive"].append(mean_absolute_error(y, preds["naive"]))
        fold_maes["corrected"].append(mean_absolute_error(y, preds["corrected"]))

        print(f"  Origin {origin.date()}: naive364={fold_maes['pure_naive364'][-1]:,.0f}, "
              f"blend={fold_maes['naive'][-1]:,.0f}, corrected={fold_maes['corrected'][-1]:,.0f}")

    # Weighted average (recent origins matter more)
    for key in fold_maes:
        maes = fold_maes[key]
        if maes:
            w = np.array([1.0 + i * 0.5 for i in range(len(maes))])
            w /= w.sum()
            cv_results[f"{col}_{key}"] = float(np.dot(maes, w))

    print(f"\n  {col} 548-day CV:")
    for key in ["pure_naive364", "naive", "corrected"]:
        k = f"{col}_{key}"
        if k in cv_results:
            print(f"    {key}: {cv_results[k]:,.0f}")

# Also do per-year CV for comparison
print("\n── Per-Year CV (for comparison with v3/v4) ──")
for col in ["Revenue", "COGS"]:
    fold_maes = []
    for year in [2020, 2021, 2022]:
        train = sales[sales.Date.dt.year < year].copy()
        val = sales[sales.Date.dt.year == year].copy()
        if len(train) < 365: continue
        preds = run_v5_pipeline(train, col, val.Date.tolist())
        mae = mean_absolute_error(val[col].values, preds["corrected"])
        fold_maes.append(mae)
    w = np.array([1.0 + i * 0.5 for i in range(len(fold_maes))]); w /= w.sum()
    avg = float(np.dot(fold_maes, w))
    cv_results[f"{col}_peryear"] = avg
    print(f"  {col} per-year CV: {avg:,.0f}")


# ══════════════════════════════════════════════════════════════════════
# FINAL FORECAST & OUTPUT
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING FINAL FORECASTS")
print("=" * 70)

# Generate with different correction strengths
VARIANTS = {
    "corrected": 1.0,       # full LightGBM correction
    "mild_correct": 0.5,    # 50% correction
    "pure_naive": 0.0,      # zero correction (pure naive blend)
}

submissions = {}
for vname, corr_strength in VARIANTS.items():
    sub = sub_tpl[["Date"]].copy()
    for col in ["Revenue", "COGS"]:
        print(f"  {vname}/{col}...", end=" ", flush=True)
        t0 = time.time()
        preds = run_v5_pipeline(sales, col, forecast_dates)

        if corr_strength > 0:
            n = len(forecast_dates)
            t = np.arange(n, dtype=float)
            damp = np.exp(-t / 300) * corr_strength
            p = preds["naive"] + preds["correction"] * damp
        else:
            p = preds["naive"]

        sub[col] = np.clip(p, 0, None)
        print(f"done ({time.time()-t0:.0f}s)")
    submissions[vname] = sub

# Validate & save
print("\n" + "=" * 70)
print("VALIDATION & OUTPUT")
print("=" * 70)

output_files = {}
for vname, sub in submissions.items():
    ok = len(sub)==N_FC and not sub[["Revenue","COGS"]].isna().any().any() and not (sub[["Revenue","COGS"]]<0).any().any()
    print(f"  {'✅' if ok else '❌'} {vname}: {len(sub)} rows")
    for c in ["Revenue","COGS"]:
        v = sub[c]
        print(f"     {c}: mean={v.mean():,.0f} med={v.median():,.0f} min={v.min():,.0f} max={v.max():,.0f}")

    fname = f"submission_v5_{vname}.csv"
    fpath = os.path.join(OUT_DIR, fname)
    out = sub.copy(); out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(fpath, index=False)
    output_files[vname] = fpath

# Diagnostics
diag = {
    "version": "v5_naive_first",
    "timestamp": pd.Timestamp.now().isoformat(),
    "cv_results": cv_results,
    "key_insight": "Prophet alone=2.3M MAE on 548-day holdout. Naive364=830k. V5 uses Naive as backbone.",
}
dp = os.path.join(OUT_DIR, "v5_diagnostics.json")
with open(dp, "w") as f: json.dump(diag, f, indent=2, default=str)

print(f"\nDiagnostics: {dp}")
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for col in ["Revenue", "COGS"]:
    for key in ["pure_naive364", "naive", "corrected", "peryear"]:
        k = f"{col}_{key}"
        if k in cv_results:
            print(f"  {col} {key}: {cv_results[k]:,.0f}")
print(f"\nFiles: {list(output_files.keys())}")
print("\n🏁 Try submission_v5_pure_naive.csv first (baseline).")
print("   Then submission_v5_corrected.csv to see if LightGBM helps on LB.")
print("   If pure_naive beats corrected on LB -> tree model is hurting, not helping.")
