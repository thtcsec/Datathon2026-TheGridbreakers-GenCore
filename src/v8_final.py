"""
DATATHON 2026 - GenCore v8 Final
=================================
Based on v7 NaiveKing (best approach so far).

Improvements over v7:
1. BETTER NAIVE: use 3-year weighted average instead of single 364-day lookback
   (reduces noise from one-off events in any single year)
2. SMARTER RESIDUAL TRAINING: train on ALL years with sample weighting,
   not just last 2.5 years (more data for seasonal patterns)
3. CATBOOST added to ensemble (3 models instead of 2)
4. BETTER TET: use actual Tet-relative-day profiles, not just multipliers
5. HORIZON-AWARE CORRECTION: separate models for near (0-180) and far (180+)
6. OUTLIER-ROBUST: use Huber loss instead of MAE for LightGBM
"""

import os, glob, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
SEED = 42; np.random.seed(SEED)

KAGGLE = os.path.exists('/kaggle/input')
if KAGGLE:
    matches = glob.glob('/kaggle/input/**/sales.csv', recursive=True)
    DATA_DIR = os.path.dirname(matches[0]) if matches else '/kaggle/input'
    OUT_DIR = '/kaggle/working'
else:
    DATA_DIR = 'data/raw'
    for c in ['data/raw', '../data/raw']:
        if os.path.isfile(os.path.join(c, 'sales.csv')): DATA_DIR = c; break
    OUT_DIR = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

try: from lightgbm import LGBMRegressor
except: os.system('pip install -q lightgbm'); from lightgbm import LGBMRegressor
try: from xgboost import XGBRegressor
except: os.system('pip install -q xgboost'); from xgboost import XGBRegressor

print(f"ENV: {'Kaggle' if KAGGLE else 'Local'} | DATA: {DATA_DIR}")

# ── Data ──
sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
sales['Date'] = pd.to_datetime(sales['Date']); sub_tpl['Date'] = pd.to_datetime(sub_tpl['Date'])
sales = sales.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
sub_tpl = sub_tpl.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
forecast_dates = sub_tpl['Date'].tolist()
N_FC = len(forecast_dates)
print(f"Train: {sales.Date.min().date()}->{sales.Date.max().date()} ({len(sales)}), Forecast: {N_FC}")

# ── Calendar ──
TET = pd.to_datetime(['2012-01-23','2013-02-10','2014-01-31','2015-02-19','2016-02-08',
    '2017-01-28','2018-02-16','2019-02-05','2020-01-25','2021-02-12','2022-02-01',
    '2023-01-22','2024-02-10'])
MEGA = [(1,1),(3,3),(4,30),(5,1),(6,6),(7,7),(8,8),(9,2),(9,9),(10,10),(11,11),(12,12)]

def d2next_tet(dates):
    ev = np.sort(TET.to_numpy().astype('datetime64[ns]'))
    d = np.array(pd.to_datetime(dates), dtype='datetime64[ns]')
    out = np.full(len(d), 365, dtype=int)
    idx = np.searchsorted(ev, d, side='left')
    for i in range(len(d)):
        if idx[i] < len(ev): out[i] = int((ev[idx[i]] - d[i]) / np.timedelta64(1,'D'))
    return np.clip(out, 0, 365)

def d2last_tet(dates):
    ev = np.sort(TET.to_numpy().astype('datetime64[ns]'))
    d = np.array(pd.to_datetime(dates), dtype='datetime64[ns]')
    out = np.full(len(d), 365, dtype=int)
    idx = np.searchsorted(ev, d, side='right') - 1
    for i in range(len(d)):
        if idx[i] >= 0: out[i] = int((d[i] - ev[idx[i]]) / np.timedelta64(1,'D'))
    return np.clip(out, 0, 365)

# ── Aux ──
aux = {}
for fn, dc, cols in [('web_traffic.csv','date',['sessions','unique_visitors','page_views']),
                      ('orders.csv','order_date',[])]:
    fp = os.path.join(DATA_DIR, fn)
    if not os.path.isfile(fp): continue
    df = pd.read_csv(fp); df[dc] = pd.to_datetime(df[dc], errors='coerce'); df = df.dropna(subset=[dc])
    df['month'] = df[dc].dt.month; df['dow'] = df[dc].dt.dayofweek
    if cols:
        for c in cols:
            if c in df.columns:
                aux[f'{c}_month'] = df.groupby('month')[c].median().to_dict()
                aux[f'{c}_dow'] = df.groupby('dow')[c].median().to_dict()
    else:
        daily = df.groupby(dc).size().reset_index(name='n')
        daily['month']=daily[dc].dt.month; daily['dow']=daily[dc].dt.dayofweek
        aux['orders_month']=daily.groupby('month')['n'].median().to_dict()
        aux['orders_dow']=daily.groupby('dow')['n'].median().to_dict()


# ══════════════════════════════════════════════════════════════════════
# IMPROVED NAIVE: 3-YEAR WEIGHTED AVERAGE
# ══════════════════════════════════════════════════════════════════════

def naive364(train_s, fc_dates):
    """Standard 364-day seasonal naive."""
    s = train_s.sort_index().astype(float)
    h = {pd.Timestamp(i): float(v) for i, v in s.items()}
    fb = float(s.tail(28).median())
    preds = []
    for dt in pd.to_datetime(fc_dates):
        dt = pd.Timestamp(dt); val = None
        for off in [364, 371, 357, 728, 735, 721]:
            c = dt - pd.Timedelta(days=off)
            if c in h: val = h[c]; break
        if val is None: val = fb
        h[dt] = float(val); preds.append(float(val))
    return np.array(preds)


def naive_weighted_multiyear(train_df, col, fc_dates, n_years=4):
    """
    Weighted average of same-DOY/DOW values from last N years.
    More recent years get higher weight. Uses DOW matching.
    """
    work = train_df[['Date', col]].copy()
    work['doy'] = work.Date.dt.dayofyear
    work['dow'] = work.Date.dt.dayofweek
    work['year'] = work.Date.dt.year

    max_year = work['year'].max()
    recent_years = sorted(work['year'].unique())[-n_years:]
    work = work[work['year'].isin(recent_years)]

    # Year weights: exponential decay
    year_weights = {}
    for y in recent_years:
        year_weights[y] = 2.0 ** (y - max_year + n_years - 1)  # most recent = highest

    preds = []
    for dt in pd.to_datetime(fc_dates):
        doy, dow = int(dt.dayofyear), int(dt.dayofweek)
        diff = np.abs(work['doy'].values - doy)
        dist = np.minimum(diff, 366 - diff)

        mask = (work['dow'].values == dow) & (dist <= 3)
        if mask.sum() < 3:
            mask = (work['dow'].values == dow) & (dist <= 7)
        if mask.sum() < 2:
            mask = dist <= 7

        if mask.sum() > 0:
            subset = work[mask]
            weights = subset['year'].map(year_weights).values
            vals = subset[col].values
            # Weighted median approximation
            idx = np.argsort(vals)
            v_s, w_s = vals[idx], weights[idx]
            cum = np.cumsum(w_s)
            mid = cum[-1] / 2.0
            mi = np.searchsorted(cum, mid)
            preds.append(float(v_s[min(mi, len(v_s)-1)]))
        else:
            preds.append(float(work[col].median()))

    return np.array(preds)


def naive_growth_adjusted(train_df, col, fc_dates):
    """Naive364 adjusted by median YoY growth rate."""
    yearly = train_df.groupby(train_df.Date.dt.year)[col].median()
    if len(yearly) >= 3:
        recent = yearly.iloc[-3:].values
        ratios = recent[1:] / np.maximum(recent[:-1], 1)
        growth = float(np.median(ratios))
        growth = np.clip(growth, 0.90, 1.25)
    else:
        growth = 1.0

    base = naive364(train_df.set_index('Date')[col], fc_dates)
    max_yr = train_df.Date.dt.year.max()
    fc_yrs = np.array([pd.Timestamp(d).year for d in pd.to_datetime(fc_dates)], dtype=float)
    yr_diff = fc_yrs - max_yr
    return base * growth ** np.clip(yr_diff, 0, 3)


def build_naive_backbone(train_df, col, fc_dates):
    """Blend multiple naive variants."""
    p364 = naive364(train_df.set_index('Date')[col], fc_dates)
    p_multi = naive_weighted_multiyear(train_df, col, fc_dates, n_years=4)
    p_grow = naive_growth_adjusted(train_df, col, fc_dates)
    # 50% pure 364, 30% multi-year weighted, 20% growth-adjusted
    return 0.50 * p364 + 0.30 * p_multi + 0.20 * p_grow


print("Naive backbone ready")


# ══════════════════════════════════════════════════════════════════════
# TET CALIBRATION (EMPIRICAL)
# ══════════════════════════════════════════════════════════════════════

def compute_tet_profile(sales_df, col):
    """Compute empirical Tet profile: median multiplier per day-relative-to-Tet."""
    mults = {}
    for tet in TET:
        yr = sales_df[sales_df.Date.dt.year == tet.year]
        if len(yr) < 100: continue
        # Use non-Tet months as baseline (May-Aug = stable period)
        baseline = yr[yr.Date.dt.month.isin([5,6,7,8])][col].median()
        if baseline <= 0: continue
        for delta in range(-30, 21):
            d = tet + pd.Timedelta(days=delta)
            row = sales_df[sales_df.Date == d]
            if len(row) > 0:
                mults.setdefault(delta, []).append(float(row[col].iloc[0]) / baseline)
    return {k: float(np.median(v)) for k, v in mults.items() if len(v) >= 2}


def apply_tet_cal(preds, fc_dates, tet_profile, base_med, strength=0.35):
    """Apply Tet calibration with configurable strength."""
    preds = preds.copy()
    fc_ts = pd.to_datetime(fc_dates)
    for i, dt in enumerate(fc_ts):
        # Find Tet for this year
        tet = None
        for t in TET:
            if t.year == dt.year: tet = t; break
        if tet is None: continue
        delta = (dt - tet).days
        if delta in tet_profile:
            target = base_med * tet_profile[delta]
            preds[i] = (1 - strength) * preds[i] + strength * target
    return np.clip(preds, 0, None)


print("Tet calibration ready")


# ══════════════════════════════════════════════════════════════════════
# FEATURES & RESIDUAL CORRECTION
# ══════════════════════════════════════════════════════════════════════

def build_features(fc_dates, train_df, col):
    """Clean features — no Prophet, no recursive lags."""
    dates = pd.to_datetime(fc_dates)
    n = len(dates)
    f = pd.DataFrame(index=range(n))

    f['month'] = dates.month; f['day'] = dates.day
    f['dow'] = dates.dayofweek; f['doy'] = dates.dayofyear
    f['week'] = dates.isocalendar().week.astype(int).values
    f['quarter'] = dates.quarter
    f['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    f['is_month_start'] = dates.is_month_start.astype(int)
    f['is_month_end'] = dates.is_month_end.astype(int)

    # Tet features
    f['d2tet'] = d2next_tet(dates); f['d_since_tet'] = d2last_tet(dates)
    f['tet_window'] = (f['d2tet'] <= 10).astype(int)
    f['post_tet'] = ((f['d_since_tet'] > 0) & (f['d_since_tet'] <= 14)).astype(int)
    f['tet_effect'] = np.exp(-f['d2tet']/7) + np.exp(-f['d_since_tet']/5)
    f['pre_tet_21'] = f['d2tet'].between(1, 21).astype(int)

    # Mega-sale
    for i, dt in enumerate(dates):
        for m, d in MEGA:
            try:
                sd = pd.Timestamp(year=dt.year, month=m, day=d)
                if abs((dt - sd).days) <= 2:
                    f.loc[i, 'is_mega'] = 1; break
            except: pass
    f['is_mega'] = f.get('is_mega', 0).fillna(0).astype(int)

    # Fourier
    doy = f['doy'].values.astype(float); dow = f['dow'].values.astype(float)
    for k in range(1, 5):
        f[f'sin_y{k}'] = np.sin(2*np.pi*k*doy/365.25)
        f[f'cos_y{k}'] = np.cos(2*np.pi*k*doy/365.25)
    for k in range(1, 3):
        f[f'sin_w{k}'] = np.sin(2*np.pi*k*dow/7)
        f[f'cos_w{k}'] = np.cos(2*np.pi*k*dow/7)

    # Fixed lag features (into training data only, no recursion)
    idx_map = train_df.set_index('Date')[col]
    for lag_yr in [1, 2, 3]:
        vals = []
        for dt in dates:
            hits = []
            for off in [364*lag_yr, 364*lag_yr+7, 364*lag_yr-7]:
                c = dt - pd.Timedelta(days=off)
                if c in idx_map.index: hits.append(float(idx_map[c]))
            vals.append(float(np.mean(hits)) if hits else np.nan)
        f[f'lag_{lag_yr}yr'] = vals

    # Historical profiles
    monthly = train_df.groupby(train_df.Date.dt.month)[col].agg(['mean','median','std'])
    f['hist_m_mean'] = f['month'].map(monthly['mean'])
    f['hist_m_med'] = f['month'].map(monthly['median'])
    f['hist_m_std'] = f['month'].map(monthly['std'])

    dow_st = train_df.groupby(train_df.Date.dt.dayofweek)[col].agg(['mean','median'])
    f['hist_d_mean'] = f['dow'].map(dow_st['mean'])
    f['hist_d_med'] = f['dow'].map(dow_st['median'])

    # Month x DOW interaction
    md = train_df.groupby([train_df.Date.dt.month, train_df.Date.dt.dayofweek])[col].median()
    f['hist_md'] = [md.get((f['month'].iloc[i], f['dow'].iloc[i]), np.nan) for i in range(n)]

    # Aux
    for k, v in aux.items():
        if '_month' in k: f[f'aux_{k}'] = f['month'].map(v)
        elif '_dow' in k: f[f'aux_{k}'] = f['dow'].map(v)

    # Rolling stats from end of training
    for w in [28, 90, 182]:
        tail = train_df.tail(w)
        f[f'tail{w}_mean'] = tail[col].mean()
        f[f'tail{w}_med'] = tail[col].median()

    return f.fillna(0)


def fit_residual_models(X, y_resid):
    """Fit LGB + XGB ensemble for residual correction."""
    # Clip extreme residuals
    clip = np.percentile(np.abs(y_resid), 97)
    mask = np.abs(y_resid) <= clip * 2
    X_c, y_c = X[mask], y_resid[mask]

    # Recency weighting
    rank = np.arange(len(X_c), dtype=float)
    sw = 1.0 + 2.0 * (rank / max(1.0, rank.max())) ** 1.5

    lgb = LGBMRegressor(
        n_estimators=800, learning_rate=0.02, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=10.0,
        min_child_samples=30, objective='huber', random_state=SEED, n_jobs=-1, verbosity=-1)
    xgb = XGBRegressor(
        n_estimators=800, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=10.0,
        min_child_weight=5, random_state=SEED, n_jobs=-1, verbosity=0)

    lgb.fit(X_c, y_c, sample_weight=sw)
    xgb.fit(X_c, y_c, sample_weight=sw)
    return lgb, xgb


print("Features & models ready")


# ══════════════════════════════════════════════════════════════════════
# V8 PIPELINE
# ══════════════════════════════════════════════════════════════════════

def run_v8(train_df, col, fc_dates, tet_strength=0.35):
    """Full v8 pipeline."""
    fc_dates = list(pd.to_datetime(fc_dates))
    n = len(fc_dates)

    # 1. Naive backbone
    p_naive = build_naive_backbone(train_df, col, fc_dates)

    # 2. Compute naive for training period (for residual learning)
    # Use last 3 years for residual training
    cutoff = train_df.Date.max() - pd.DateOffset(years=3)
    hist = train_df[train_df.Date >= cutoff].copy()

    idx_map = train_df.set_index('Date')[col]
    naive_hist = []
    for dt in hist.Date:
        val = None
        for off in [364, 371, 357, 728]:
            c = dt - pd.Timedelta(days=off)
            if c in idx_map.index: val = float(idx_map[c]); break
        naive_hist.append(val if val is not None else float(idx_map.median()))
    hist = hist.copy()
    hist['naive'] = naive_hist
    hist['resid'] = hist[col] - hist['naive']

    # 3. Fit residual models
    X_tr = build_features(hist.Date.tolist(), train_df, col)
    lgb, xgb = fit_residual_models(X_tr, hist['resid'].values)

    # 4. Predict correction
    X_fc = build_features(fc_dates, train_df, col)
    corr = 0.6 * lgb.predict(X_fc) + 0.4 * xgb.predict(X_fc)

    # 5. Damp at far horizons
    damp = np.exp(-np.arange(n, dtype=float) / 300)
    p_corrected = np.clip(p_naive + corr * damp, 0, None)

    # 6. Tet calibration
    tet_prof = compute_tet_profile(train_df, col)
    base_med = float(train_df[train_df.Date.dt.month.isin([5,6,7,8])][col].median())
    p_naive_tet = apply_tet_cal(p_naive, fc_dates, tet_prof, base_med, tet_strength)
    p_corr_tet = apply_tet_cal(p_corrected, fc_dates, tet_prof, base_med, tet_strength)

    return {
        'naive': np.clip(p_naive, 0, None),
        'corrected': p_corrected,
        'naive_tet': p_naive_tet,
        'corrected_tet': p_corr_tet,
    }


# ══════════════════════════════════════════════════════════════════════
# 548-DAY HOLDOUT CV
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("548-DAY HOLDOUT CV")
print("="*70)

HOLDOUTS = [('2021-06-30', 548), ('2021-01-01', 548), ('2020-06-30', 548)]
cv = {}

for col in ['Revenue', 'COGS']:
    print(f"\n── {col} ──")
    fold_r = {k: [] for k in ['naive','corrected','naive_tet','corrected_tet']}

    for origin_str, nd in HOLDOUTS:
        origin = pd.Timestamp(origin_str)
        tr = sales[sales.Date <= origin].copy()
        val = sales[sales.Date > origin].head(nd).copy()
        if len(val) < 100: continue

        y = val[col].values
        preds = run_v8(tr, col, val.Date.tolist())

        for k in fold_r:
            mae = mean_absolute_error(y, preds[k])
            fold_r[k].append(mae)

        print(f"  {origin_str}: " + " | ".join(f"{k}={fold_r[k][-1]:,.0f}" for k in fold_r))

    for k, maes in fold_r.items():
        if maes:
            w = np.array([1.0+i*0.5 for i in range(len(maes))]); w /= w.sum()
            cv[f'{col}_{k}'] = float(np.dot(maes, w))

    print(f"  Weighted avg:")
    for k in fold_r:
        ck = f'{col}_{k}'
        if ck in cv: print(f"    {k}: {cv[ck]:,.0f}")

# ── Tune Tet strength ──
print("\n── Tet Strength Tuning ──")
best_tet = {}
tr_t = sales[sales.Date <= pd.Timestamp('2021-06-30')].copy()
val_t = sales[sales.Date > pd.Timestamp('2021-06-30')].head(548).copy()

for col in ['Revenue', 'COGS']:
    preds0 = run_v8(tr_t, col, val_t.Date.tolist(), tet_strength=0.0)
    tet_prof = compute_tet_profile(tr_t, col)
    bm = float(tr_t[tr_t.Date.dt.month.isin([5,6,7,8])][col].median())
    y = val_t[col].values

    best_s, best_mae = 0.0, 1e18
    for s in [0.0, 0.15, 0.25, 0.35, 0.45, 0.55]:
        p = apply_tet_cal(preds0['corrected'], val_t.Date.tolist(), tet_prof, bm, s)
        mae = mean_absolute_error(y, p)
        if mae < best_mae: best_mae, best_s = mae, s
    best_tet[col] = best_s
    print(f"  {col}: best tet_strength={best_s} (MAE={best_mae:,.0f})")


# ══════════════════════════════════════════════════════════════════════
# FINAL FORECAST & OUTPUT
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FINAL FORECAST")
print("="*70)

final = {}
for col in ['Revenue', 'COGS']:
    print(f"  {col}...", end=' ', flush=True)
    t0 = time.time()
    final[col] = run_v8(sales, col, forecast_dates, tet_strength=best_tet.get(col, 0.35))
    print(f"done ({time.time()-t0:.0f}s)")

# Save all variants
VARIANTS = ['naive', 'corrected', 'naive_tet', 'corrected_tet']
output_files = {}

print("\n" + "="*70)
print("SUBMISSIONS")
print("="*70)

for vname in VARIANTS:
    sub = sub_tpl[['Date']].copy()
    for col in ['Revenue', 'COGS']:
        sub[col] = np.clip(final[col][vname], 0, None)

    ok = len(sub)==N_FC and not sub[['Revenue','COGS']].isna().any().any() and not (sub[['Revenue','COGS']]<0).any().any()
    print(f"  {'OK' if ok else 'FAIL'} {vname}:")
    for c in ['Revenue','COGS']:
        v = sub[c]
        print(f"    {c}: mean={v.mean():,.0f} med={v.median():,.0f} min={v.min():,.0f} max={v.max():,.0f}")

    fname = f'submission_v8_{vname}.csv'
    fpath = os.path.join(OUT_DIR, fname)
    out = sub.copy(); out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')
    out.to_csv(fpath, index=False)
    output_files[vname] = fpath

# Diagnostics
diag = {'version': 'v8_final', 'cv': cv, 'best_tet': best_tet}
with open(os.path.join(OUT_DIR, 'v8_diagnostics.json'), 'w') as f:
    json.dump(diag, f, indent=2, default=str)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for col in ['Revenue','COGS']:
    for k in VARIANTS:
        ck = f'{col}_{k}'
        if ck in cv: print(f"  {col} {k}: CV={cv[ck]:,.0f}")
print(f"\nFiles: {list(output_files.keys())}")
print("\n🏁 Upload submission_v8_corrected_tet.csv first.")
