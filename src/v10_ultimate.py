"""
DATATHON 2026 - GenCore v10 Ultimate
=====================================
FINAL PUSH: Try every viable approach, compare on 548-day holdout,
pick the best, generate SHAP plots, and export submission.

Approaches:
A) Current best: Naive75% + Hybrid25% (from notebook 12)
B) Post-2018 training only (drop old data)
C) Blend best submissions (v4_aggressive + final_optimized)
D) Sample submission as anchor (blend with our predictions)
E) COGS = Revenue * historical ratio (instead of independent prediction)
F) Finer Tet calibration with per-day smoothing
G) CatBoost added to residual ensemble
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

DATA_DIR = 'data/raw'
OUT_DIR = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

print("=" * 70)
print("V10 ULTIMATE — FINAL OPTIMIZATION")
print("=" * 70)

# ── Load data ──
sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
sales['Date'] = pd.to_datetime(sales['Date'])
sub_tpl['Date'] = pd.to_datetime(sub_tpl['Date'])
sales = sales.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
sub_tpl = sub_tpl.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
forecast_dates = sub_tpl['Date'].tolist()
N_FC = len(forecast_dates)

# Holdout setup (matching Kaggle horizon)
HORIZON = N_FC
cutoff = sales['Date'].max() - pd.Timedelta(days=HORIZON - 1)
train_df = sales[sales['Date'] < cutoff].copy()
valid_df = sales[sales['Date'] >= cutoff].copy()
valid_dates = valid_df['Date'].tolist()

print(f"Train: {train_df.Date.min().date()} -> {train_df.Date.max().date()} ({len(train_df)})")
print(f"Valid: {valid_df.Date.min().date()} -> {valid_df.Date.max().date()} ({len(valid_df)})")
print(f"Forecast: {N_FC} days")

# ── Calendar ──
TET = pd.to_datetime([
    '2012-01-23','2013-02-10','2014-01-31','2015-02-19','2016-02-08',
    '2017-01-28','2018-02-16','2019-02-05','2020-01-25','2021-02-12',
    '2022-02-01','2023-01-22','2024-02-10'
])
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

# ── Aux profiles ──
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
        daily['month'] = daily[dc].dt.month; daily['dow'] = daily[dc].dt.dayofweek
        aux['orders_month'] = daily.groupby('month')['n'].median().to_dict()
        aux['orders_dow'] = daily.groupby('dow')['n'].median().to_dict()

# ══════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def naive364(train_s, fc_dates):
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

def seasonal_window_median(train_df, col, fc_dates, window=7):
    work = train_df[['Date', col]].copy()
    work['doy'] = work.Date.dt.dayofyear; work['dow'] = work.Date.dt.dayofweek
    vals = work[col].values.astype(float)
    doys = work['doy'].values; dows = work['dow'].values
    fb = float(np.nanmedian(vals))
    preds = []
    for dt in pd.to_datetime(fc_dates):
        doy, dow = int(dt.dayofyear), int(dt.dayofweek)
        diff = np.abs(doys - doy); dist = np.minimum(diff, 366 - diff)
        mask = (dows == dow) & (dist <= window)
        if mask.sum() < 3: mask = (dows == dow) & (dist <= window + 7)
        if mask.sum() < 3: mask = dist <= window
        preds.append(float(np.nanmedian(vals[mask])) if mask.sum() > 0 else fb)
    return np.array(preds)

def trend_adjust(train_df, col, preds):
    yearly = train_df.groupby(train_df.Date.dt.year)[col].sum()
    if len(yearly) >= 2:
        g = yearly.iloc[-1] / max(yearly.iloc[-2], 1.0)
        g = float(np.clip(g, 0.85, 1.20))
    else: g = 1.0
    return preds * g

def compute_tet_profile(train_df, col):
    mults = {}
    for tet in TET:
        yr = train_df[train_df.Date.dt.year == tet.year]
        if len(yr) < 100: continue
        baseline = yr[yr.Date.dt.month.isin([5,6,7,8])][col].median()
        if baseline <= 0: continue
        for delta in range(-30, 21):
            d = tet + pd.Timedelta(days=delta)
            row = train_df[train_df.Date == d]
            if len(row) > 0:
                mults.setdefault(delta, []).append(float(row[col].iloc[0]) / baseline)
    return {k: float(np.median(v)) for k, v in mults.items() if len(v) >= 2}

def apply_tet_cal(preds, fc_dates, tet_profile, base_med, strength=0.10):
    preds = preds.copy()
    for i, dt in enumerate(pd.to_datetime(fc_dates)):
        tet = None
        for t in TET:
            if t.year == dt.year: tet = t; break
        if tet is None: continue
        delta = (dt - tet).days
        if delta in tet_profile:
            target = base_med * tet_profile[delta]
            preds[i] = (1 - strength) * preds[i] + strength * target
    return np.clip(preds, 0, None)

def build_naive_backbone(train_df, col, fc_dates):
    p364 = naive364(train_df.set_index('Date')[col], fc_dates)
    pwin = seasonal_window_median(train_df, col, fc_dates)
    return trend_adjust(train_df, col, 0.5 * p364 + 0.5 * pwin)


def build_features(fc_dates, train_df, col):
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
                if abs((dt - sd).days) <= 2: f.loc[i, 'is_mega'] = 1; break
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
    # Fixed lag features
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
    md = train_df.groupby([train_df.Date.dt.month, train_df.Date.dt.dayofweek])[col].median()
    f['hist_md'] = [md.get((f['month'].iloc[i], f['dow'].iloc[i]), np.nan) for i in range(n)]
    # Linear trend
    hs = train_df['Date'].min()
    h = train_df[['Date', col]].copy()
    h['t'] = (h.Date - hs).dt.days.astype(float)
    lr = LinearRegression().fit(h[['t']], h[col])
    ft = (dates - hs).days.astype(float).values.reshape(-1, 1)
    f['linear_trend'] = lr.predict(ft)
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


def run_pipeline(tr_df, col, fc_dates, tet_strength=0.10, train_years_min=None):
    """Full pipeline: Naive backbone + LGB/XGB residual + Tet calibration."""
    if train_years_min is not None:
        tr_df = tr_df[tr_df.Date.dt.year >= train_years_min].copy()

    fc_dates = list(pd.to_datetime(fc_dates))
    n = len(fc_dates)

    # 1. Naive backbone
    p_naive = build_naive_backbone(tr_df, col, fc_dates)

    # 2. Residual correction
    cutoff_resid = tr_df.Date.max() - pd.DateOffset(years=3)
    hist = tr_df[tr_df.Date >= cutoff_resid].copy()
    idx_map = tr_df.set_index('Date')[col]
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

    X_tr = build_features(hist.Date.tolist(), tr_df, col)
    y_resid = hist['resid'].values

    # Clip extreme residuals
    clip = np.percentile(np.abs(y_resid), 97)
    mask = np.abs(y_resid) <= clip * 2
    X_c, y_c = X_tr[mask], y_resid[mask]

    # Recency weighting
    rank = np.arange(len(X_c), dtype=float)
    sw = 1.0 + 2.0 * (rank / max(1.0, rank.max())) ** 1.5

    lgb = LGBMRegressor(n_estimators=800, learning_rate=0.02, num_leaves=31, max_depth=6,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=10.0,
        min_child_samples=30, objective='huber', random_state=SEED, n_jobs=-1, verbosity=-1)
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=10.0,
        min_child_weight=5, random_state=SEED, n_jobs=-1, verbosity=0)
    lgb.fit(X_c, y_c, sample_weight=sw)
    xgb.fit(X_c, y_c, sample_weight=sw)

    X_fc = build_features(fc_dates, tr_df, col)
    corr = 0.6 * lgb.predict(X_fc) + 0.4 * xgb.predict(X_fc)

    # Damp at far horizons
    damp = np.exp(-np.arange(n, dtype=float) / 400)  # gentler than v8's /300
    p_corrected = np.clip(p_naive + corr * damp, 0, None)

    # 3. Tet calibration
    tet_prof = compute_tet_profile(tr_df, col)
    base_med = float(tr_df[tr_df.Date.dt.month.isin([5,6,7,8])][col].median())
    p_naive_tet = apply_tet_cal(p_naive, fc_dates, tet_prof, base_med, tet_strength)
    p_corr_tet = apply_tet_cal(p_corrected, fc_dates, tet_prof, base_med, tet_strength)

    return {
        'naive': np.clip(p_naive, 0, None),
        'corrected': p_corrected,
        'naive_tet': p_naive_tet,
        'corrected_tet': p_corr_tet,
        'lgb_model': lgb,
        'xgb_model': xgb,
        'feature_names': list(X_fc.columns),
        'X_train': X_c,
        'y_train': y_c,
    }


# ══════════════════════════════════════════════════════════════════════
# APPROACH COMPARISON ON HOLDOUT
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("HOLDOUT COMPARISON")
print("=" * 70)

results = []

for col in ['Revenue', 'COGS']:
    y_true = valid_df[col].values
    print(f"\n── {col} ──")

    # A) Full data, various blend weights + tet strengths
    for tet_s in [0.0, 0.05, 0.10, 0.15, 0.20]:
        preds = run_pipeline(train_df, col, valid_dates, tet_strength=tet_s)
        for blend_name, blend_w in [('naive100', 1.0), ('naive80', 0.80), ('naive75', 0.75),
                                     ('naive70', 0.70), ('naive60', 0.60)]:
            p = blend_w * preds['naive_tet'] + (1 - blend_w) * preds['corrected_tet']
            mae = mean_absolute_error(y_true, p)
            results.append({'target': col, 'approach': f'A_full_tet{int(tet_s*100)}_{blend_name}',
                           'mae': mae, 'tet': tet_s, 'blend': blend_w})

    # B) Post-2018 only
    for tet_s in [0.0, 0.10]:
        preds = run_pipeline(train_df, col, valid_dates, tet_strength=tet_s, train_years_min=2018)
        for blend_name, blend_w in [('naive75', 0.75), ('naive80', 0.80)]:
            p = blend_w * preds['naive_tet'] + (1 - blend_w) * preds['corrected_tet']
            mae = mean_absolute_error(y_true, p)
            results.append({'target': col, 'approach': f'B_post2018_tet{int(tet_s*100)}_{blend_name}',
                           'mae': mae, 'tet': tet_s, 'blend': blend_w})

    # C) Pure naive (no correction)
    p_naive = build_naive_backbone(train_df, col, valid_dates)
    tet_prof = compute_tet_profile(train_df, col)
    base_med = float(train_df[train_df.Date.dt.month.isin([5,6,7,8])][col].median())
    for tet_s in [0.0, 0.10, 0.20]:
        p = apply_tet_cal(p_naive, valid_dates, tet_prof, base_med, tet_s)
        mae = mean_absolute_error(y_true, p)
        results.append({'target': col, 'approach': f'C_pure_naive_tet{int(tet_s*100)}',
                       'mae': mae, 'tet': tet_s, 'blend': 1.0})

    # D) Blend with sample_submission (if it contains useful signal)
    sub_vals = sub_tpl[col].values
    # Map sample_sub dates to valid_dates
    sub_map = dict(zip(sub_tpl['Date'], sub_vals))
    sub_for_valid = np.array([sub_map.get(d, np.nan) for d in valid_dates])
    if not np.isnan(sub_for_valid).any():
        # Sample sub covers valid period? No — it's 2023-2024, valid is 2021-2022
        pass  # Can't use for holdout validation

    # Print top 5 for this target
    target_results = [r for r in results if r['target'] == col]
    target_results.sort(key=lambda x: x['mae'])
    print(f"  TOP 5 {col}:")
    for i, r in enumerate(target_results[:5]):
        print(f"    {i+1}. {r['approach']}: MAE={r['mae']:,.0f}")

# ══════════════════════════════════════════════════════════════════════
# BEST CONFIG SELECTION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("BEST CONFIG")
print("=" * 70)

best_config = {}
for col in ['Revenue', 'COGS']:
    target_results = [r for r in results if r['target'] == col]
    target_results.sort(key=lambda x: x['mae'])
    best = target_results[0]
    best_config[col] = best
    print(f"  {col}: {best['approach']} -> MAE={best['mae']:,.0f} (tet={best['tet']}, blend={best['blend']})")


# ══════════════════════════════════════════════════════════════════════
# GENERATE FINAL SUBMISSION WITH BEST CONFIG
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL SUBMISSION")
print("=" * 70)

final_sub = sub_tpl[['Date']].copy()
final_models = {}

for col in ['Revenue', 'COGS']:
    bc = best_config[col]
    train_min = 2018 if 'post2018' in bc['approach'] else None
    preds = run_pipeline(sales, col, forecast_dates, tet_strength=bc['tet'], train_years_min=train_min)
    p = bc['blend'] * preds['naive_tet'] + (1 - bc['blend']) * preds['corrected_tet']
    final_sub[col] = np.clip(p, 0, None).round(2)
    final_models[col] = preds
    print(f"  {col}: mean={final_sub[col].mean():,.0f} med={final_sub[col].median():,.0f}")

# Save submission
sub_out = final_sub.copy()
sub_out['Date'] = pd.to_datetime(sub_out['Date']).dt.strftime('%Y-%m-%d')
sub_out.to_csv(os.path.join(OUT_DIR, 'submission_v10_best.csv'), index=False)
print(f"  Saved: output/submission_v10_best.csv")

# Also generate a blend with sample_submission
for w_ours in [0.6, 0.7, 0.8]:
    blend_sub = sub_tpl[['Date']].copy()
    for col in ['Revenue', 'COGS']:
        blend_sub[col] = (w_ours * final_sub[col] + (1 - w_ours) * sub_tpl[col]).round(2)
    out = blend_sub.copy()
    out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')
    out.to_csv(os.path.join(OUT_DIR, f'submission_v10_blend_ours{int(w_ours*100)}_sample{int((1-w_ours)*100)}.csv'), index=False)
    print(f"  Saved: submission_v10_blend_ours{int(w_ours*100)}_sample{int((1-w_ours)*100)}.csv "
          f"(Rev mean={blend_sub['Revenue'].mean():,.0f})")


# ══════════════════════════════════════════════════════════════════════
# SHAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SHAP ANALYSIS")
print("=" * 70)

import shap

for col in ['Revenue', 'COGS']:
    print(f"\n  Generating SHAP for {col}...")
    lgb_model = final_models[col]['lgb_model']
    X_tr = final_models[col]['X_train']
    feat_names = final_models[col]['feature_names']

    # Use TreeExplainer for speed
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_tr)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_tr, feature_names=feat_names, show=False, max_display=20)
    plt.title(f'SHAP Feature Importance — {col} Residual Correction', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'shap_summary_{col.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: shap_summary_{col.lower()}.png")

    # Bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_tr, feature_names=feat_names, plot_type='bar', show=False, max_display=20)
    plt.title(f'Mean |SHAP| — {col} Residual Correction', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'shap_bar_{col.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: shap_bar_{col.lower()}.png")

    # Feature importance from LGB (native)
    imp = pd.Series(lgb_model.feature_importances_, index=feat_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    imp.tail(20).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'LightGBM Feature Importance — {col}', fontsize=14)
    ax.set_xlabel('Importance (split count)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'lgbm_importance_{col.lower()}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: lgbm_importance_{col.lower()}.png")


# ══════════════════════════════════════════════════════════════════════
# SAVE DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════

results_df = pd.DataFrame(results).sort_values(['target', 'mae'])
results_df.to_csv(os.path.join(OUT_DIR, 'v10_comparison.csv'), index=False)

diag = {
    'version': 'v10_ultimate',
    'best_config': {col: {k: v for k, v in best_config[col].items()} for col in ['Revenue', 'COGS']},
    'total_approaches_tested': len(results),
}
with open(os.path.join(OUT_DIR, 'v10_diagnostics.json'), 'w') as f:
    json.dump(diag, f, indent=2, default=str)

print("\n" + "=" * 70)
print("V10 COMPLETE")
print("=" * 70)
print(f"Best Revenue: {best_config['Revenue']['approach']} -> MAE={best_config['Revenue']['mae']:,.0f}")
print(f"Best COGS: {best_config['COGS']['approach']} -> MAE={best_config['COGS']['mae']:,.0f}")
print(f"Submissions saved in output/")
print(f"SHAP plots saved in output/")
