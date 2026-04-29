"""
V9: Optuna-tuned LGB+XGB+CatBoost ensemble on top of v8 naive backbone.
"""
import os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

SEED = 42; np.random.seed(SEED)
DATA_DIR = 'data/raw'
for c in ['data/raw', '../data/raw']:
    if os.path.isfile(os.path.join(c, 'sales.csv')): DATA_DIR = c; break
OUT_DIR = 'output'; os.makedirs(OUT_DIR, exist_ok=True)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
sales['Date'] = pd.to_datetime(sales['Date']); sub_tpl['Date'] = pd.to_datetime(sub_tpl['Date'])
sales = sales.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
sub_tpl = sub_tpl.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
forecast_dates = sub_tpl['Date'].tolist(); N_FC = len(forecast_dates)
print(f"Train: {sales.Date.min().date()}->{sales.Date.max().date()} ({len(sales)}), Forecast: {N_FC}")

# Calendar
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

# Aux data
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

# ── Naive backbone (same as v8) ──
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

def naive_weighted(train_df, col, fc_dates, n_years=4):
    work = train_df[['Date', col]].copy()
    work['doy'] = work.Date.dt.dayofyear; work['dow'] = work.Date.dt.dayofweek; work['year'] = work.Date.dt.year
    max_year = work['year'].max()
    recent_years = sorted(work['year'].unique())[-n_years:]
    work = work[work['year'].isin(recent_years)]
    year_weights = {y: 2.0 ** (y - max_year + n_years - 1) for y in recent_years}
    preds = []
    for dt in pd.to_datetime(fc_dates):
        doy, dow = int(dt.dayofyear), int(dt.dayofweek)
        diff = np.abs(work['doy'].values - doy); dist = np.minimum(diff, 366 - diff)
        mask = (work['dow'].values == dow) & (dist <= 3)
        if mask.sum() < 3: mask = (work['dow'].values == dow) & (dist <= 7)
        if mask.sum() < 2: mask = dist <= 7
        if mask.sum() > 0:
            subset = work[mask]; weights = subset['year'].map(year_weights).values; vals = subset[col].values
            idx = np.argsort(vals); v_s, w_s = vals[idx], weights[idx]
            cum = np.cumsum(w_s); mi = np.searchsorted(cum, cum[-1]/2.0)
            preds.append(float(v_s[min(mi, len(v_s)-1)]))
        else: preds.append(float(work[col].median()))
    return np.array(preds)

def naive_growth(train_df, col, fc_dates):
    yearly = train_df.groupby(train_df.Date.dt.year)[col].median()
    growth = 1.0
    if len(yearly) >= 3:
        recent = yearly.iloc[-3:].values; ratios = recent[1:] / np.maximum(recent[:-1], 1)
        growth = np.clip(float(np.median(ratios)), 0.90, 1.25)
    base = naive364(train_df.set_index('Date')[col], fc_dates)
    max_yr = train_df.Date.dt.year.max()
    fc_yrs = np.array([pd.Timestamp(d).year for d in pd.to_datetime(fc_dates)], dtype=float)
    return base * growth ** np.clip(fc_yrs - max_yr, 0, 3)

def build_naive(train_df, col, fc_dates):
    p364 = naive364(train_df.set_index('Date')[col], fc_dates)
    p_multi = naive_weighted(train_df, col, fc_dates)
    p_grow = naive_growth(train_df, col, fc_dates)
    return 0.50 * p364 + 0.30 * p_multi + 0.20 * p_grow

# ── Tet ──
def compute_tet_profile(sales_df, col):
    mults = {}
    for tet in TET:
        yr = sales_df[sales_df.Date.dt.year == tet.year]
        if len(yr) < 100: continue
        baseline = yr[yr.Date.dt.month.isin([5,6,7,8])][col].median()
        if baseline <= 0: continue
        for delta in range(-30, 21):
            d = tet + pd.Timedelta(days=delta)
            row = sales_df[sales_df.Date == d]
            if len(row) > 0: mults.setdefault(delta, []).append(float(row[col].iloc[0]) / baseline)
    return {k: float(np.median(v)) for k, v in mults.items() if len(v) >= 2}

def apply_tet(preds, fc_dates, tet_profile, base_med, strength=0.35):
    preds = preds.copy(); fc_ts = pd.to_datetime(fc_dates)
    for i, dt in enumerate(fc_ts):
        tet = None
        for t in TET:
            if t.year == dt.year: tet = t; break
        if tet is None: continue
        delta = (dt - tet).days
        if delta in tet_profile:
            preds[i] = (1 - strength) * preds[i] + strength * base_med * tet_profile[delta]
    return np.clip(preds, 0, None)

# ── Features (enhanced) ──
def build_features(fc_dates, train_df, col):
    dates = pd.to_datetime(fc_dates); n = len(dates)
    f = pd.DataFrame(index=range(n))
    f['month'] = dates.month; f['day'] = dates.day
    f['dow'] = dates.dayofweek; f['doy'] = dates.dayofyear
    f['week'] = dates.isocalendar().week.astype(int).values
    f['quarter'] = dates.quarter
    f['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    f['is_month_start'] = dates.is_month_start.astype(int)
    f['is_month_end'] = dates.is_month_end.astype(int)
    # Tet
    f['d2tet'] = d2next_tet(dates); f['d_since_tet'] = d2last_tet(dates)
    f['tet_window'] = (f['d2tet'] <= 10).astype(int)
    f['post_tet'] = ((f['d_since_tet'] > 0) & (f['d_since_tet'] <= 14)).astype(int)
    f['tet_effect'] = np.exp(-f['d2tet']/7) + np.exp(-f['d_since_tet']/5)
    f['pre_tet_21'] = f['d2tet'].between(1, 21).astype(int)
    # Mega
    for i, dt in enumerate(dates):
        for m, d in MEGA:
            try:
                sd = pd.Timestamp(year=dt.year, month=m, day=d)
                if abs((dt - sd).days) <= 2: f.loc[i, 'is_mega'] = 1; break
            except: pass
    f['is_mega'] = f.get('is_mega', 0).fillna(0).astype(int)
    # Fourier
    doy = f['doy'].values.astype(float); dow = f['dow'].values.astype(float)
    for k in range(1, 6):
        f[f'sin_y{k}'] = np.sin(2*np.pi*k*doy/365.25)
        f[f'cos_y{k}'] = np.cos(2*np.pi*k*doy/365.25)
    for k in range(1, 4):
        f[f'sin_w{k}'] = np.sin(2*np.pi*k*dow/7)
        f[f'cos_w{k}'] = np.cos(2*np.pi*k*dow/7)
    # Lag features
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
    # YoY ratio
    if 'lag_1yr' in f.columns and 'lag_2yr' in f.columns:
        f['yoy_ratio'] = f['lag_1yr'] / f['lag_2yr'].replace(0, np.nan)
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
    # Aux
    for k, v in aux.items():
        if '_month' in k: f[f'aux_{k}'] = f['month'].map(v)
        elif '_dow' in k: f[f'aux_{k}'] = f['dow'].map(v)
    # Rolling from train tail
    for w in [7, 14, 28, 90, 182]:
        tail = train_df.tail(w)
        f[f'tail{w}_mean'] = tail[col].mean()
        f[f'tail{w}_med'] = tail[col].median()
        if w <= 28: f[f'tail{w}_std'] = tail[col].std()
    return f.fillna(0)

# ── Optuna tuning ──
def tune_and_fit(X_tr, y_resid, n_trials=40):
    clip = np.percentile(np.abs(y_resid), 97)
    mask = np.abs(y_resid) <= clip * 2
    X_c, y_c = X_tr[mask], y_resid[mask]
    rank = np.arange(len(X_c), dtype=float)
    sw = 1.0 + 2.0 * (rank / max(1.0, rank.max())) ** 1.5

    # Split for validation
    split = int(len(X_c) * 0.75)
    Xt, Xv = X_c[:split], X_c[split:]
    yt, yv = y_c[:split], y_c[split:]
    swt = sw[:split]

    def objective_lgb(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1500),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lambda', 1.0, 50.0, log=True),
            'min_child_samples': trial.suggest_int('min_child', 10, 60),
        }
        m = LGBMRegressor(**p, objective='huber', random_state=SEED, n_jobs=-1, verbosity=-1)
        m.fit(Xt, yt, sample_weight=swt)
        return mean_absolute_error(yv, m.predict(Xv))

    def objective_xgb(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1500),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('lambda', 1.0, 50.0, log=True),
            'min_child_weight': trial.suggest_int('min_child', 3, 30),
        }
        m = XGBRegressor(**p, random_state=SEED, n_jobs=-1, verbosity=0)
        m.fit(Xt, yt, sample_weight=swt)
        return mean_absolute_error(yv, m.predict(Xv))

    def objective_cat(trial):
        p = {
            'iterations': trial.suggest_int('iterations', 400, 1500),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2', 1.0, 50.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample', 0.5, 0.9),
        }
        m = CatBoostRegressor(**p, loss_function='MAE', random_seed=SEED, verbose=0)
        m.fit(Xt, yt, sample_weight=swt)
        return mean_absolute_error(yv, m.predict(Xv))

    print("  Tuning LGB...", end=' ', flush=True)
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(objective_lgb, n_trials=n_trials, show_progress_bar=False)
    bp_lgb = study_lgb.best_params
    print(f"MAE={study_lgb.best_value:,.0f}")

    print("  Tuning XGB...", end=' ', flush=True)
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=False)
    bp_xgb = study_xgb.best_params
    print(f"MAE={study_xgb.best_value:,.0f}")

    print("  Tuning CatBoost...", end=' ', flush=True)
    study_cat = optuna.create_study(direction='minimize')
    study_cat.optimize(objective_cat, n_trials=n_trials, show_progress_bar=False)
    bp_cat = study_cat.best_params
    print(f"MAE={study_cat.best_value:,.0f}")

    # Refit on full data with best params
    lgb = LGBMRegressor(
        n_estimators=bp_lgb['n_estimators'], learning_rate=bp_lgb['lr'],
        num_leaves=bp_lgb['num_leaves'], max_depth=bp_lgb['max_depth'],
        subsample=bp_lgb['subsample'], colsample_bytree=bp_lgb['colsample'],
        reg_alpha=bp_lgb['alpha'], reg_lambda=bp_lgb['lambda'],
        min_child_samples=bp_lgb['min_child'],
        objective='huber', random_state=SEED, n_jobs=-1, verbosity=-1)
    xgb = XGBRegressor(
        n_estimators=bp_xgb['n_estimators'], learning_rate=bp_xgb['lr'],
        max_depth=bp_xgb['max_depth'], subsample=bp_xgb['subsample'],
        colsample_bytree=bp_xgb['colsample'], reg_alpha=bp_xgb['alpha'],
        reg_lambda=bp_xgb['lambda'], min_child_weight=bp_xgb['min_child'],
        random_state=SEED, n_jobs=-1, verbosity=0)
    cat = CatBoostRegressor(
        iterations=bp_cat['iterations'], learning_rate=bp_cat['lr'],
        depth=bp_cat['depth'], l2_leaf_reg=bp_cat['l2'],
        subsample=bp_cat['subsample'], colsample_bylevel=bp_cat['colsample'],
        loss_function='MAE', random_seed=SEED, verbose=0)

    lgb.fit(X_c, y_c, sample_weight=sw)
    xgb.fit(X_c, y_c, sample_weight=sw)
    cat.fit(X_c, y_c, sample_weight=sw)

    return lgb, xgb, cat, {'lgb': bp_lgb, 'xgb': bp_xgb, 'cat': bp_cat}

# ── Pipeline ──
def run_v9(train_df, col, fc_dates, tet_strength=0.35, n_trials=40):
    fc_dates = list(pd.to_datetime(fc_dates)); n = len(fc_dates)
    p_naive = build_naive(train_df, col, fc_dates)

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
    hist = hist.copy(); hist['naive'] = naive_hist; hist['resid'] = hist[col] - hist['naive']

    X_tr = build_features(hist.Date.tolist(), train_df, col)
    lgb, xgb, cat, best_params = tune_and_fit(X_tr, hist['resid'].values, n_trials=n_trials)

    X_fc = build_features(fc_dates, train_df, col)
    # 3-model ensemble
    corr = 0.40 * lgb.predict(X_fc) + 0.35 * xgb.predict(X_fc) + 0.25 * cat.predict(X_fc)
    damp = np.exp(-np.arange(n, dtype=float) / 300)
    p_corrected = np.clip(p_naive + corr * damp, 0, None)

    tet_prof = compute_tet_profile(train_df, col)
    base_med = float(train_df[train_df.Date.dt.month.isin([5,6,7,8])][col].median())
    p_corr_tet = apply_tet(p_corrected, fc_dates, tet_prof, base_med, tet_strength)

    return {
        'naive': np.clip(p_naive, 0, None),
        'corrected': p_corrected,
        'corrected_tet': p_corr_tet,
        'best_params': best_params,
    }

# ══════════════════════════════════════════════════════════════════
# CV + FINAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("V9 OPTUNA - 548-DAY HOLDOUT CV")
print("="*60)

cv = {}
for col in ['Revenue', 'COGS']:
    print(f"\n── {col} ──")
    origin = pd.Timestamp('2021-06-30')
    tr = sales[sales.Date <= origin].copy()
    val = sales[sales.Date > origin].head(548).copy()
    y = val[col].values
    preds = run_v9(tr, col, val.Date.tolist(), n_trials=30)
    for k in ['naive', 'corrected', 'corrected_tet']:
        mae = mean_absolute_error(y, preds[k])
        cv[f'{col}_{k}'] = mae
        print(f"  {k}: MAE={mae:,.0f}")

# Tet strength tuning
print("\n── Tet Strength Tuning ──")
best_tet = {}
tr_t = sales[sales.Date <= pd.Timestamp('2021-06-30')].copy()
val_t = sales[sales.Date > pd.Timestamp('2021-06-30')].head(548).copy()
for col in ['Revenue', 'COGS']:
    preds0 = run_v9(tr_t, col, val_t.Date.tolist(), tet_strength=0.0, n_trials=15)
    tet_prof = compute_tet_profile(tr_t, col)
    bm = float(tr_t[tr_t.Date.dt.month.isin([5,6,7,8])][col].median())
    y = val_t[col].values
    best_s, best_mae = 0.0, 1e18
    for s in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
        p = apply_tet(preds0['corrected'], val_t.Date.tolist(), tet_prof, bm, s)
        mae = mean_absolute_error(y, p)
        if mae < best_mae: best_mae, best_s = mae, s
    best_tet[col] = best_s
    print(f"  {col}: best_tet={best_s} (MAE={best_mae:,.0f})")

# Final forecast
print("\n" + "="*60 + "\nFINAL FORECAST\n" + "="*60)
final = {}
for col in ['Revenue', 'COGS']:
    print(f"  {col}...", end=' ', flush=True)
    t0 = time.time()
    final[col] = run_v9(sales, col, forecast_dates, tet_strength=best_tet.get(col, 0.35), n_trials=40)
    print(f"done ({time.time()-t0:.0f}s)")

for vname in ['naive', 'corrected', 'corrected_tet']:
    sub = sub_tpl[['Date']].copy()
    for col in ['Revenue', 'COGS']:
        sub[col] = np.clip(final[col][vname], 0, None)
    fname = f'submission_v9_{vname}.csv'
    out = sub.copy(); out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')
    out.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"  {fname}: Rev mean={sub.Revenue.mean():,.0f}, COGS mean={sub.COGS.mean():,.0f}")

# Save diagnostics
diag = {'version': 'v9_optuna', 'cv': cv, 'best_tet': best_tet}
for col in ['Revenue', 'COGS']:
    if 'best_params' in final[col]:
        diag[f'{col}_params'] = {k: str(v) for k, v in final[col]['best_params'].items()}
with open(os.path.join(OUT_DIR, 'v9_diagnostics.json'), 'w') as f:
    json.dump(diag, f, indent=2, default=str)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for k, v in cv.items(): print(f"  {k}: {v:,.0f}")
print(f"\n🏁 Upload submission_v9_corrected_tet.csv to Kaggle!")
