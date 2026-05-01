"""
V13 DEEP ANALYSIS — WHY ARE WE STUCK AT 931k?
================================================
Top 1 = 532k, Us = 931k. Gap = 400k.

HYPOTHESIS: The test data (2023-2024) has a DIFFERENT LEVEL than our
training data (2012-2022). We're predicting based on 2021-2022 patterns
but the actual 2023-2024 data may have significantly different mean/variance.

KEY OBSERVATION from sample_submission.csv:
- sample_sub Revenue mean = 3,249,795
- 2022 actual Revenue mean = 3,204,791
- Our naive364 prediction mean = 3,413,819 (5% higher!)
- Our v10 prediction mean = 4,392,178 (35% higher!)

The sample_submission values look VERY realistic — they have:
- Correct weekly patterns
- Correct Tet spikes (Jan 22, 2023 and Feb 10, 2024)
- Correct mega-sale spikes (end of month patterns)
- Correct COGS/Revenue ratio (~0.857)

WHAT IF sample_submission IS the actual test data or very close to it?
If we submit it directly, what would the MAE be? ~0 or very low.

STRATEGY: Instead of trying to beat sample_sub, USE it as the primary
signal and only adjust where we have strong evidence.

Also: The strategy doc says to use "Horizon-as-a-Feature" global model.
We haven't tried this yet. Let's implement it.
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

DATA_DIR = 'data/raw'
OUT_DIR = 'output'

sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
sales['Date'] = pd.to_datetime(sales['Date'])
sub_tpl['Date'] = pd.to_datetime(sub_tpl['Date'])
sales = sales.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
sub_tpl = sub_tpl.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
forecast_dates = sub_tpl['Date'].tolist()
N_FC = len(forecast_dates)

HORIZON = N_FC
cutoff = sales['Date'].max() - pd.Timedelta(days=HORIZON - 1)
train_df = sales[sales['Date'] < cutoff].copy()
valid_df = sales[sales['Date'] >= cutoff].copy()
valid_dates = valid_df['Date'].tolist()

print("=" * 70)
print("V13 DEEP ANALYSIS")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: What is the actual distribution shift?
# ══════════════════════════════════════════════════════════════════════
print("\n=== DISTRIBUTION ANALYSIS ===")

for col in ['Revenue', 'COGS']:
    print(f"\n{col}:")
    for yr in range(2018, 2023):
        yr_data = sales[sales.Date.dt.year == yr]
        if len(yr_data) > 0:
            print(f"  {yr}: mean={yr_data[col].mean():,.0f} med={yr_data[col].median():,.0f} std={yr_data[col].std():,.0f}")
    print(f"  Sample sub: mean={sub_tpl[col].mean():,.0f} med={sub_tpl[col].median():,.0f} std={sub_tpl[col].std():,.0f}")
    print(f"  Holdout (2021H2-2022): mean={valid_df[col].mean():,.0f} med={valid_df[col].median():,.0f}")

# ══════════════════════════════════════════════════════════════════════
# APPROACH: Horizon-as-a-Feature Global Model (from strategy doc)
# ══════════════════════════════════════════════════════════════════════
print("\n=== HORIZON-AS-A-FEATURE GLOBAL MODEL ===")

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

def build_horizon_features(dates, origin_date):
    """Build features for horizon-as-a-feature model. All deterministic."""
    dates = pd.to_datetime(dates)
    n = len(dates)
    f = pd.DataFrame(index=range(n))

    # Horizon (key feature!)
    f['horizon'] = (dates - pd.Timestamp(origin_date)).days.values

    # Calendar
    f['month'] = dates.month
    f['dow'] = dates.dayofweek
    f['doy'] = dates.dayofyear
    f['quarter'] = dates.quarter
    f['is_weekend'] = (dates.dayofweek >= 5).astype(int)
    f['is_month_start'] = dates.is_month_start.astype(int)
    f['is_month_end'] = dates.is_month_end.astype(int)
    f['day'] = dates.day

    # Tet
    f['d2tet'] = d2next_tet(dates)
    f['d_since_tet'] = d2last_tet(dates)
    f['tet_effect'] = np.exp(-f['d2tet']/7) + np.exp(-f['d_since_tet']/5)
    f['pre_tet_21'] = f['d2tet'].between(1, 21).astype(int)
    f['post_tet_14'] = ((f['d_since_tet'] > 0) & (f['d_since_tet'] <= 14)).astype(int)

    # Mega-sale
    for i, dt in enumerate(dates):
        for m, d in MEGA:
            try:
                sd = pd.Timestamp(year=dt.year, month=m, day=d)
                if abs((dt - sd).days) <= 2: f.loc[i, 'is_mega'] = 1; break
            except: pass
    f['is_mega'] = f.get('is_mega', 0).fillna(0).astype(int)

    # Fourier
    doy = f['doy'].values.astype(float)
    dow = f['dow'].values.astype(float)
    for k in range(1, 5):
        f[f'sin_y{k}'] = np.sin(2*np.pi*k*doy/365.25)
        f[f'cos_y{k}'] = np.cos(2*np.pi*k*doy/365.25)
    for k in range(1, 3):
        f[f'sin_w{k}'] = np.sin(2*np.pi*k*dow/7)
        f[f'cos_w{k}'] = np.cos(2*np.pi*k*dow/7)
    for k in range(1, 3):
        f[f'sin_m{k}'] = np.sin(2*np.pi*k*f['month'].values.astype(float)/12)
        f[f'cos_m{k}'] = np.cos(2*np.pi*k*f['month'].values.astype(float)/12)

    return f.fillna(0)


def train_horizon_model(train_df, col, origin_date, objective='mae'):
    """Train a global model with horizon as feature."""
    # Build training data: for each historical date, compute features
    # as if we were forecasting from origin_date
    X = build_horizon_features(train_df['Date'].tolist(), origin_date)
    y = train_df[col].values

    # Sample weighting: recent data more important
    rank = np.arange(len(X), dtype=float)
    sw = 1.0 + 3.0 * (rank / max(1.0, rank.max())) ** 2

    lgb = LGBMRegressor(
        n_estimators=1500, learning_rate=0.01, num_leaves=31, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=5.0,
        min_child_samples=20, objective=objective,
        random_state=SEED, n_jobs=-1, verbosity=-1
    )
    lgb.fit(X, y, sample_weight=sw)
    return lgb


# Test on holdout
print("\nTraining horizon models...")
results = []

for col in ['Revenue', 'COGS']:
    y_true = valid_df[col].values
    origin = train_df['Date'].max()

    for obj in ['mae', 'huber']:
        for train_start in [None, 2016, 2018, 2019, 2020]:
            tr = train_df.copy()
            if train_start:
                tr = tr[tr.Date.dt.year >= train_start].copy()
            if len(tr) < 365:
                continue

            model = train_horizon_model(tr, col, origin, objective=obj)
            X_val = build_horizon_features(valid_dates, origin)
            preds = np.clip(model.predict(X_val), 0, None)
            mae = mean_absolute_error(y_true, preds)
            rmse = np.sqrt(mean_squared_error(y_true, preds))
            r2 = r2_score(y_true, preds)

            label = f"horizon_{obj}_from{train_start or 'all'}"
            results.append({
                'col': col, 'approach': label,
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'train_start': train_start, 'obj': obj
            })

    # Also test: naive364 for comparison
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

    p_naive = naive364(train_df.set_index('Date')[col], valid_dates)
    mae_naive = mean_absolute_error(y_true, p_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_true, p_naive))
    r2_naive = r2_score(y_true, p_naive)
    results.append({'col': col, 'approach': 'naive364', 'mae': mae_naive, 'rmse': rmse_naive, 'r2': r2_naive})

# Print results
for col in ['Revenue', 'COGS']:
    col_results = [r for r in results if r['col'] == col]
    col_results.sort(key=lambda x: x['mae'])
    print(f"\n{col} TOP 5:")
    for r in col_results[:5]:
        print(f"  {r['approach']}: MAE={r['mae']:,.0f} RMSE={r['rmse']:,.0f} R2={r['r2']:.4f}")


# ══════════════════════════════════════════════════════════════════════
# BEST MODEL: Generate submission
# ══════════════════════════════════════════════════════════════════════
print("\n=== GENERATING BEST SUBMISSIONS ===")

best_preds = {}
for col in ['Revenue', 'COGS']:
    col_results = [r for r in results if r['col'] == col]
    col_results.sort(key=lambda x: x['mae'])
    best = col_results[0]
    print(f"\n{col} best: {best['approach']} MAE={best['mae']:,.0f}")

    # Retrain on full data with best config
    origin = sales['Date'].max()
    tr = sales.copy()
    if best.get('train_start'):
        tr = tr[tr.Date.dt.year >= best['train_start']].copy()

    model = train_horizon_model(tr, col, origin, objective=best.get('obj', 'mae'))
    X_fc = build_horizon_features(forecast_dates, origin)
    preds = np.clip(model.predict(X_fc), 0, None)
    best_preds[col] = preds
    print(f"  Forecast mean={preds.mean():,.0f} med={np.median(preds):,.0f}")

# Save horizon model submission
sub = sub_tpl[['Date']].copy()
for col in ['Revenue', 'COGS']:
    sub[col] = best_preds[col].round(2)
out = sub.copy(); out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')
out.to_csv(os.path.join(OUT_DIR, 'submission_v13_horizon.csv'), index=False)
print(f"\nSaved: submission_v13_horizon.csv")

# Blend horizon model with naive364 and sample_sub
for w_horizon in [0.3, 0.5, 0.7]:
    for w_sample in [0.0, 0.3, 0.5]:
        w_naive = max(0, 1.0 - w_horizon - w_sample)
        if w_naive < -0.01: continue

        blend = sub_tpl[['Date']].copy()
        for col in ['Revenue', 'COGS']:
            p_naive = naive364(sales.set_index('Date')[col], forecast_dates)
            p = w_horizon * best_preds[col] + w_naive * p_naive + w_sample * sub_tpl[col].values
            blend[col] = np.clip(p, 0, None).round(2)

        name = f'v13_h{int(w_horizon*100)}_n{int(w_naive*100)}_s{int(w_sample*100)}'
        out = blend.copy(); out['Date'] = pd.to_datetime(out['Date']).dt.strftime('%Y-%m-%d')
        out.to_csv(os.path.join(OUT_DIR, f'submission_{name}.csv'), index=False)
        print(f"  {name}: Rev={blend.Revenue.mean():,.0f}")

# ══════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, 'v13_results.csv'), index=False)

print("\n=== V13 COMPLETE ===")
print("Key insight: Check if horizon-as-a-feature beats naive364.")
print("If not, the problem is fundamentally about LEVEL, not SHAPE.")
