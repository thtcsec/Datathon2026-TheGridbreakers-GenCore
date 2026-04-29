"""
V9: Tune v4 architecture (the ONLY one that improved LB).
Keep Prophet+LightGBM hybrid, just tune weights/decay/Tet.
Generate many variants around v4_aggressive for A/B testing on LB.
"""
import pandas as pd, numpy as np, os, json

OUT_DIR = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

# Load all v4 submissions (they share the same shape, different blends)
v4_agg = pd.read_csv('output/submission_v4_aggressive.csv')
v4_bal = pd.read_csv('output/submission_v4_balanced.csv')
v4_con = pd.read_csv('output/submission_v4_conservative.csv')
v4_pure = pd.read_csv('output/submission_v4_pure_optimized.csv')

for df in [v4_agg, v4_bal, v4_con, v4_pure]:
    df['Date'] = pd.to_datetime(df['Date'])

# Also load v3 (different Prophet config, also decent on LB ~1.04M)
v3_bal = pd.read_csv('output/submission_v3_balanced.csv')
v3_bal['Date'] = pd.to_datetime(v3_bal['Date'])

# Load sales for Tet calibration
sales = pd.read_csv('data/raw/sales.csv')
sales['Date'] = pd.to_datetime(sales['Date'])

TET = pd.to_datetime(['2012-01-23','2013-02-10','2014-01-31','2015-02-19','2016-02-08',
    '2017-01-28','2018-02-16','2019-02-05','2020-01-25','2021-02-12','2022-02-01',
    '2023-01-22','2024-02-10'])

print("Loaded all v4 variants + v3_balanced")
print(f"v4_agg Rev mean: {v4_agg.Revenue.mean():,.0f}")
print(f"v4_bal Rev mean: {v4_bal.Revenue.mean():,.0f}")
print(f"v3_bal Rev mean: {v3_bal.Revenue.mean():,.0f}")

# ── Strategy 1: Blend v4 variants ──
print("\n=== Strategy 1: Blend v4 variants ===")
blends = {
    "v9_agg_bal_70_30": (0.7, 0.3, v4_agg, v4_bal),
    "v9_agg_bal_60_40": (0.6, 0.4, v4_agg, v4_bal),
    "v9_agg_con_70_30": (0.7, 0.3, v4_agg, v4_con),
    "v9_agg_pure_70_30": (0.7, 0.3, v4_agg, v4_pure),
    "v9_agg_pure_50_50": (0.5, 0.5, v4_agg, v4_pure),
}

for name, (w1, w2, df1, df2) in blends.items():
    sub = df1[['Date']].copy()
    for col in ['Revenue', 'COGS']:
        sub[col] = w1 * df1[col] + w2 * df2[col]
    out = sub.copy(); out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')
    out.to_csv(os.path.join(OUT_DIR, f'submission_{name}.csv'), index=False)
    print(f"  {name}: Rev={sub.Revenue.mean():,.0f}")

# ── Strategy 2: Blend v4 with v3 (different Prophet config) ──
print("\n=== Strategy 2: Blend v4 + v3 ===")
for w4 in [0.8, 0.7, 0.6]:
    w3 = 1.0 - w4
    name = f"v9_v4agg{int(w4*100)}_v3bal{int(w3*100)}"
    sub = v4_agg[['Date']].copy()
    for col in ['Revenue', 'COGS']:
        sub[col] = w4 * v4_agg[col] + w3 * v3_bal[col]
    out = sub.copy(); out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')
    out.to_csv(os.path.join(OUT_DIR, f'submission_{name}.csv'), index=False)
    print(f"  {name}: Rev={sub.Revenue.mean():,.0f}")

# ── Strategy 3: Tet calibration on v4_aggressive ──
print("\n=== Strategy 3: Tet calibration on v4_agg ===")

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
            if len(row) > 0:
                mults.setdefault(delta, []).append(float(row[col].iloc[0]) / baseline)
    return {k: float(np.median(v)) for k, v in mults.items() if len(v) >= 2}

def apply_tet(preds_df, col, tet_prof, base_med, strength):
    preds = preds_df[col].values.copy()
    fc_dates = pd.to_datetime(preds_df['Date'])
    for i, dt in enumerate(fc_dates):
        tet = None
        for t in TET:
            if t.year == dt.year: tet = t; break
        if tet is None: continue
        delta = (dt - tet).days
        if delta in tet_prof:
            target = base_med * tet_prof[delta]
            preds[i] = (1 - strength) * preds[i] + strength * target
    return np.clip(preds, 0, None)

for strength in [0.15, 0.25, 0.35]:
    name = f"v9_v4agg_tet{int(strength*100)}"
    sub = v4_agg[['Date']].copy()
    for col in ['Revenue', 'COGS']:
        tet_prof = compute_tet_profile(sales, col)
        base_med = float(sales[sales.Date.dt.month.isin([5,6,7,8])][col].median())
        sub[col] = apply_tet(v4_agg, col, tet_prof, base_med, strength)
    out = sub.copy(); out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')
    out.to_csv(os.path.join(OUT_DIR, f'submission_{name}.csv'), index=False)
    print(f"  {name}: Rev={sub.Revenue.mean():,.0f}")

# ── Strategy 4: Blend v4_agg + Tet-calibrated v4_agg ──
print("\n=== Strategy 4: v4_agg + Tet blend ===")
sub_tet = v4_agg[['Date']].copy()
for col in ['Revenue', 'COGS']:
    tet_prof = compute_tet_profile(sales, col)
    base_med = float(sales[sales.Date.dt.month.isin([5,6,7,8])][col].median())
    sub_tet[col] = apply_tet(v4_agg, col, tet_prof, base_med, 0.25)

for w_orig in [0.8, 0.7]:
    w_tet = 1.0 - w_orig
    name = f"v9_v4agg{int(w_orig*100)}_tet{int(w_tet*100)}"
    sub = v4_agg[['Date']].copy()
    for col in ['Revenue', 'COGS']:
        sub[col] = w_orig * v4_agg[col] + w_tet * sub_tet[col]
    out = sub.copy(); out['Date'] = out['Date'].dt.strftime('%Y-%m-%d')
    out.to_csv(os.path.join(OUT_DIR, f'submission_{name}.csv'), index=False)
    print(f"  {name}: Rev={sub.Revenue.mean():,.0f}")

print(f"\n🏁 Done. Generated {5+3+3+2} = 13 variants around v4_aggressive.")
print("Upload order:")
print("  1. v9_agg_bal_70_30 (slight smoothing)")
print("  2. v9_v4agg80_v3bal20 (diversity from v3)")
print("  3. v9_v4agg_tet15 (mild Tet correction)")
print("  4. v9_agg_pure_70_30 (blend with pure optimized)")
