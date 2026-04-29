"""
V11: Pure Time-Series (Zero Data Leakage)
Strictly adheres to competition rules by:
1. Using ONLY sales.csv (no external data, no auxiliary data that is missing in 2023)
2. Avoiding hardcoded holidays (to be 100% safe from "external data" rule)
3. Using data from 2018-2022 to capture recent trends and avoid ancient noisy data.
4. Using Fourier terms to capture pure mathematical seasonality.
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
SEED = 42; np.random.seed(SEED)

DATA_DIR = 'data/raw'
for c in ['data/raw', '../data/raw']:
    if os.path.isfile(os.path.join(c, 'sales.csv')): DATA_DIR = c; break
OUT_DIR = 'output'; os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Core Data ──
sales = pd.read_csv(os.path.join(DATA_DIR, 'sales.csv'))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
sales['Date'] = pd.to_datetime(sales['Date'])
sub_tpl['Date'] = pd.to_datetime(sub_tpl['Date'])

# Truncate to 2018-2022 (recent 5 years)
sales = sales[sales['Date'] >= '2018-01-01'].sort_values('Date').reset_index(drop=True)
forecast_dates = sub_tpl['Date'].tolist()

def create_fourier_features(dates, K=3):
    """Create Fourier terms for daily data to model yearly seasonality."""
    day_of_year = dates.dt.dayofyear.values
    features = {}
    for k in range(1, K + 1):
        features[f'sin_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        features[f'cos_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)
    return pd.DataFrame(features)

def build_features(df_dates):
    """Build pure time-series features. No external data."""
    dates = pd.Series(pd.to_datetime(df_dates)).reset_index(drop=True)
    f = pd.DataFrame({'Date': dates})
    
    f['month'] = f.Date.dt.month
    f['dow'] = f.Date.dt.dayofweek
    f['doy'] = f.Date.dt.dayofyear
    f['is_weekend'] = (f['dow'] >= 5).astype(int)
    f['quarter'] = f.Date.dt.quarter
    f['week'] = f.Date.dt.isocalendar().week.astype(int)
    
    # We explicitly EXCLUDE 'year' because tree models cannot extrapolate
    # to 2023/2024. They will just predict the average of 2018-2022.
    
    # Add Fourier terms
    fourier_df = create_fourier_features(f.Date, K=4) # 4 harmonics
    f = pd.concat([f, fourier_df], axis=1)
    
    return f.drop(columns=['Date'])

# ── Pipeline ──
def fit_predict_log(X_tr, y_tr, X_fc):
    """Train ensemble on LOG-transformed target."""
    y_log = np.log1p(y_tr)
    
    # Simple, robust models that won't overfit
    lgb = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, 
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)
    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, 
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)
    
    lgb.fit(X_tr, y_log)
    xgb.fit(X_tr, y_log)
    
    # 50/50 blend
    pred_log = 0.5 * lgb.predict(X_fc) + 0.5 * xgb.predict(X_fc)
    return np.expm1(pred_log)

def run_v11_cv(sales_df):
    """Holdout CV: use 2022 as validation, train on 2018-2021."""
    # Since we truncated data to start at 2018, train = 4 years, val = 1 year
    tr = sales_df[sales_df.Date < '2022-01-01'].copy()
    val = sales_df[sales_df.Date >= '2022-01-01'].copy()
    
    X_tr = build_features(tr['Date'])
    X_val = build_features(val['Date'])
    
    preds = {}
    for col in ['Revenue', 'COGS']:
        preds[col] = fit_predict_log(X_tr, tr[col].values, X_val)
    
    # Post-processing Margin Check
    for i in range(len(preds['Revenue'])):
        if preds['COGS'][i] >= preds['Revenue'][i]:
            preds['COGS'][i] = preds['Revenue'][i] * 0.90
        preds['Revenue'][i] = max(0, preds['Revenue'][i])
        preds['COGS'][i] = max(0, preds['COGS'][i])
        
    mae_rev = mean_absolute_error(val['Revenue'].values, preds['Revenue'])
    mae_cogs = mean_absolute_error(val['COGS'].values, preds['COGS'])
    
    print(f"CV (2022) MAE Revenue: {mae_rev:,.0f}")
    print(f"CV (2022) MAE COGS:    {mae_cogs:,.0f}")
    return mae_rev, mae_cogs

def run_v11_full(sales_df, fc_dates):
    """Train on 2018-2022 and forecast 2023-2024."""
    X_tr = build_features(sales_df['Date'])
    X_fc = build_features(fc_dates)
    
    preds = {}
    for col in ['Revenue', 'COGS']:
        print(f"  Training {col}...")
        preds[col] = fit_predict_log(X_tr, sales_df[col].values, X_fc)
        
    # Post-processing
    for i in range(len(preds['Revenue'])):
        if preds['COGS'][i] >= preds['Revenue'][i]:
            preds['COGS'][i] = preds['Revenue'][i] * 0.90
        preds['Revenue'][i] = max(0, preds['Revenue'][i])
        preds['COGS'][i] = max(0, preds['COGS'][i])
        
    return preds

# ── Execution ──
if __name__ == '__main__':
    print("="*60)
    print("V11: PURE TIME-SERIES (ZERO LEAKAGE)")
    print("="*60)
    
    print("\n── 2022 HOLDOUT CV ──")
    t0 = time.time()
    cv_rev, cv_cogs = run_v11_cv(sales)
    print(f"CV Done in {time.time() - t0:.0f}s")
    
    print("\n── FULL FORECAST (2023-2024) ──")
    t0 = time.time()
    final_preds = run_v11_full(sales, forecast_dates)
    print(f"Forecast Done in {time.time() - t0:.0f}s")
    
    # Save submission
    sub = sub_tpl[['Date']].copy()
    sub['Revenue'] = final_preds['Revenue']
    sub['COGS'] = final_preds['COGS']
    sub['Date'] = pd.to_datetime(sub['Date']).dt.strftime('%Y-%m-%d')
    
    out_path = os.path.join(OUT_DIR, 'submission_v11_pure_ts.csv')
    sub.to_csv(out_path, index=False)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Revenue CV MAE: {cv_rev:,.0f}")
    print(f"  COGS CV MAE:    {cv_cogs:,.0f}")
    print(f"  Revenue Mean:   {sub.Revenue.mean():,.0f}")
    print(f"  COGS Mean:      {sub.COGS.mean():,.0f}")
    margin_violations = (sub['COGS'] >= sub['Revenue']).sum()
    print(f"  Margin Violations (Fixed): {margin_violations}")
    print(f"\n🏁 Upload {out_path} to Kaggle!")
