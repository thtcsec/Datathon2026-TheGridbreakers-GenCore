"""
V12: Hybrid Trend-Seasonality Pipeline
1. Truncate data to 2020-2022 to avoid the 2018-2019 distribution shift.
2. Use LinearRegression on Day Index to capture global Trend.
3. Use LightGBM & XGBoost on Fourier Terms to capture Seasonality (Residuals).
4. No data leakage (No external datasets used).
"""
import os, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
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

# Truncate to 2020-2022 (New Normal)
sales = sales[sales['Date'] >= '2020-01-01'].sort_values('Date').reset_index(drop=True)
forecast_dates = sub_tpl['Date'].tolist()

def create_fourier_features(dates, K=4):
    day_of_year = dates.dt.dayofyear.values
    features = {}
    for k in range(1, K + 1):
        features[f'sin_{k}'] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        features[f'cos_{k}'] = np.cos(2 * np.pi * k * day_of_year / 365.25)
    return pd.DataFrame(features)

def build_features(df_dates, base_date=pd.to_datetime('2020-01-01')):
    """Build pure time-series features and Day Index."""
    dates = pd.Series(pd.to_datetime(df_dates)).reset_index(drop=True)
    f = pd.DataFrame({'Date': dates})
    
    # Trend Feature
    f['day_index'] = (f['Date'] - base_date).dt.days
    
    # Seasonal Features
    f['month'] = f.Date.dt.month
    f['dow'] = f.Date.dt.dayofweek
    f['doy'] = f.Date.dt.dayofyear
    f['is_weekend'] = (f['dow'] >= 5).astype(int)
    f['quarter'] = f.Date.dt.quarter
    f['week'] = f.Date.dt.isocalendar().week.astype(int)
    
    fourier_df = create_fourier_features(f.Date, K=4)
    f = pd.concat([f, fourier_df], axis=1)
    
    return f.drop(columns=['Date'])

# ── Pipeline ──
def fit_predict_hybrid(X_tr, y_tr, X_fc):
    """Hybrid Model: LinearRegression (Trend) + LightGBM/XGBoost (Residuals)"""
    y_log = np.log1p(y_tr)
    
    # 1. Trend Model
    lr = LinearRegression()
    day_idx_tr = X_tr[['day_index']].values
    day_idx_fc = X_fc[['day_index']].values
    
    lr.fit(day_idx_tr, y_log)
    trend_tr = lr.predict(day_idx_tr)
    trend_fc = lr.predict(day_idx_fc)
    
    # 2. Residuals (Detrended target)
    residuals = y_log - trend_tr
    
    # 3. Seasonality Model
    X_season_tr = X_tr.drop(columns=['day_index'])
    X_season_fc = X_fc.drop(columns=['day_index'])
    
    lgb = LGBMRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, 
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, 
                       subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=0)
    
    lgb.fit(X_season_tr, residuals)
    xgb.fit(X_season_tr, residuals)
    
    season_fc = 0.5 * lgb.predict(X_season_fc) + 0.5 * xgb.predict(X_season_fc)
    
    # 4. Combine
    final_pred_log = trend_fc + season_fc
    return np.expm1(final_pred_log)

def run_v12_cv(sales_df):
    """Holdout CV: use 2022 as validation, train on 2020-2021."""
    tr = sales_df[sales_df.Date < '2022-01-01'].copy()
    val = sales_df[sales_df.Date >= '2022-01-01'].copy()
    
    X_tr = build_features(tr['Date'])
    X_val = build_features(val['Date'])
    
    preds = {}
    for col in ['Revenue', 'COGS']:
        preds[col] = fit_predict_hybrid(X_tr, tr[col].values, X_val)
    
    # Post-processing
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

def run_v12_full(sales_df, fc_dates):
    X_tr = build_features(sales_df['Date'])
    X_fc = build_features(fc_dates)
    
    preds = {}
    for col in ['Revenue', 'COGS']:
        print(f"  Training Hybrid Model for {col}...")
        preds[col] = fit_predict_hybrid(X_tr, sales_df[col].values, X_fc)
        
    # Post-processing
    for i in range(len(preds['Revenue'])):
        if preds['COGS'][i] >= preds['Revenue'][i]:
            preds['COGS'][i] = preds['Revenue'][i] * 0.90
        preds['Revenue'][i] = max(0, preds['Revenue'][i])
        preds['COGS'][i] = max(0, preds['COGS'][i])
        
    return preds

if __name__ == '__main__':
    print("="*60)
    print("V12: HYBRID TREND-SEASONALITY")
    print("="*60)
    
    print("\n── 2022 HOLDOUT CV ──")
    t0 = time.time()
    cv_rev, cv_cogs = run_v12_cv(sales)
    print(f"CV Done in {time.time() - t0:.0f}s")
    
    print("\n── FULL FORECAST (2023-2024) ──")
    t0 = time.time()
    final_preds = run_v12_full(sales, forecast_dates)
    print(f"Forecast Done in {time.time() - t0:.0f}s")
    
    sub = sub_tpl[['Date']].copy()
    sub['Revenue'] = final_preds['Revenue']
    sub['COGS'] = final_preds['COGS']
    sub['Date'] = pd.to_datetime(sub['Date']).dt.strftime('%Y-%m-%d')
    
    out_path = os.path.join(OUT_DIR, 'submission_v12_hybrid.csv')
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
