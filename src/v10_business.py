"""
V10: Business Logic & Feature Engineering
Focuses on Target Engineering (Log Transformation), VN Holidays, Paydays, 
Cross-table features (Promotions, Traffic), and Margin Post-processing.
"""
import os, json, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

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
sales = sales.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
forecast_dates = sub_tpl['Date'].tolist(); N_FC = len(forecast_dates)

# ── Load Aux Data ──
# 1. Web Traffic
traffic = pd.read_csv(os.path.join(DATA_DIR, 'web_traffic.csv'))
traffic['Date'] = pd.to_datetime(traffic['date'])
traffic = traffic.groupby('Date').agg({'sessions': 'sum', 'unique_visitors': 'sum', 'page_views': 'sum'}).reset_index()

# 2. Promotions
try:
    promos = pd.read_csv(os.path.join(DATA_DIR, 'promotions.csv'))
    promos['start_date'] = pd.to_datetime(promos['start_date'])
    promos['end_date'] = pd.to_datetime(promos['end_date'])
except FileNotFoundError:
    promos = pd.DataFrame()

# ── Feature Engineering ──
TET_DATES = pd.to_datetime(['2012-01-23','2013-02-10','2014-01-31','2015-02-19','2016-02-08',
    '2017-01-28','2018-02-16','2019-02-05','2020-01-25','2021-02-12','2022-02-01',
    '2023-01-22','2024-02-10'])

VN_HOLIDAYS_MD = ['01-01', '04-30', '05-01', '09-02', '10-20', '11-20', '12-24', '12-31']

def build_features(df_dates, history_df):
    """Build rich business features for given dates."""
    dates = pd.to_datetime(df_dates)
    f = pd.DataFrame({'Date': dates})
    
    # Time basics
    f['month'] = f.Date.dt.month
    f['dow'] = f.Date.dt.dayofweek
    f['doy'] = f.Date.dt.dayofyear
    f['year'] = f.Date.dt.year
    f['is_weekend'] = (f['dow'] >= 5).astype(int)
    
    # Payday: 1-3 and 15-17
    f['day'] = f.Date.dt.day
    f['is_payday'] = f['day'].isin([1,2,3, 15,16,17]).astype(int)
    
    # Fixed VN Holidays
    md = f.Date.dt.strftime('%m-%d')
    f['is_vn_holiday'] = md.isin(VN_HOLIDAYS_MD).astype(int)
    
    # Tet proximity (Lunar New Year)
    ev = np.sort(TET_DATES.to_numpy().astype('datetime64[ns]'))
    d = np.array(dates, dtype='datetime64[ns]')
    d2tet = np.full(len(d), 365, dtype=int)
    idx = np.searchsorted(ev, d, side='left')
    for i in range(len(d)):
        if idx[i] < len(ev): d2tet[i] = int((ev[idx[i]] - d[i]) / np.timedelta64(1,'D'))
    f['d2tet'] = np.clip(d2tet, 0, 365)
    f['is_tet_season'] = (f['d2tet'] <= 21).astype(int) # 3 weeks before tet
    
    # Promotions
    f['promo_discount'] = 0.0
    f['promo_count'] = 0
    if not promos.empty:
        discount_vals = []
        count_vals = []
        for dt in dates:
            active = promos[(promos['start_date'] <= dt) & (promos['end_date'] >= dt)]
            if len(active) > 0:
                discount_vals.append(active['discount_value'].max()) # Max discount active
                count_vals.append(len(active))
            else:
                discount_vals.append(0.0)
                count_vals.append(0)
        f['promo_discount'] = discount_vals
        f['promo_count'] = count_vals

    # Web Traffic (merge and ffill)
    f = pd.merge(f, traffic, on='Date', how='left')
    for col in ['sessions', 'unique_visitors', 'page_views']:
        if col in f.columns:
            f[col] = f[col].ffill().fillna(0) # Forward fill

    # Historical Lags (Target encoded based on LOG revenue/cogs)
    for tgt in ['Revenue', 'COGS']:
        if tgt not in history_df.columns: continue
        # Use log transformed history
        log_hist = history_df.set_index('Date')[tgt].apply(np.log1p)
        
        # YoY lags
        for lag_yr in [1, 2]:
            vals = []
            for dt in dates:
                c = dt - pd.Timedelta(days=364*lag_yr)
                vals.append(log_hist.get(c, np.nan))
            f[f'{tgt}_log_lag_{lag_yr}yr'] = vals
            
        # Rolling means from end of history
        tail = log_hist.tail(28)
        f[f'{tgt}_log_tail7_mean'] = tail.tail(7).mean()
        f[f'{tgt}_log_tail28_mean'] = tail.mean()
            
    f = f.drop(columns=['Date', 'day']).fillna(-1)
    return f

# ── Pipeline ──
def fit_predict_log(X_tr, y_tr, X_fc):
    """Train ensemble on LOG-transformed target."""
    y_log = np.log1p(y_tr)
    
    # Fast, robust defaults
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, subsample=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8, random_state=SEED, n_jobs=-1, verbosity=0)
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, random_seed=SEED, verbose=0)
    
    lgb.fit(X_tr, y_log)
    xgb.fit(X_tr, y_log)
    cat.fit(X_tr, y_log)
    
    pred_log = 0.4 * lgb.predict(X_fc) + 0.4 * xgb.predict(X_fc) + 0.2 * cat.predict(X_fc)
    return np.expm1(pred_log) # Inverse log transform

def run_v10_cv(sales_df):
    """548-day holdout CV."""
    origin = pd.Timestamp('2021-06-30')
    tr = sales_df[sales_df.Date <= origin].copy()
    val = sales_df[sales_df.Date > origin].head(548).copy()
    
    X_tr = build_features(tr['Date'], tr)
    X_val = build_features(val['Date'], tr) # Note: history features from `tr`
    
    preds = {}
    for col in ['Revenue', 'COGS']:
        preds[col] = fit_predict_log(X_tr, tr[col].values, X_val)
    
    # Post-processing: COGS < Revenue margin check
    for i in range(len(preds['Revenue'])):
        if preds['COGS'][i] >= preds['Revenue'][i]:
            preds['COGS'][i] = preds['Revenue'][i] * 0.90 # Enforce 10% gross margin minimum
            
        preds['Revenue'][i] = max(0, preds['Revenue'][i])
        preds['COGS'][i] = max(0, preds['COGS'][i])
        
    mae_rev = mean_absolute_error(val['Revenue'].values, preds['Revenue'])
    mae_cogs = mean_absolute_error(val['COGS'].values, preds['COGS'])
    
    print(f"CV MAE Revenue: {mae_rev:,.0f}")
    print(f"CV MAE COGS:    {mae_cogs:,.0f}")
    return mae_rev, mae_cogs

def run_v10_full(sales_df, fc_dates):
    """Train on full data and forecast."""
    X_tr = build_features(sales_df['Date'], sales_df)
    X_fc = build_features(fc_dates, sales_df)
    
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
    print("V10: BUSINESS LOGIC & FEATURE ENGINEERING")
    print("="*60)
    
    print("\n── 548-DAY HOLDOUT CV ──")
    t0 = time.time()
    cv_rev, cv_cogs = run_v10_cv(sales)
    print(f"CV Done in {time.time() - t0:.0f}s")
    
    print("\n── FULL FORECAST ──")
    t0 = time.time()
    final_preds = run_v10_full(sales, forecast_dates)
    print(f"Forecast Done in {time.time() - t0:.0f}s")
    
    # Save submission
    sub = sub_tpl[['Date']].copy()
    sub['Revenue'] = final_preds['Revenue']
    sub['COGS'] = final_preds['COGS']
    sub['Date'] = pd.to_datetime(sub['Date']).dt.strftime('%Y-%m-%d')
    
    out_path = os.path.join(OUT_DIR, 'submission_v10_business.csv')
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
