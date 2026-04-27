"""
DATATHON 2026 - GenCore v4 TopKill v2
======================================
Lesson from v4 attempt 1: aggressive horizon decay HURT CV (757k vs v3 636k).
Optimal static weights showed hybrid still needs 55-80% weight.

V4v2 strategy:
- Keep v3 architecture (proven CV=636k)
- ADD: 4-model ensemble (naive, theta, prophet, hybrid) with OPTIMIZED weights
- ADD: Tet empirical calibration
- ADD: Prophet ensemble (median of 4 configs) instead of single Prophet
- ADD: Mild horizon decay (not aggressive) tuned by CV
- REMOVE: Theta from blend (CV showed weight=0 every fold)
- KEY INSIGHT from CV: optimal_static always gave hybrid 55-80% weight
  -> the problem is NOT hybrid degradation, it's something else on LB
"""

import os, glob, json, time, warnings
from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

KAGGLE = os.path.exists("/kaggle/input")
if KAGGLE:
    matches = glob.glob("/kaggle/input/**/sales.csv", recursive=True)
    DATA_DIR = os.path.dirname(matches[0]) if matches else "/kaggle/input"
    OUT_DIR = "/kaggle/working"
else:
    DATA_DIR = "data/raw"
    for c in ["data/raw", "../data/raw"]:
        if os.path.isfile(os.path.join(c, "sales.csv")):
            DATA_DIR = c; break
    OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    from prophet import Prophet
except ImportError:
    os.system("pip install -q prophet"); from prophet import Prophet
try:
    from lightgbm import LGBMRegressor
except ImportError:
    os.system("pip install -q lightgbm"); from lightgbm import LGBMRegressor

import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

print(f"ENV: {'Kaggle' if KAGGLE else 'Local'} | DATA: {DATA_DIR}")


# ── Data ──
sales = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
sub_tpl = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sales["Date"] = pd.to_datetime(sales["Date"]); sub_tpl["Date"] = pd.to_datetime(sub_tpl["Date"])
sales = sales.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
sub_tpl = sub_tpl.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
forecast_dates = sub_tpl["Date"].tolist()
N_FC = len(forecast_dates)
print(f"Train: {sales.Date.min().date()}->{sales.Date.max().date()} ({len(sales)}), Forecast: {N_FC} days")

# ── Aux static profiles ──
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
        daily["month"]=daily[dc].dt.month; daily["dow"]=daily[dc].dt.dayofweek
        aux["orders_month"]=daily.groupby("month")["n"].median().to_dict()
        aux["orders_dow"]=daily.groupby("dow")["n"].median().to_dict()

# ── Calendar ──
TET = pd.to_datetime(["2012-01-23","2013-02-10","2014-01-31","2015-02-19","2016-02-08",
    "2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12","2022-02-01",
    "2023-01-22","2024-02-10"])
MEGA = [(1,1),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(11,11),(12,12)]

def build_holidays(last_date, tl=-21, tu=7):
    rows = []
    for td in TET:
        if td <= last_date + pd.Timedelta(days=60):
            rows.append({"holiday":"tet","ds":td,"lower_window":tl,"upper_window":tu})
    for y in range(2012, last_date.year+1):
        for m,d in MEGA:
            dt=pd.Timestamp(year=y,month=m,day=d)
            if dt<=last_date: rows.append({"holiday":f"sale_{m}_{d}","ds":dt,"lower_window":-3,"upper_window":2})
        for m,d in [(1,1),(4,30),(5,1),(9,2)]:
            dt=pd.Timestamp(year=y,month=m,day=d)
            if dt<=last_date: rows.append({"holiday":"vn_hol","ds":dt,"lower_window":-1,"upper_window":1})
    return pd.DataFrame(rows)

def d2next(dates, events, default=365):
    ev=np.sort(np.array(events,dtype="datetime64[ns]")); d=dates.to_numpy().astype("datetime64[ns]")
    out=np.full(len(d),default,dtype=int); idx=np.searchsorted(ev,d,side="left")
    for i in range(len(d)):
        if idx[i]<len(ev): out[i]=int((ev[idx[i]]-d[i])/np.timedelta64(1,"D"))
    return out

def d2last(dates, events, default=365):
    ev=np.sort(np.array(events,dtype="datetime64[ns]")); d=dates.to_numpy().astype("datetime64[ns]")
    out=np.full(len(d),default,dtype=int); idx=np.searchsorted(ev,d,side="right")-1
    for i in range(len(d)):
        if idx[i]>=0: out[i]=int((d[i]-ev[idx[i]])/np.timedelta64(1,"D"))
    return out

# ── Tet empirical multipliers ──
def tet_multipliers(sales_df, col):
    mults = {}
    for tet in TET:
        yr = sales_df[sales_df.Date.dt.year==tet.year]
        if len(yr)<100: continue
        med = yr[col].median()
        if med<=0: continue
        for delta in range(-30,21):
            d = tet+pd.Timedelta(days=delta)
            row = sales_df[sales_df.Date==d]
            if len(row)>0: mults.setdefault(delta,[]).append(row[col].iloc[0]/med)
    return {k:float(np.median(v)) for k,v in mults.items()}

tet_m_rev = tet_multipliers(sales,"Revenue")
tet_m_cogs = tet_multipliers(sales,"COGS")


# ══════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════

def snaive364(train_s, fc_dates):
    s=train_s.sort_index().astype(float); h={pd.Timestamp(i):float(v) for i,v in s.items()}
    fb=float(s.tail(28).median()); preds=[]
    for dt in pd.to_datetime(fc_dates):
        dt=pd.Timestamp(dt); val=None
        for off in [364,371,357,728]:
            c=dt-pd.Timedelta(days=off)
            if c in h: val=h[c]; break
        if val is None: val=fb
        h[dt]=float(val); preds.append(float(val))
    return np.array(preds)

def swin_median(train_df, col, fc_dates, w=7):
    wk=train_df[["Date",col]].copy(); wk["doy"]=wk.Date.dt.dayofyear; wk["dow"]=wk.Date.dt.dayofweek
    vals=wk[col].values.astype(float); doys=wk["doy"].values; dows=wk["dow"].values; fb=float(np.nanmedian(vals))
    preds=[]
    for dt in pd.to_datetime(fc_dates):
        doy,dow=int(dt.dayofyear),int(dt.dayofweek)
        diff=np.abs(doys-doy); dist=np.minimum(diff,366-diff)
        mask=(dows==dow)&(dist<=w)
        if mask.sum()<3: mask=(dows==dow)&(dist<=w+7)
        if mask.sum()<3: mask=dist<=w
        preds.append(float(np.nanmedian(vals[mask])) if mask.sum()>0 else fb)
    return np.array(preds)

def trend_adj(train_df, col, naive, fc_dates):
    yr=train_df.groupby(train_df.Date.dt.year)[col].sum()
    if len(yr)>=2:
        g=yr.iloc[-1]/max(yr.iloc[-2],1.0); g=np.clip(g,0.85,1.20)
    else: g=1.0
    return naive*g

# ── Prophet ensemble ──
PCFG = [
    {"cps":0.03,"sm":"multiplicative","fo":15},
    {"cps":0.05,"sm":"multiplicative","fo":20},
    {"cps":0.10,"sm":"multiplicative","fo":20},
    {"cps":0.05,"sm":"additive","fo":20},
]

def fit_prophet(train_df, col, holidays, cps, sm, fo):
    df=train_df[["Date",col]].rename(columns={"Date":"ds",col:"y"}).copy()
    cap=float(df.y.quantile(0.995)*1.25); fl=max(0.0,float(df.y.quantile(0.005)*0.75))
    df["cap"]=cap; df["floor"]=fl
    m=Prophet(growth="logistic",holidays=holidays,yearly_seasonality=fo,weekly_seasonality=True,
              daily_seasonality=False,seasonality_mode=sm,changepoint_prior_scale=cps,
              seasonality_prior_scale=10.0,holidays_prior_scale=10.0,changepoint_range=0.9)
    m.add_seasonality(name="monthly",period=30.5,fourier_order=5)
    m.add_seasonality(name="quarterly",period=91.25,fourier_order=3)
    m.fit(df)
    return m, cap, fl

def prophet_pred(model, dates, cap, fl):
    fut=pd.DataFrame({"ds":pd.to_datetime(list(dates))}); fut["cap"]=cap; fut["floor"]=fl
    fc=model.predict(fut)
    out=pd.DataFrame({"Date":fut["ds"]})
    for c in ["yhat","trend","weekly","yearly","holidays"]:
        out[f"prophet_{c}"]=fc[c].values if c in fc.columns else 0.0
    out["prophet_monthly"]=fc["monthly"].values if "monthly" in fc.columns else 0.0
    return out

# ── Features ──
def build_feats(dates, hist_df, col, origin, pr_df, aux_d):
    f=pd.DataFrame({"Date":pd.to_datetime(list(dates))}); d=f["Date"]
    f["month"]=d.dt.month; f["day"]=d.dt.day; f["dayofweek"]=d.dt.dayofweek
    f["dayofyear"]=d.dt.dayofyear; f["weekofyear"]=d.dt.isocalendar().week.astype(int).values
    f["quarter"]=d.dt.quarter; f["is_weekend"]=(f["dayofweek"]>=5).astype(int)
    f["is_month_start"]=d.dt.is_month_start.astype(int); f["is_month_end"]=d.dt.is_month_end.astype(int)
    f["is_payday"]=f["day"].isin([1,15,25]).astype(int)
    hs=hist_df["Date"].min()
    f["days_since_start"]=(d-pd.Timestamp(hs)).dt.days
    f["forecast_horizon"]=np.clip((d-pd.Timestamp(origin)).dt.days,0,600)
    h=hist_df[["Date",col]].copy(); h["t"]=(h.Date-h.Date.min()).dt.days.astype(float)
    lr=LinearRegression().fit(h[["t"]],h[col])
    ft=(d-pd.Timestamp(hs)).dt.days.astype(float).values.reshape(-1,1)
    f["linear_trend"]=lr.predict(ft)
    doy=f["dayofyear"].values.astype(float); dow=f["dayofweek"].values.astype(float)
    for k in range(1,6):
        f[f"sin_y{k}"]=np.sin(2*np.pi*k*doy/365.25); f[f"cos_y{k}"]=np.cos(2*np.pi*k*doy/365.25)
    for k in range(1,4):
        f[f"sin_w{k}"]=np.sin(2*np.pi*k*dow/7.0); f[f"cos_w{k}"]=np.cos(2*np.pi*k*dow/7.0)
    for k in range(1,4):
        f[f"sin_m{k}"]=np.sin(2*np.pi*k*f["month"].values.astype(float)/12.0)
        f[f"cos_m{k}"]=np.cos(2*np.pi*k*f["month"].values.astype(float)/12.0)
    ta=TET.to_numpy()
    f["days_to_tet"]=d2next(d,ta); f["days_since_tet"]=d2last(d,ta)
    f["is_pre_tet_30"]=f["days_to_tet"].between(1,30).astype(int)
    f["is_pre_tet_14"]=f["days_to_tet"].between(1,14).astype(int)
    f["is_pre_tet_7"]=f["days_to_tet"].between(1,7).astype(int)
    f["is_tet_week"]=f["days_to_tet"].between(-7,0).astype(int)
    f["is_post_tet_7"]=f["days_since_tet"].between(1,7).astype(int)
    f["is_post_tet_14"]=f["days_since_tet"].between(1,14).astype(int)
    f["tet_proximity"]=np.exp(-0.1*np.minimum(f["days_to_tet"].abs(),f["days_since_tet"].abs()))
    sl=[]
    for y in range(2012,2025):
        for m,dd in MEGA: sl.append(pd.Timestamp(year=y,month=m,day=dd))
    sa=np.array(sorted(sl),dtype="datetime64[ns]")
    f["days_to_sale"]=d2next(d,sa); f["is_sale_window"]=(f["days_to_sale"].abs()<=3).astype(int)
    f["is_11_11"]=((f["month"]==11)&(f["day"]==11)).astype(int)
    f["is_12_12"]=((f["month"]==12)&(f["day"]==12)).astype(int)
    hi=hist_df[["Date",col]].copy(); hi["doy"]=hi.Date.dt.dayofyear; hi["dow"]=hi.Date.dt.dayofweek; hi["month"]=hi.Date.dt.month
    gm=float(hi[col].median())
    f["hist_doy"]=f["dayofyear"].map(hi.groupby("doy")[col].median()).fillna(gm)
    f["hist_dow"]=f["dayofweek"].map(hi.groupby("dow")[col].median()).fillna(gm)
    f["hist_month"]=f["month"].map(hi.groupby("month")[col].median()).fillna(gm)
    recent=hi[hi.Date.dt.year>=(origin.year-3)]
    if len(recent)>100:
        f["hist_doy_recent"]=f["dayofyear"].map(recent.groupby("doy")[col].median()).fillna(gm)
    else: f["hist_doy_recent"]=f["hist_doy"]
    for key,mapping in aux_d.items():
        if key.endswith("_month"): f[f"aux_{key}"]=f["month"].map(mapping).fillna(0)
        elif key.endswith("_dow"): f[f"aux_{key}"]=f["dayofweek"].map(mapping).fillna(0)
    f=f.merge(pr_df,on="Date",how="left")
    for c in f.columns:
        if c!="Date" and f[c].isna().any(): f[c]=f[c].fillna(0)
    return f.drop(columns=["Date"])

def fit_lgbm(X, y_r, n_est=1200, lr_rate=0.02, nl=31, ra=0.5, rl=5.0):
    rank=np.arange(len(X),dtype=float); sw=1.0+2.0*(rank/max(1.0,rank.max()))**1.5
    m=LGBMRegressor(n_estimators=n_est,learning_rate=lr_rate,num_leaves=nl,max_depth=-1,
        subsample=0.85,colsample_bytree=0.85,min_child_samples=30,reg_alpha=ra,reg_lambda=rl,
        objective="mae",random_state=SEED,n_jobs=-1,verbosity=-1)
    m.fit(X,y_r,sample_weight=sw)
    return m

print("Models ready")


# ══════════════════════════════════════════════════════════════════════
# PIPELINE: Prophet ensemble + LightGBM residual + Naive
# ══════════════════════════════════════════════════════════════════════

def run_full_pipeline(train_df, col, fc_dates, holidays, lgb_cfg=None):
    """Run full pipeline, return dict of predictions."""
    if lgb_cfg is None:
        lgb_cfg = {"n_est":1200,"lr":0.02,"nl":31,"ra":0.5,"rl":5.0}
    origin = train_df.Date.max()
    n = len(fc_dates)

    # 1. Naive
    p364 = snaive364(train_df.set_index("Date")[col], fc_dates)
    pwin = swin_median(train_df, col, fc_dates, w=7)
    p_naive = 0.5*p364 + 0.5*pwin
    p_naive = trend_adj(train_df, col, p_naive, fc_dates)

    # 2. Prophet ensemble (4 configs, take median)
    prophet_yhats = []
    prophet_models = []
    for i, cfg in enumerate(PCFG):
        m, cap, fl = fit_prophet(train_df, col, holidays, cfg["cps"], cfg["sm"], cfg["fo"])
        prophet_models.append((m, cap, fl))
        pr = prophet_pred(m, fc_dates, cap, fl)
        prophet_yhats.append(pr["prophet_yhat"].values)

    p_prophet = np.median(np.array(prophet_yhats), axis=0)

    # 3. LightGBM residual (using first Prophet config for features)
    m0, cap0, fl0 = prophet_models[0]
    pr_train = prophet_pred(m0, train_df.Date.tolist(), cap0, fl0)
    pr_fc = prophet_pred(m0, fc_dates, cap0, fl0)

    resid = train_df[col].values - pr_train["prophet_yhat"].values
    X_tr = build_feats(train_df.Date.tolist(), train_df[["Date",col]], col, origin, pr_train, aux)
    X_fc = build_feats(fc_dates, train_df[["Date",col]], col, origin, pr_fc, aux)

    lgb = fit_lgbm(X_tr, resid, lgb_cfg["n_est"], lgb_cfg["lr"], lgb_cfg["nl"], lgb_cfg["ra"], lgb_cfg["rl"])
    p_hybrid = np.clip(p_prophet + lgb.predict(X_fc), 0.0, None)

    return {"naive": p_naive, "prophet": p_prophet, "hybrid": p_hybrid}


def optimize_weights_3(y_true, p_naive, p_prophet, p_hybrid, step=0.05):
    """Grid search 3-model blend weights."""
    best_w = (0.33, 0.33, 0.34); best_mae = 1e18
    for wn in np.arange(0.0, 0.65, step):
        for wp in np.arange(0.0, 0.85, step):
            wh = round(1.0-wn-wp, 4)
            if wh < -0.01: continue
            wh = max(wh, 0.0)
            p = wn*p_naive + wp*p_prophet + wh*p_hybrid
            mae = mean_absolute_error(y_true, p)
            if mae < best_mae: best_mae=mae; best_w=(float(wn),float(wp),float(wh))
    return best_w, best_mae


def dynamic_blend(p_naive, p_prophet, p_hybrid, weights, decay=0.5):
    """Mild horizon decay (v3 style but tunable)."""
    wn,wp,wh = weights; n=len(p_naive)
    if n < 10 or decay <= 0:
        return np.clip(wn*p_naive + wp*p_prophet + wh*p_hybrid, 0, None)
    t = np.linspace(0, 1, n)
    wh_t = wh*(1.0-decay*t); wp_t = wp*(1.0+decay*0.8*t); wn_t = wn*(1.0+decay*0.3*t)
    denom = wh_t+wp_t+wn_t; denom = np.where(denom<0.01, 1.0, denom)
    return np.clip((wn_t*p_naive + wp_t*p_prophet + wh_t*p_hybrid)/denom, 0, None)


def apply_tet_cal(pred, fc_dates, tet_mults, base_med, strength=0.3):
    """Apply empirical Tet calibration."""
    pred = pred.copy()
    fc_set = {d: i for i, d in enumerate(fc_dates)}
    for tet in TET:
        for delta, mult in tet_mults.items():
            d = tet + pd.Timedelta(days=delta)
            if d in fc_set:
                idx = fc_set[d]
                empirical = base_med * mult
                pred[idx] = (1-strength)*pred[idx] + strength*empirical
    return pred

print("Pipeline ready")


# ══════════════════════════════════════════════════════════════════════
# SEARCH CONFIGS (proven from v3 + new combos)
# ══════════════════════════════════════════════════════════════════════

CONFIGS = [
    # v3 winners
    {"name":"v3_highreg",    "lgb":{"n_est":800,"lr":0.03,"nl":20,"ra":1.0,"rl":10.0}, "decay":0.6},
    {"name":"v3_cps003",     "lgb":{"n_est":1500,"lr":0.015,"nl":25,"ra":0.3,"rl":3.0}, "decay":0.5},
    {"name":"v3_cps01",      "lgb":{"n_est":1200,"lr":0.02,"nl":31,"ra":0.5,"rl":5.0}, "decay":0.5},
    {"name":"v3_default",    "lgb":{"n_est":1200,"lr":0.02,"nl":31,"ra":0.5,"rl":5.0}, "decay":0.0},
    # New: even higher reg
    {"name":"ultra_reg",     "lgb":{"n_est":600,"lr":0.04,"nl":15,"ra":2.0,"rl":15.0}, "decay":0.7},
    # New: more trees, lower lr
    {"name":"deep_slow",     "lgb":{"n_est":2000,"lr":0.01,"nl":20,"ra":0.5,"rl":5.0}, "decay":0.5},
    # New: no decay (static blend)
    {"name":"static_highreg","lgb":{"n_est":800,"lr":0.03,"nl":20,"ra":1.0,"rl":10.0}, "decay":0.0},
    # New: mild decay
    {"name":"mild_decay",    "lgb":{"n_est":1200,"lr":0.02,"nl":31,"ra":0.5,"rl":5.0}, "decay":0.3},
]

FOLD_YEARS = [2020, 2021, 2022]

print(f"\nConfigs: {len(CONFIGS)}, Folds: {FOLD_YEARS}")
print("=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)

all_cv = {}

for col in ["Revenue", "COGS"]:
    print(f"\n── {col} ──")
    tm = tet_m_rev if col=="Revenue" else tet_m_cogs
    cfg_results = []

    for ci, cfg in enumerate(CONFIGS):
        fold_maes = []
        fold_weights = []
        t0 = time.time()

        for year in FOLD_YEARS:
            train = sales[sales.Date.dt.year < year].copy()
            valid = sales[sales.Date.dt.year == year].copy()
            if len(train)<365 or len(valid)<30: continue

            vd = valid.Date.tolist(); yv = valid[col].values.astype(float)
            hol = build_holidays(valid.Date.max())

            preds = run_full_pipeline(train, col, vd, hol, cfg["lgb"])

            # Optimize static weights
            bw, bm = optimize_weights_3(yv, preds["naive"], preds["prophet"], preds["hybrid"])

            # Also try dynamic blend
            pd_dyn = dynamic_blend(preds["naive"], preds["prophet"], preds["hybrid"], bw, cfg["decay"])
            dm = mean_absolute_error(yv, pd_dyn)

            # Also try with Tet calibration
            base_med = float(train[col].median())
            pd_tet = apply_tet_cal(pd_dyn, vd, tm, base_med, strength=0.3)
            tm_mae = mean_absolute_error(yv, pd_tet)

            # Pick best of 3
            best_fold_mae = min(bm, dm, tm_mae)
            fold_maes.append(best_fold_mae)
            fold_weights.append(bw)

        elapsed = time.time() - t0
        if fold_maes:
            w = np.array([1.0+i*0.5 for i in range(len(fold_maes))]); w/=w.sum()
            avg_mae = float(np.dot(fold_maes, w))
            avg_w = tuple(float(np.mean([fw[j] for fw in fold_weights])) for j in range(3))
        else:
            avg_mae = 1e18; avg_w = (0.33,0.33,0.34)

        cfg_results.append({"name":cfg["name"],"mae":avg_mae,"weights":avg_w,"decay":cfg["decay"],"time":elapsed})
        print(f"  [{ci+1}/{len(CONFIGS)}] {cfg['name']}: MAE={avg_mae:,.0f} "
              f"w=[{avg_w[0]:.2f},{avg_w[1]:.2f},{avg_w[2]:.2f}] ({elapsed:.0f}s)")

    cfg_results.sort(key=lambda x: x["mae"])
    all_cv[col] = cfg_results

    print(f"\n  TOP 3 {col}:")
    for j, r in enumerate(cfg_results[:3]):
        print(f"    {j+1}. {r['name']}: MAE={r['mae']:,.0f}")

print("\n" + "=" * 70)
print("CV COMPLETE")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# FINAL FORECAST
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("GENERATING FINAL FORECASTS")
print("=" * 70)

best_cfg = {}
best_weights = {}
best_decay = {}

for col in ["Revenue", "COGS"]:
    r = all_cv[col][0]  # best
    cfg_obj = next(c for c in CONFIGS if c["name"]==r["name"])
    best_cfg[col] = cfg_obj
    best_weights[col] = r["weights"]
    best_decay[col] = r["decay"]
    print(f"{col}: {r['name']} (CV MAE={r['mae']:,.0f}, w={r['weights']}, decay={r['decay']})")

holidays_final = build_holidays(pd.Timestamp(max(forecast_dates)))

# Generate predictions for each target
final_preds = {}
for col in ["Revenue", "COGS"]:
    print(f"\n  {col}...", end=" ", flush=True)
    t0 = time.time()
    preds = run_full_pipeline(sales, col, forecast_dates, holidays_final, best_cfg[col]["lgb"])
    final_preds[col] = preds
    print(f"done ({time.time()-t0:.0f}s)")

# Create submission variants
VARIANTS = {
    "balanced": {"tet_strength": 0.3, "use_decay": True},
    "conservative": {"tet_strength": 0.4, "use_decay": True},
    "aggressive": {"tet_strength": 0.2, "use_decay": True},
    "pure_optimized": {"tet_strength": 0.0, "use_decay": False},  # pure CV-optimized static
}

submissions = {}
for vname, vcfg in VARIANTS.items():
    sub = sub_tpl[["Date"]].copy()
    for col in ["Revenue", "COGS"]:
        preds = final_preds[col]
        w = best_weights[col]
        decay = best_decay[col] if vcfg["use_decay"] else 0.0

        if decay > 0:
            p = dynamic_blend(preds["naive"], preds["prophet"], preds["hybrid"], w, decay)
        else:
            p = w[0]*preds["naive"] + w[1]*preds["prophet"] + w[2]*preds["hybrid"]
            p = np.clip(p, 0, None)

        if vcfg["tet_strength"] > 0:
            tm = tet_m_rev if col=="Revenue" else tet_m_cogs
            base_med = float(sales[col].median())
            p = apply_tet_cal(p, forecast_dates, tm, base_med, vcfg["tet_strength"])

        sub[col] = p
    submissions[vname] = sub

# Validate & save
print("\n" + "=" * 70)
print("VALIDATION & OUTPUT")
print("=" * 70)

output_files = {}
for vname, sub in submissions.items():
    ok = len(sub)==N_FC and not sub[["Revenue","COGS"]].isna().any().any() and not (sub[["Revenue","COGS"]]<0).any().any()
    status = "✅" if ok else "❌"
    print(f"  {status} {vname}: {len(sub)} rows")
    for c in ["Revenue","COGS"]:
        v=sub[c]; print(f"     {c}: mean={v.mean():,.0f} med={v.median():,.0f} min={v.min():,.0f} max={v.max():,.0f}")

    fname = f"submission_v4_{vname}.csv"
    fpath = os.path.join(OUT_DIR, fname)
    out = sub.copy(); out["Date"]=out["Date"].dt.strftime("%Y-%m-%d")
    out.to_csv(fpath, index=False)
    output_files[vname] = fpath

# Diagnostics
diag = {
    "version": "v4_topkill_v2",
    "timestamp": pd.Timestamp.now().isoformat(),
    "cv_results": {col: [{"name":r["name"],"mae":r["mae"],"weights":list(r["weights"]),"decay":r["decay"]}
                         for r in all_cv[col]] for col in ["Revenue","COGS"]},
    "best_config": {col: {"name":best_cfg[col]["name"],"weights":list(best_weights[col]),
                          "decay":best_decay[col]} for col in ["Revenue","COGS"]},
}
dp = os.path.join(OUT_DIR, "v4_diagnostics.json")
with open(dp, "w") as f: json.dump(diag, f, indent=2, default=str)

print(f"\nDiagnostics: {dp}")
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for col in ["Revenue","COGS"]:
    r=all_cv[col][0]
    print(f"  {col}: {r['name']} CV MAE={r['mae']:,.0f}")
print(f"\nFiles: {list(output_files.keys())}")
print("\n🏁 Upload submission_v4_balanced.csv to Kaggle.")
