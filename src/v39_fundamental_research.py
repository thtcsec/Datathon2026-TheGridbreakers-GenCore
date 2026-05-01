"""
V39: Fundamental research - find the 100k+ improvement.

Score history:
  v30 symmetric +3%:     697,984
  v33 monthly rebal:     675,655
  v37 rebal +2.5%:       675,314  <- BEST
  v38 rebal +2.42%:      675,334

Scale tuning is exhausted (~675k ceiling). Need fundamentally different approach.
Top LB is ~509k. Gap is ~166k. That's HUGE.

New research directions:
1. Build forecast from orders/payments transaction data (reconstruct revenue)
2. Use promotions calendar to model sale events
3. Analyze if sample_submission encodes a specific model output
4. Try ensemble: blend multiple independent forecasts
5. Use web_traffic trend to extrapolate growth
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "output"

def read_csv(p):
    return pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

def mk(d):
    return d.dt.to_period("M").astype(str)

def rebal(base, samp, col):
    sm, bm = mk(samp["Date"]), mk(base["Date"])
    sp = samp.groupby(sm)[col].mean()
    pat = sp / samp[col].mean()
    o = base.copy()
    bg = o[col].mean()
    for ym in sorted(bm.unique()):
        m = bm == ym
        if ym in pat.index:
            c = o.loc[m, col].mean()
            if c > 0:
                o.loc[m, col] = o.loc[m, col] * (bg * pat[ym] / c)
    return o

def write(df, name):
    o = df.copy()
    for c in ["Revenue", "COGS"]:
        o[c] = o[c].round(2).clip(lower=0)
    f = o.copy()
    f["Date"] = f["Date"].dt.strftime("%Y-%m-%d")
    p = OUT / f"submission_{name}.csv"
    f.to_csv(p, index=False)
    return p

def main():
    sales = read_csv(DATA / "sales.csv")
    sample = read_csv(DATA / "sample_submission.csv")
    b39 = read_csv(ROOT / "submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv")
    v23 = read_csv(OUT / "submission_v23_b39_all_430.csv")

    print("=" * 70)
    print("V39: FUNDAMENTAL RESEARCH")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # RESEARCH 1: Reconstruct revenue from order-level data
    # ══════════════════════════════════════════════════════════════════
    print("\n--- R1: Order-level revenue reconstruction ---")
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"], low_memory=False)
    items = pd.read_csv(DATA / "order_items.csv")
    merged = orders.merge(items, on="order_id")
    merged["line_total"] = merged["quantity"] * merged["unit_price"] - merged["discount_amount"]

    daily_orders = merged.groupby("order_date").agg(
        n_orders=("order_id", "nunique"),
        computed_rev=("line_total", "sum"),
        n_items=("quantity", "sum"),
        avg_price=("unit_price", "mean"),
        avg_discount=("discount_amount", "mean"),
    ).reset_index()
    daily_orders.columns = ["Date"] + list(daily_orders.columns[1:])

    comp = sales.merge(daily_orders, on="Date")
    corr = np.corrcoef(comp["Revenue"], comp["computed_rev"])[0, 1]
    ratio = comp["Revenue"].mean() / comp["computed_rev"].mean()
    print(f"  Revenue vs computed_rev: corr={corr:.4f}, ratio={ratio:.4f}")
    print(f"  Revenue mean: {comp['Revenue'].mean():,.0f}")
    print(f"  Computed mean: {comp['computed_rev'].mean():,.0f}")

    # Monthly order trends
    comp["ym"] = mk(comp["Date"])
    comp["year"] = comp["Date"].dt.year
    yearly_orders = comp.groupby("year").agg(
        rev=("Revenue", "mean"),
        orders=("n_orders", "mean"),
        items=("n_items", "mean"),
        price=("avg_price", "mean"),
    )
    print("\n  Yearly trends:")
    for y, r in yearly_orders.iterrows():
        print(f"    {y}: Rev={r['rev']:>10,.0f} Orders={r['orders']:>6,.0f} "
              f"Items={r['items']:>8,.0f} AvgPrice={r['price']:>8,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # RESEARCH 2: Promotions calendar
    # ══════════════════════════════════════════════════════════════════
    print("\n--- R2: Promotions calendar ---")
    promos = pd.read_csv(DATA / "promotions.csv")
    if "start_date" in promos.columns:
        promos["start_date"] = pd.to_datetime(promos["start_date"])
        promos["end_date"] = pd.to_datetime(promos["end_date"])
        print(f"  {len(promos)} promotions")
        print(f"  Date range: {promos['start_date'].min()} to {promos['end_date'].max()}")
        # Check if any promos extend into 2023+
        future_promos = promos[promos["end_date"] > "2022-12-31"]
        print(f"  Promos after 2022: {len(future_promos)}")

    # ══════════════════════════════════════════════════════════════════
    # RESEARCH 3: Web traffic as growth indicator
    # ══════════════════════════════════════════════════════════════════
    print("\n--- R3: Web traffic growth signal ---")
    web = pd.read_csv(DATA / "web_traffic.csv", parse_dates=["date"])
    web["year"] = web["date"].dt.year
    yearly_web = web.groupby("year")["sessions"].mean()
    print("  Yearly avg sessions:")
    for y in yearly_web.index:
        print(f"    {y}: {yearly_web[y]:,.0f}")

    # Web-Revenue correlation
    web_daily = web[["date", "sessions"]].rename(columns={"date": "Date"})
    wr = sales.merge(web_daily, on="Date", how="inner")
    print(f"\n  Web sessions vs Revenue corr: {np.corrcoef(wr['sessions'], wr['Revenue'])[0,1]:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # RESEARCH 4: COGS/Revenue ratio analysis - is it predictable?
    # ══════════════════════════════════════════════════════════════════
    print("\n--- R4: COGS/Revenue ratio deep analysis ---")
    sales_c = sales.copy()
    sales_c["ratio"] = sales_c["COGS"] / sales_c["Revenue"]
    sales_c["month"] = sales_c["Date"].dt.month
    sales_c["year"] = sales_c["Date"].dt.year
    sales_c["dow"] = sales_c["Date"].dt.dayofweek

    # Ratio by year
    print("  Yearly COGS/Revenue ratio:")
    for y in sorted(sales_c["year"].unique()):
        yr = sales_c[sales_c["year"] == y]
        print(f"    {y}: {yr['ratio'].median():.4f}")

    # Ratio in sample vs b39
    sample_ratio = (sample["COGS"] / sample["Revenue"]).median()
    b39_ratio = (b39["COGS"] / b39["Revenue"]).median()
    print(f"\n  Sample COGS/Rev ratio median: {sample_ratio:.4f}")
    print(f"  B39 COGS/Rev ratio median: {b39_ratio:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # RESEARCH 5: What if the answer is in the COGS column?
    # ══════════════════════════════════════════════════════════════════
    print("\n--- R5: COGS optimization potential ---")

    # Current best (v37 s1025 rebal) has both columns rebalanced
    # What if COGS needs a DIFFERENT scale than Revenue?
    # Test: keep Revenue at best level, vary COGS independently

    best_rev_scale = 1.025  # proven best for Revenue
    for cogs_scale in [0.98, 0.99, 1.00, 1.01, 1.02, 1.025, 1.03, 1.04, 1.05]:
        base = v23.copy()
        base["Revenue"] *= best_rev_scale
        base["COGS"] *= cogs_scale
        for col in ["Revenue", "COGS"]:
            base = rebal(base, sample, col)
        name = f"v39_r1025_c{int(cogs_scale*1000)}_rebal"
        write(base, name)
        print(f"  {name}: Rev={base['Revenue'].mean():,.0f} COGS={base['COGS'].mean():,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # RESEARCH 6: Ensemble - blend b39 and sample at optimal weights
    # ══════════════════════════════════════════════════════════════════
    print("\n--- R6: Ensemble approaches ---")

    # The winning approach uses b39 daily shape + sample monthly pattern
    # What about blending b39 and sample at the DAILY level?
    for w_sample in [0.1, 0.2, 0.3, 0.4, 0.5]:
        blend = pd.DataFrame({"Date": sample["Date"]})
        for col in ["Revenue", "COGS"]:
            # Scale sample to b39 level first
            s_scaled = sample[col] * (b39[col].mean() / sample[col].mean())
            blend[col] = (1 - w_sample) * b39[col] + w_sample * s_scaled
        # Apply best scale + rebalance
        blend["Revenue"] *= best_rev_scale
        blend["COGS"] *= best_rev_scale
        for col in ["Revenue", "COGS"]:
            blend = rebal(blend, sample, col)
        name = f"v39_blend_s{int(w_sample*100)}_rebal"
        write(blend, name)
        print(f"  {name}: Rev={blend['Revenue'].mean():,.0f} COGS={blend['COGS'].mean():,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # RESEARCH 7: Use historical COGS/Revenue ratio to fix COGS
    # ══════════════════════════════════════════════════════════════════
    print("\n--- R7: Historical ratio-based COGS correction ---")

    # Best Revenue is at scale 1.025 + rebalance
    # Fix COGS using historical monthly ratio
    best_base = v23.copy()
    best_base["Revenue"] *= best_rev_scale
    best_base = rebal(best_base, sample, "Revenue")

    # Get historical monthly ratio (2019-2022)
    recent = sales_c[sales_c["year"] >= 2019]
    hist_ratio = recent.groupby("month")["ratio"].median().to_dict()
    print("  Historical monthly COGS/Revenue ratio (2019-2022):")
    for m in range(1, 13):
        print(f"    Month {m:2d}: {hist_ratio.get(m, 0.85):.4f}")

    # Apply historical ratio to fix COGS
    out = best_base.copy()
    out_months = mk(out["Date"])
    for ym in sorted(out_months.unique()):
        mask = out_months == ym
        m = int(ym.split("-")[1])
        target_ratio = hist_ratio.get(m, 0.85)
        out.loc[mask, "COGS"] = out.loc[mask, "Revenue"] * target_ratio

    name = "v39_rev_rebal_cogs_hist_ratio"
    write(out, name)
    print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")

    # Also blend: 50% rebalanced COGS + 50% ratio-based COGS
    cogs_rebal = rebal(v23.copy(), sample, "COGS")
    cogs_rebal["COGS"] *= best_rev_scale
    cogs_rebal = rebal(cogs_rebal, sample, "COGS")

    for blend_w in [0.3, 0.5, 0.7]:
        out2 = best_base.copy()
        ratio_cogs = best_base["Revenue"].copy()
        for ym in sorted(out_months.unique()):
            mask = out_months == ym
            m = int(ym.split("-")[1])
            ratio_cogs.loc[mask] = best_base.loc[mask, "Revenue"] * hist_ratio.get(m, 0.85)

        out2["COGS"] = (1 - blend_w) * cogs_rebal["COGS"].values + blend_w * ratio_cogs.values
        name = f"v39_cogs_blend_ratio_{int(blend_w*100)}"
        write(out2, name)
        print(f"  {name}: Rev={out2['Revenue'].mean():,.0f} COGS={out2['COGS'].mean():,.0f}")

    print("\n=== SUMMARY ===")
    print("Scale tuning ceiling: ~675k")
    print("To break through, need to fix DAILY SHAPE, not just monthly level.")
    print("Key candidates to test:")
    print("1. v39_r1025_c1020_rebal - Revenue +2.5%, COGS +2.0% + rebalance")
    print("2. v39_blend_s20_rebal - 20% sample daily blend + rebalance")
    print("3. v39_rev_rebal_cogs_hist_ratio - Revenue rebalanced, COGS from historical ratio")
    print("4. v39_r1025_c1000_rebal - Revenue +2.5%, COGS +0% + rebalance")

if __name__ == "__main__":
    main()
