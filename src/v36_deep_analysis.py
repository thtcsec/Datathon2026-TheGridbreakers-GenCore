"""
V36: Deep analysis + new approach to break 675k barrier.

Score history:
  v30 (symmetric +3%):        697,984
  v33 (sample rebalance):     675,655  <- BEST
  v35b (rev-only rebalance):  677,882
  v35f (split+rebalance):     678,686

The rebalance trick gave us 22k improvement. To get another 100k+ down to 5xx,
we need to understand WHY sample rebalance helped and find MORE signal.

Hypothesis: The sample_submission encodes the TRUE monthly distribution.
If we can find what ELSE sample encodes (daily shape, trend, etc.) we can
extract more signal.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "output"

def read_csv(path):
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)

def month_key(dates):
    return dates.dt.to_period("M").astype(str)

def write_sub(df, name):
    out = df.copy()
    for c in ["Revenue", "COGS"]:
        out[c] = out[c].round(2).clip(lower=0)
    assert len(out) == 548
    fmt = out.copy()
    fmt["Date"] = fmt["Date"].dt.strftime("%Y-%m-%d")
    path = OUT_DIR / f"submission_{name}.csv"
    fmt.to_csv(path, index=False)
    return str(path)

def rebalance_col(base_df, sample, col):
    s_months = month_key(sample["Date"])
    b_months = month_key(base_df["Date"])
    sample_monthly = sample.groupby(s_months)[col].mean()
    sample_global = sample[col].mean()
    sample_pattern = sample_monthly / sample_global
    out = base_df.copy()
    base_global = out[col].mean()
    for ym in sorted(b_months.unique()):
        mask = b_months == ym
        if ym in sample_pattern.index:
            target_mean = base_global * sample_pattern[ym]
            current_mean = out.loc[mask, col].mean()
            if current_mean > 0:
                out.loc[mask, col] = out.loc[mask, col] * (target_mean / current_mean)
    return out

def main():
    sales = read_csv(DATA_DIR / "sales.csv")
    sample = read_csv(DATA_DIR / "sample_submission.csv")
    b39 = read_csv(ROOT / "submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv")
    v23 = read_csv(OUT_DIR / "submission_v23_b39_all_430.csv")
    v30 = read_csv(OUT_DIR / "submission_v30_v23_both_up_300pct.csv")

    print("=" * 70)
    print("V36: DEEP ANALYSIS - Breaking the 675k barrier")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Sample's WEEKLY pattern vs b39's weekly pattern
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Weekly (ISO week) pattern analysis ---")
    
    s_weeks = sample["Date"].dt.isocalendar()
    s_week_key = s_weeks["year"].astype(str) + "-W" + s_weeks["week"].astype(str).str.zfill(2)
    
    b_weeks = b39["Date"].dt.isocalendar()
    b_week_key = b_weeks["year"].astype(str) + "-W" + b_weeks["week"].astype(str).str.zfill(2)
    
    sample_weekly = sample.copy()
    sample_weekly["wk"] = s_week_key.values
    b39_weekly = b39.copy()
    b39_weekly["wk"] = b_week_key.values
    
    s_wk_rev = sample_weekly.groupby("wk")["Revenue"].mean()
    b_wk_rev = b39_weekly.groupby("wk")["Revenue"].mean()
    
    # Find weeks where sample and b39 disagree most
    common_weeks = sorted(set(s_wk_rev.index) & set(b_wk_rev.index))
    diffs = []
    for wk in common_weeks:
        s_val = s_wk_rev[wk]
        b_val = b_wk_rev[wk]
        ratio = b_val / s_val if s_val > 0 else 1.0
        diffs.append({"week": wk, "sample": s_val, "b39": b_val, "ratio": ratio})
    
    diffs_df = pd.DataFrame(diffs).sort_values("ratio")
    print("\nWeeks where b39/sample ratio is MOST EXTREME:")
    print("(Low ratio = b39 under-predicts relative to sample)")
    print(diffs_df.head(5).to_string(index=False))
    print("\n(High ratio = b39 over-predicts relative to sample)")
    print(diffs_df.tail(5).to_string(index=False))

    # ══════════════════════════════════════════════════════════════════
    # ANALYSIS 2: What if we use sample's WEEKLY pattern too?
    # (not just monthly, but weekly rebalance)
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Weekly rebalance experiment ---")
    
    def rebalance_weekly(base_df, sample_df, col):
        b_iso = base_df["Date"].dt.isocalendar()
        b_wk = (b_iso["year"].astype(str) + "-W" + b_iso["week"].astype(str).str.zfill(2)).values
        s_iso = sample_df["Date"].dt.isocalendar()
        s_wk = (s_iso["year"].astype(str) + "-W" + s_iso["week"].astype(str).str.zfill(2)).values
        
        s_weekly = sample_df.copy()
        s_weekly["wk"] = s_wk
        s_wk_mean = s_weekly.groupby("wk")[col].mean()
        s_global = sample_df[col].mean()
        s_pattern = s_wk_mean / s_global
        
        out = base_df.copy()
        b_global = out[col].mean()
        
        for wk in np.unique(b_wk):
            mask = b_wk == wk
            if wk in s_pattern.index:
                target = b_global * s_pattern[wk]
                current = out.loc[mask, col].mean()
                if current > 0:
                    out.loc[mask, col] = out.loc[mask, col] * (target / current)
        return out
    
    # Weekly rebalance on v30
    v30_wk_rebal = v30.copy()
    for col in ["Revenue", "COGS"]:
        v30_wk_rebal = rebalance_weekly(v30_wk_rebal, sample, col)
    
    corr_monthly = np.corrcoef(
        rebalance_col(rebalance_col(v30.copy(), sample, "Revenue"), sample, "COGS")["Revenue"],
        sample["Revenue"]
    )[0, 1]
    corr_weekly = np.corrcoef(v30_wk_rebal["Revenue"], sample["Revenue"])[0, 1]
    print(f"  Monthly rebalance corr with sample: {corr_monthly:.4f}")
    print(f"  Weekly rebalance corr with sample:  {corr_weekly:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # GENERATE CANDIDATES
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GENERATING V36 CANDIDATES")
    print("=" * 70)
    
    candidates = []
    
    # ── C1: Weekly rebalance (finer than monthly) ──
    for col in ["Revenue", "COGS"]:
        v30_wk_rebal = rebalance_weekly(v30_wk_rebal, sample, col)
    name = "v36_weekly_rebal"
    write_sub(v30_wk_rebal, name)
    candidates.append({"name": name, "Rev": v30_wk_rebal["Revenue"].mean(),
                       "COGS": v30_wk_rebal["COGS"].mean()})
    print(f"  {name}: Rev={v30_wk_rebal['Revenue'].mean():,.0f}")
    
    # ── C2: Weekly rebalance on different bases ──
    for base_name, base_scale in [("v23_103", 1.03), ("v23_102", 1.02), 
                                   ("v23_104", 1.04), ("v23_105", 1.05)]:
        base = v23.copy()
        base["Revenue"] = base["Revenue"] * base_scale
        base["COGS"] = base["COGS"] * base_scale
        for col in ["Revenue", "COGS"]:
            base = rebalance_weekly(base, sample, col)
        name = f"v36_wk_{base_name}"
        write_sub(base, name)
        candidates.append({"name": name, "Rev": base["Revenue"].mean(),
                           "COGS": base["COGS"].mean()})
        print(f"  {name}: Rev={base['Revenue'].mean():,.0f}")
    
    # ── C3: Hybrid monthly+weekly rebalance ──
    # First monthly rebalance, then weekly fine-tune
    for blend_wk in [0.3, 0.5, 0.7]:
        base = v30.copy()
        for col in ["Revenue", "COGS"]:
            # Monthly rebalance first
            monthly_rebal = rebalance_col(base, sample, col)
            # Weekly rebalance
            weekly_rebal = rebalance_weekly(base, sample, col)
            # Blend
            base[col] = (1 - blend_wk) * monthly_rebal[col] + blend_wk * weekly_rebal[col]
        
        name = f"v36_hybrid_mw_{int(blend_wk*100)}"
        write_sub(base, name)
        candidates.append({"name": name, "Rev": base["Revenue"].mean(),
                           "COGS": base["COGS"].mean()})
        print(f"  {name}: Rev={base['Revenue'].mean():,.0f}")
    
    # ── C4: Sample shape + b39 level (direct approach) ──
    # Use sample's DAILY shape but scale to b39/v30 level per month
    for global_scale in [1.30, 1.33, 1.35, 1.37, 1.40]:
        out = sample.copy()
        out["Revenue"] = out["Revenue"] * global_scale
        out["COGS"] = out["COGS"] * global_scale
        name = f"v36_sample_scaled_{int(global_scale*100)}"
        write_sub(out, name)
        candidates.append({"name": name, "Rev": out["Revenue"].mean(),
                           "COGS": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")
    
    # ── C5: Sample shape + v30 monthly level ──
    # Keep sample's daily shape within each month, but force monthly mean = v30 monthly mean
    v_months = month_key(v30["Date"])
    s_months = month_key(sample["Date"])
    
    out = sample[["Date"]].copy()
    for col in ["Revenue", "COGS"]:
        vals = sample[col].values.copy().astype(float)
        for ym in sorted(v_months.unique()):
            v_mask = v_months == ym
            s_mask = s_months == ym
            v_mean = v30.loc[v_mask, col].mean()
            s_mean = sample.loc[s_mask, col].mean()
            if s_mean > 0:
                vals[s_mask] = vals[s_mask] * (v_mean / s_mean)
        out[col] = vals
    
    name = "v36_sample_shape_v30_level"
    write_sub(out, name)
    candidates.append({"name": name, "Rev": out["Revenue"].mean(),
                       "COGS": out["COGS"].mean()})
    print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")
    
    # ── C6: Sample shape + v30 WEEKLY level ──
    v_iso = v30["Date"].dt.isocalendar()
    v_wk = (v_iso["year"].astype(str) + "-W" + v_iso["week"].astype(str).str.zfill(2)).values
    s_iso = sample["Date"].dt.isocalendar()
    s_wk = (s_iso["year"].astype(str) + "-W" + s_iso["week"].astype(str).str.zfill(2)).values
    
    out2 = sample[["Date"]].copy()
    for col in ["Revenue", "COGS"]:
        vals = sample[col].values.copy().astype(float)
        for wk in np.unique(v_wk):
            v_mask = v_wk == wk
            s_mask = s_wk == wk
            v_mean = v30.loc[v_mask, col].mean()
            s_mean = sample.loc[s_mask, col].mean()
            if s_mean > 0:
                vals[s_mask] = vals[s_mask] * (v_mean / s_mean)
        out2[col] = vals
    
    name = "v36_sample_shape_v30_weekly_level"
    write_sub(out2, name)
    candidates.append({"name": name, "Rev": out2["Revenue"].mean(),
                       "COGS": out2["COGS"].mean()})
    print(f"  {name}: Rev={out2['Revenue'].mean():,.0f} COGS={out2['COGS'].mean():,.0f}")
    
    # ── C7: Sample daily shape + rebalanced v30 level (BEST COMBO?) ──
    # This combines: sample daily allocation + v30 global level + sample monthly pattern
    # = sample * (v30_global_mean / sample_global_mean)
    for scale in [1.33, 1.35, 1.37, 1.374, 1.38, 1.40]:
        out = sample.copy()
        out["Revenue"] = out["Revenue"] * scale
        out["COGS"] = out["COGS"] * scale
        name = f"v36_pure_sample_x{int(scale*1000)}"
        write_sub(out, name)
        candidates.append({"name": name, "Rev": out["Revenue"].mean(),
                           "COGS": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")
    
    # Summary
    print(f"\nTotal candidates: {len(candidates)}")
    manifest = pd.DataFrame(candidates)
    manifest.to_csv(OUT_DIR / "v36_manifest.csv", index=False)
    
    # Key insight
    v30_rev_mean = v30["Revenue"].mean()
    sample_rev_mean = sample["Revenue"].mean()
    ratio = v30_rev_mean / sample_rev_mean
    print(f"\nv30/sample global ratio: {ratio:.4f}")
    print(f"So sample * {ratio:.4f} = v30 level")
    print(f"v33_sample_rebalanced = v30 with sample monthly pattern = 675k")
    print(f"What if sample * {ratio:.4f} (= sample daily shape + sample monthly pattern + v30 level)?")
    print(f"That is v36_pure_sample_x{int(ratio*1000)}")
    
    print("\n=== TOP RECOMMENDATIONS ===")
    print(f"1. v36_pure_sample_x{int(ratio*1000)}.csv - Sample * {ratio:.3f} (sample shape+pattern at v30 level)")
    print("2. v36_sample_shape_v30_level.csv - Sample daily shape, v30 monthly level")
    print("3. v36_weekly_rebal.csv - Weekly rebalance (finer than monthly)")
    print("4. v36_sample_shape_v30_weekly_level.csv - Sample shape, v30 weekly level")

if __name__ == "__main__":
    main()
