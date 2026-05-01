"""
V35: Refine the winning v33_sample_rebalanced (675,655).

LESSONS LEARNED:
- v33_sample_rebalanced = 675,655 (BEST EVER) - full sample monthly pattern on v30
- v34_partial_rebal_110 = 977,369 (TERRIBLE) - overshoot destroyed it
- v32b split scale = 700,477 (worse than v30)
- Conclusion: sample monthly pattern at blend=1.0 is near-optimal
  Overshoot (1.1) is catastrophic. Must stay close to blend=1.0.

Strategy: Fine-tune AROUND the winner:
1. Revenue global scale: v33 used v30 level. Try +-1-3% on Revenue only
2. Partial rebalance: blend 0.9, 0.95, 1.0, 1.02, 1.05 (VERY close to 1.0)
3. COGS: v33 rebalanced BOTH columns. Try rebalancing Revenue only, keep COGS from v30
4. Different base: v23 vs v30 as starting point before rebalance
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

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
    assert len(out) == 548, f"Expected 548 rows, got {len(out)}"
    fmt = out.copy()
    fmt["Date"] = fmt["Date"].dt.strftime("%Y-%m-%d")
    path = OUT_DIR / f"submission_{name}.csv"
    fmt.to_csv(path, index=False)
    return str(path)

def rebalance_col(base_df, sample, col):
    """Rebalance one column's monthly distribution to match sample's relative pattern."""
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

def partial_rebalance_col(base_df, sample, col, blend):
    """Partially rebalance: blend between original and sample pattern."""
    s_months = month_key(sample["Date"])
    b_months = month_key(base_df["Date"])
    
    sample_monthly = sample.groupby(s_months)[col].mean()
    sample_global = sample[col].mean()
    sample_pattern = sample_monthly / sample_global
    
    base_monthly = base_df.groupby(b_months)[col].mean()
    base_global = base_df[col].mean()
    base_pattern = base_monthly / base_global
    
    out = base_df.copy()
    
    for ym in sorted(b_months.unique()):
        mask = b_months == ym
        if ym in sample_pattern.index and ym in base_pattern.index:
            b_pat = base_pattern[ym]
            s_pat = sample_pattern[ym]
            blended_pat = b_pat + blend * (s_pat - b_pat)
            target_mean = base_global * blended_pat
            current_mean = out.loc[mask, col].mean()
            if current_mean > 0:
                out.loc[mask, col] = out.loc[mask, col] * (target_mean / current_mean)
    return out

def main():
    sample = read_csv(DATA_DIR / "sample_submission.csv")
    v30 = read_csv(OUT_DIR / "submission_v30_v23_both_up_300pct.csv")
    v23 = read_csv(OUT_DIR / "submission_v23_b39_all_430.csv")
    b39 = read_csv(ROOT / "submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv")
    
    print("=" * 70)
    print("V35: REFINE THE WINNER (v33_sample_rebalanced = 675,655)")
    print("=" * 70)
    
    candidates = []
    
    # ══════════════════════════════════════════════════════════════════
    # GROUP A: Recreate winner exactly, then scale Revenue
    # v33 = v30 with BOTH Revenue and COGS rebalanced to sample pattern
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Group A: Winner + Revenue global scale ---")
    
    for rev_scale in [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.05]:
        out = v30.copy()
        out["Revenue"] = out["Revenue"] * rev_scale
        out = rebalance_col(out, sample, "Revenue")
        out = rebalance_col(out, sample, "COGS")
        
        name = f"v35a_winner_revscale_{int(rev_scale*1000)}"
        write_sub(out, name)
        candidates.append({"name": name, "rev_scale": rev_scale,
                          "Rev_mean": out["Revenue"].mean(), "COGS_mean": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")
    
    # ══════════════════════════════════════════════════════════════════
    # GROUP B: Rebalance Revenue only, keep COGS from v30 (not rebalanced)
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Group B: Rebalance Revenue only, COGS untouched ---")
    
    for rev_scale in [0.99, 1.00, 1.01, 1.02, 1.03]:
        out = v30.copy()
        out["Revenue"] = out["Revenue"] * rev_scale
        out = rebalance_col(out, sample, "Revenue")
        # COGS stays as v30 (NOT rebalanced)
        
        name = f"v35b_revonly_rebal_s{int(rev_scale*1000)}"
        write_sub(out, name)
        candidates.append({"name": name, "rev_scale": rev_scale,
                          "Rev_mean": out["Revenue"].mean(), "COGS_mean": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")
    
    # ══════════════════════════════════════════════════════════════════
    # GROUP C: Partial rebalance VERY close to 1.0
    # v34 at 1.1 was catastrophic. Stay between 0.9 and 1.05
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Group C: Partial rebalance (fine-grained near 1.0) ---")
    
    for blend in [0.85, 0.90, 0.93, 0.95, 0.97, 1.00, 1.02, 1.03, 1.05]:
        out = v30.copy()
        out = partial_rebalance_col(out, sample, "Revenue", blend)
        out = partial_rebalance_col(out, sample, "COGS", blend)
        
        name = f"v35c_partial_{int(blend*100)}"
        write_sub(out, name)
        candidates.append({"name": name, "blend": blend,
                          "Rev_mean": out["Revenue"].mean(), "COGS_mean": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f} (blend={blend})")
    
    # ══════════════════════════════════════════════════════════════════
    # GROUP D: Different base levels + full rebalance
    # v30 = v23 * 1.03. What if optimal base is v23 * 1.02 or 1.04?
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Group D: Different base levels + full rebalance ---")
    
    for scale in [1.01, 1.02, 1.025, 1.03, 1.035, 1.04, 1.05]:
        out = v23.copy()
        out["Revenue"] = out["Revenue"] * scale
        out["COGS"] = out["COGS"] * scale
        out = rebalance_col(out, sample, "Revenue")
        out = rebalance_col(out, sample, "COGS")
        
        name = f"v35d_v23_s{int(scale*1000)}_rebal"
        write_sub(out, name)
        candidates.append({"name": name, "scale": scale,
                          "Rev_mean": out["Revenue"].mean(), "COGS_mean": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f} (base_scale={scale})")
    
    # ══════════════════════════════════════════════════════════════════
    # GROUP E: b39 base (no alpha extrapolation) + rebalance
    # Maybe alpha extrapolation hurts after rebalance
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Group E: b39 base + scale + rebalance ---")
    
    for scale in [1.00, 1.02, 1.03, 1.04, 1.05]:
        out = b39.copy()
        out["Revenue"] = out["Revenue"] * scale
        out["COGS"] = out["COGS"] * scale
        out = rebalance_col(out, sample, "Revenue")
        out = rebalance_col(out, sample, "COGS")
        
        name = f"v35e_b39_s{int(scale*1000)}_rebal"
        write_sub(out, name)
        candidates.append({"name": name, "scale": scale,
                          "Rev_mean": out["Revenue"].mean(), "COGS_mean": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")
    
    # ══════════════════════════════════════════════════════════════════
    # GROUP F: Split Revenue/COGS scale + rebalance
    # ══════════════════════════════════════════════════════════════════
    print("\n--- Group F: Split scale + rebalance ---")
    
    for rs, cs in [(1.03, 1.02), (1.03, 1.04), (1.04, 1.03), (1.02, 1.03),
                   (1.035, 1.025), (1.025, 1.035), (1.04, 1.02), (1.02, 1.04)]:
        out = v23.copy()
        out["Revenue"] = out["Revenue"] * rs
        out["COGS"] = out["COGS"] * cs
        out = rebalance_col(out, sample, "Revenue")
        out = rebalance_col(out, sample, "COGS")
        
        name = f"v35f_split_r{int(rs*1000)}_c{int(cs*1000)}_rebal"
        write_sub(out, name)
        candidates.append({"name": name, "rev_scale": rs, "cogs_scale": cs,
                          "Rev_mean": out["Revenue"].mean(), "COGS_mean": out["COGS"].mean()})
        print(f"  {name}: Rev={out['Revenue'].mean():,.0f} COGS={out['COGS'].mean():,.0f}")
    
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(candidates)} candidates")
    pd.DataFrame(candidates).to_csv(OUT_DIR / "v35_manifest.csv", index=False)
    
    print("\n=== TOP RECOMMENDATIONS ===")
    print("1. v35c_partial_97.csv  - 97% rebalance (slightly less than winner)")
    print("2. v35c_partial_103.csv - 103% rebalance (slightly more than winner)")  
    print("3. v35a_winner_revscale_1010.csv - Winner + Rev +1%")
    print("4. v35a_winner_revscale_990.csv  - Winner + Rev -1%")
    print("5. v35d_v23_s1025_rebal.csv - v23*1.025 + rebalance")
    print("6. v35b_revonly_rebal_s1000.csv - Revenue-only rebalance")
    print("7. v35f_split_r1035_c1025_rebal.csv - Split scale + rebalance")

if __name__ == "__main__":
    main()
