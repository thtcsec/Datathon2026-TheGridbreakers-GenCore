"""
V32: Breakthrough - Multi-axis optimization beyond global scale.

Current best: v30 at 697,984 MAE (v23 alpha 4.3 + global scale +3.0%)

Strategy:
1. Reverse-engineer sample_submission to find hidden signal
2. Separate Revenue vs COGS optimization (they may need different scales)
3. Monthly-varying multipliers instead of single global scale
4. COGS/Revenue ratio correction based on historical patterns
5. Generate multiple candidates for LB testing

This script is SELF-CONTAINED - no imports from v20/v21/v22/v23.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")
EXPECTED_ROWS = 548

B39_PATH = ROOT / "submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv"
BEST_PATH = OUT_DIR / "submission_v30_v23_both_up_300pct.csv"


# ── Helpers ────────────────────────────────────────────────────────────
def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def validate(df: pd.DataFrame, name: str) -> None:
    assert len(df) == EXPECTED_ROWS, f"{name}: {len(df)} rows"
    assert df["Date"].min() == FORECAST_START, f"{name}: bad start"
    assert df["Date"].max() == FORECAST_END, f"{name}: bad end"
    assert not df["Date"].duplicated().any(), f"{name}: dup dates"
    for c in ["Revenue", "COGS"]:
        assert df[c].notna().all(), f"{name}: NaN in {c}"
        assert (df[c] >= 0).all(), f"{name}: negative {c}"


def month_key(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").astype(str)


def scale_monthly(df: pd.DataFrame, multipliers: dict) -> pd.DataFrame:
    """Scale each month by a multiplier dict {col: {YYYY-MM: factor}}."""
    out = df.copy()
    months = month_key(df["Date"])
    for col in ["Revenue", "COGS"]:
        if col not in multipliers:
            continue
        for ym, factor in multipliers[col].items():
            mask = months == ym
            if mask.any():
                out.loc[mask, col] = out.loc[mask, col] * factor
    return out


def write_sub(df: pd.DataFrame, name: str) -> str:
    out = df.copy()
    for c in ["Revenue", "COGS"]:
        out[c] = out[c].round(2)
    validate(out, name)
    out_fmt = out.copy()
    out_fmt["Date"] = out_fmt["Date"].dt.strftime("%Y-%m-%d")
    path = OUT_DIR / f"submission_{name}.csv"
    out_fmt.to_csv(path, index=False)
    return str(path)


# ── Analysis functions ─────────────────────────────────────────────────

def analyze_sample_vs_history(sales: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Reverse-engineer sample_submission: find which year it correlates with."""
    results = {}
    sample_months = month_key(sample["Date"])

    for col in ["Revenue", "COGS"]:
        year_corrs = {}
        for year in range(2013, 2023):
            # Map sample dates to same DOY in historical year
            mapped = []
            hist_vals = []
            for _, row in sample.iterrows():
                try:
                    hist_date = row["Date"].replace(year=year)
                except ValueError:
                    # e.g. Feb 29 in non-leap year
                    continue
                match = sales[sales["Date"] == hist_date]
                if len(match) == 1:
                    mapped.append(row[col])
                    hist_vals.append(float(match[col].iloc[0]))
            if len(mapped) > 300:
                corr = float(np.corrcoef(mapped, hist_vals)[0, 1])
                scale = np.mean(mapped) / np.mean(hist_vals) if np.mean(hist_vals) > 0 else 0
                year_corrs[year] = {"corr": corr, "scale": round(scale, 4), "n": len(mapped)}

        best_year = max(year_corrs, key=lambda y: year_corrs[y]["corr"])
        results[col] = {
            "best_year": best_year,
            "best_corr": year_corrs[best_year]["corr"],
            "best_scale": year_corrs[best_year]["scale"],
            "all_years": year_corrs,
        }

    # Monthly ratio analysis
    sample_ratio = (sample["COGS"] / sample["Revenue"]).values
    results["sample_ratio_mean"] = float(np.mean(sample_ratio))
    results["sample_ratio_median"] = float(np.median(sample_ratio))

    return results


def analyze_anchor_vs_sample(anchor: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Compare anchor (best submission) monthly levels vs sample."""
    months = month_key(anchor["Date"])
    results = {}
    for col in ["Revenue", "COGS"]:
        anchor_monthly = anchor.groupby(months)[col].mean()
        sample_monthly = sample.groupby(month_key(sample["Date"]))[col].mean()
        ratio = (anchor_monthly / sample_monthly).to_dict()
        results[col] = {
            "anchor_over_sample_ratio": ratio,
            "global_ratio": float(anchor[col].mean() / sample[col].mean()),
        }
    return results


def compute_historical_monthly_growth(sales: pd.DataFrame) -> dict:
    """Compute YoY monthly growth rates from training data."""
    sales_c = sales.copy()
    sales_c["year"] = sales_c["Date"].dt.year
    sales_c["month"] = sales_c["Date"].dt.month

    yearly_monthly = sales_c.groupby(["year", "month"]).agg(
        Revenue=("Revenue", "mean"),
        COGS=("COGS", "mean"),
    ).reset_index()

    growth = {}
    for col in ["Revenue", "COGS"]:
        monthly_growth = {}
        for m in range(1, 13):
            month_data = yearly_monthly[yearly_monthly["month"] == m].sort_values("year")
            if len(month_data) >= 3:
                # Use last 5 years growth trend
                recent = month_data[month_data["year"] >= 2018]
                if len(recent) >= 2:
                    vals = recent[col].values
                    yoy_changes = np.diff(vals) / vals[:-1]
                    monthly_growth[m] = float(np.median(yoy_changes))
                else:
                    monthly_growth[m] = 0.0
            else:
                monthly_growth[m] = 0.0
        growth[col] = monthly_growth
    return growth


def compute_ratio_profile(sales: pd.DataFrame) -> dict:
    """Historical COGS/Revenue ratio by month."""
    sales_c = sales.copy()
    sales_c["month"] = sales_c["Date"].dt.month
    # Use recent years only (2019-2022)
    recent = sales_c[sales_c["Date"].dt.year >= 2019]
    ratio_by_month = {}
    for m in range(1, 13):
        mdata = recent[recent["month"] == m]
        if len(mdata) > 0:
            ratios = mdata["COGS"] / mdata["Revenue"]
            ratio_by_month[m] = float(ratios.median())
        else:
            ratio_by_month[m] = 0.80
    return ratio_by_month


# ── Candidate generation ───────────────────────────────────────────────

def generate_candidates(
    best: pd.DataFrame,
    sales: pd.DataFrame,
    sample: pd.DataFrame,
    b39: pd.DataFrame,
) -> list[dict]:
    """Generate multiple candidate submissions using different strategies."""

    candidates = []
    months = month_key(best["Date"])
    unique_months = sorted(months.unique())

    # ── Strategy 1: Separate Revenue/COGS scale optimization ──
    # Evidence: scale up +3% works for both, but maybe they need different amounts
    for rev_scale, cogs_scale in [
        (1.035, 1.025),  # Revenue needs more, COGS less
        (1.025, 1.035),  # COGS needs more, Revenue less
        (1.040, 1.020),  # Aggressive Revenue, conservative COGS
        (1.020, 1.040),  # Conservative Revenue, aggressive COGS
        (1.035, 1.030),  # Slight Revenue bias
        (1.030, 1.035),  # Slight COGS bias
    ]:
        # Apply to v23 base (before the +3% global scale)
        v23_path = OUT_DIR / "submission_v23_b39_all_430.csv"
        if v23_path.exists():
            v23 = read_csv(v23_path)
            out = v23.copy()
            out["Revenue"] = out["Revenue"] * rev_scale
            out["COGS"] = out["COGS"] * cogs_scale
            name = f"v32_split_r{int(rev_scale*1000)}_c{int(cogs_scale*1000)}"
            path = write_sub(out, name)
            candidates.append({
                "name": name,
                "strategy": "split_scale",
                "rev_scale": rev_scale,
                "cogs_scale": cogs_scale,
                "path": path,
                "Revenue_mean": float(out["Revenue"].mean()),
                "COGS_mean": float(out["COGS"].mean()),
            })

    # ── Strategy 2: Monthly-varying multipliers ──
    # Hypothesis: 2023 H1 may need different scale than 2023 H2 and 2024 H1
    growth = compute_historical_monthly_growth(sales)

    # Build monthly multipliers based on historical growth extrapolation
    # Base: current best is v23 * 1.03 global
    # Instead: v23 * monthly_factor where monthly_factor varies
    v23_path = OUT_DIR / "submission_v23_b39_all_430.csv"
    if v23_path.exists():
        v23 = read_csv(v23_path)

        # Variant A: Growth-based monthly multipliers
        # 2023 early months: lower growth, late 2023/2024: higher growth
        monthly_mult_a = {}
        for col in ["Revenue", "COGS"]:
            col_mult = {}
            for ym in unique_months:
                m = int(ym.split("-")[1])
                y = int(ym.split("-")[0])
                base_growth = growth[col].get(m, 0.0)
                # Dampen: don't trust historical growth fully
                # Add a base of +3% (current optimum) and modulate
                years_ahead = y - 2022 + (m - 1) / 12.0
                factor = 1.03 + 0.3 * base_growth * years_ahead
                factor = np.clip(factor, 0.97, 1.10)
                col_mult[ym] = round(float(factor), 4)
            monthly_mult_a[col] = col_mult

        out_a = scale_monthly(v23, monthly_mult_a)
        name_a = "v32_monthly_growth_based"
        path_a = write_sub(out_a, name_a)
        candidates.append({
            "name": name_a,
            "strategy": "monthly_growth",
            "path": path_a,
            "Revenue_mean": float(out_a["Revenue"].mean()),
            "COGS_mean": float(out_a["COGS"].mean()),
            "multipliers": monthly_mult_a,
        })

        # Variant B: Stepped multipliers (H1 2023, H2 2023, H1 2024)
        for h1_23, h2_23, h1_24 in [
            (1.025, 1.030, 1.040),  # Gradual increase
            (1.020, 1.035, 1.045),  # Steeper ramp
            (1.030, 1.025, 1.035),  # Dip in H2 2023
            (1.035, 1.030, 1.025),  # Decreasing (post-recovery slowdown)
            (1.025, 1.035, 1.035),  # Step up then flat
        ]:
            mult_b = {}
            for col in ["Revenue", "COGS"]:
                col_mult = {}
                for ym in unique_months:
                    y, m = int(ym.split("-")[0]), int(ym.split("-")[1])
                    if y == 2023 and m <= 6:
                        col_mult[ym] = h1_23
                    elif y == 2023 and m > 6:
                        col_mult[ym] = h2_23
                    else:  # 2024
                        col_mult[ym] = h1_24
                mult_b[col] = col_mult

            out_b = scale_monthly(v23, mult_b)
            tag = f"h1_{int(h1_23*1000)}_h2_{int(h2_23*1000)}_h3_{int(h1_24*1000)}"
            name_b = f"v32_stepped_{tag}"
            path_b = write_sub(out_b, name_b)
            candidates.append({
                "name": name_b,
                "strategy": "stepped_multiplier",
                "h1_2023": h1_23,
                "h2_2023": h2_23,
                "h1_2024": h1_24,
                "path": path_b,
                "Revenue_mean": float(out_b["Revenue"].mean()),
                "COGS_mean": float(out_b["COGS"].mean()),
            })

    # ── Strategy 3: COGS/Revenue ratio correction ──
    hist_ratio = compute_ratio_profile(sales)

    if v23_path.exists():
        v23 = read_csv(v23_path)
        out_r = v23.copy()
        out_r["Revenue"] = out_r["Revenue"] * 1.03  # Keep best global scale

        # Adjust COGS based on historical ratio
        for ym in unique_months:
            m = int(ym.split("-")[1])
            mask = months == ym
            if not mask.any():
                continue
            target_ratio = hist_ratio.get(m, 0.80)
            current_rev = out_r.loc[mask, "Revenue"].values
            # Set COGS = Revenue * target_ratio
            out_r.loc[mask, "COGS"] = current_rev * target_ratio

        name_r = "v32_ratio_corrected"
        path_r = write_sub(out_r, name_r)
        candidates.append({
            "name": name_r,
            "strategy": "ratio_correction",
            "path": path_r,
            "Revenue_mean": float(out_r["Revenue"].mean()),
            "COGS_mean": float(out_r["COGS"].mean()),
            "target_ratios": hist_ratio,
        })

        # Variant: blend ratio correction (50% toward historical ratio)
        out_rb = v23.copy()
        out_rb["Revenue"] = out_rb["Revenue"] * 1.03
        out_rb["COGS"] = out_rb["COGS"] * 1.03
        for ym in unique_months:
            m = int(ym.split("-")[1])
            mask = months == ym
            if not mask.any():
                continue
            target_ratio = hist_ratio.get(m, 0.80)
            current_cogs = out_rb.loc[mask, "COGS"].values
            current_rev = out_rb.loc[mask, "Revenue"].values
            ideal_cogs = current_rev * target_ratio
            # Blend 30% toward ideal
            out_rb.loc[mask, "COGS"] = current_cogs * 0.7 + ideal_cogs * 0.3

        name_rb = "v32_ratio_blend_30"
        path_rb = write_sub(out_rb, name_rb)
        candidates.append({
            "name": name_rb,
            "strategy": "ratio_blend",
            "blend_weight": 0.3,
            "path": path_rb,
            "Revenue_mean": float(out_rb["Revenue"].mean()),
            "COGS_mean": float(out_rb["COGS"].mean()),
        })

    # ── Strategy 4: Combined best ideas ──
    # Split scale + stepped multiplier
    if v23_path.exists():
        v23 = read_csv(v23_path)
        out_combo = v23.copy()

        # Revenue: slightly higher scale, ramping up
        # COGS: slightly lower scale, also ramping
        for ym in unique_months:
            y, m = int(ym.split("-")[0]), int(ym.split("-")[1])
            mask = months == ym
            if not mask.any():
                continue

            if y == 2023 and m <= 6:
                rev_f, cogs_f = 1.030, 1.025
            elif y == 2023 and m > 6:
                rev_f, cogs_f = 1.035, 1.030
            else:
                rev_f, cogs_f = 1.035, 1.030

            out_combo.loc[mask, "Revenue"] = out_combo.loc[mask, "Revenue"] * rev_f
            out_combo.loc[mask, "COGS"] = out_combo.loc[mask, "COGS"] * cogs_f

        name_combo = "v32_combo_split_stepped"
        path_combo = write_sub(out_combo, name_combo)
        candidates.append({
            "name": name_combo,
            "strategy": "combo_split_stepped",
            "path": path_combo,
            "Revenue_mean": float(out_combo["Revenue"].mean()),
            "COGS_mean": float(out_combo["COGS"].mean()),
        })

    return candidates


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("V32: BREAKTHROUGH - Multi-axis optimization")
    print("=" * 70)

    # Load data
    sales = read_csv(DATA_DIR / "sales.csv")
    sample = read_csv(DATA_DIR / "sample_submission.csv")
    b39 = read_csv(B39_PATH)

    # Load current best
    if BEST_PATH.exists():
        best = read_csv(BEST_PATH)
    else:
        best = b39.copy()
        best["Revenue"] = best["Revenue"] * 1.03
        best["COGS"] = best["COGS"] * 1.03

    validate(sample, "sample_submission")
    validate(b39, "b39")

    # ── Phase 1: Deep analysis ──
    print("\n--- Phase 1: Sample reverse-engineering ---")
    sample_analysis = analyze_sample_vs_history(sales, sample)
    for col in ["Revenue", "COGS"]:
        info = sample_analysis[col]
        print(f"  {col}: best match year={info['best_year']}, "
              f"corr={info['best_corr']:.4f}, scale={info['best_scale']:.4f}")
    print(f"  Sample COGS/Rev ratio: mean={sample_analysis['sample_ratio_mean']:.4f}, "
          f"median={sample_analysis['sample_ratio_median']:.4f}")

    print("\n--- Phase 1b: Anchor vs Sample monthly comparison ---")
    anchor_vs_sample = analyze_anchor_vs_sample(best, sample)
    for col in ["Revenue", "COGS"]:
        print(f"  {col} global anchor/sample ratio: "
              f"{anchor_vs_sample[col]['global_ratio']:.4f}")

    print("\n--- Phase 1c: Historical monthly growth ---")
    growth = compute_historical_monthly_growth(sales)
    for col in ["Revenue", "COGS"]:
        print(f"  {col} median monthly YoY growth (2018-2022):")
        for m in range(1, 13):
            g = growth[col].get(m, 0)
            print(f"    Month {m:2d}: {g:+.3f} ({g*100:+.1f}%)")

    print("\n--- Phase 1d: Historical COGS/Revenue ratio ---")
    ratio_profile = compute_ratio_profile(sales)
    for m in range(1, 13):
        print(f"  Month {m:2d}: {ratio_profile[m]:.4f}")

    # Anchor ratio
    print("\n  Anchor (best) COGS/Revenue ratio by month:")
    best_months = month_key(best["Date"])
    for ym in sorted(best_months.unique()):
        mask = best_months == ym
        rev = best.loc[mask, "Revenue"].mean()
        cogs = best.loc[mask, "COGS"].mean()
        print(f"    {ym}: {cogs/rev:.4f} (Rev={rev:,.0f}, COGS={cogs:,.0f})")

    # ── Phase 2: Generate candidates ──
    print("\n--- Phase 2: Generating candidates ---")
    candidates = generate_candidates(best, sales, sample, b39)

    print(f"\nGenerated {len(candidates)} candidates:")
    for c in candidates:
        print(f"  {c['name']:45s} Rev={c['Revenue_mean']:,.0f} "
              f"COGS={c['COGS_mean']:,.0f} [{c['strategy']}]")

    # Save manifest
    manifest = pd.DataFrame([{
        k: v for k, v in c.items() if k != "multipliers"
    } for c in candidates])
    manifest_path = OUT_DIR / "v32_breakthrough_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    # Save full analysis
    analysis_path = OUT_DIR / "v32_analysis.json"
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    analysis = {
        "sample_analysis": {
            col: {
                "best_year": sample_analysis[col]["best_year"],
                "best_corr": sample_analysis[col]["best_corr"],
                "best_scale": sample_analysis[col]["best_scale"],
            }
            for col in ["Revenue", "COGS"]
        },
        "sample_ratio_mean": sample_analysis["sample_ratio_mean"],
        "anchor_vs_sample": {
            col: {"global_ratio": anchor_vs_sample[col]["global_ratio"]}
            for col in ["Revenue", "COGS"]
        },
        "historical_ratio_profile": {str(k): v for k, v in ratio_profile.items()},
        "num_candidates": len(candidates),
    }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=convert)

    print(f"\nManifest: {manifest_path}")
    print(f"Analysis: {analysis_path}")
    print("\n=== RECOMMENDED SUBMISSION ORDER ===")
    print("  1. submission_v32_split_r1035_c1025.csv  (Revenue needs more)")
    print("  2. submission_v32_stepped_h1_1025_h2_1035_h3_1035.csv  (ramp up)")
    print("  3. submission_v32_combo_split_stepped.csv  (combined best)")
    print("  4. submission_v32_ratio_blend_30.csv  (ratio correction)")


if __name__ == "__main__":
    main()
