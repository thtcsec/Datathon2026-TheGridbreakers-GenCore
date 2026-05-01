"""
V22: Repeat the v20/v21 shape correction with b39 as the only anchor.

Why this exists:
    v20 used the median of b39 and b45. If b45 is weaker on the leaderboard,
    using it even at 50% may inject avoidable noise. This branch keeps the
    original best known b39 level/shape as the sole anchor, then applies the
    same LB-validated daily-shape correction and alpha extrapolation.

Outputs:
    output/submission_v22_b39_*.csv
    output/v22_b39_candidate_manifest.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20
from v21_anchor_extrapolation import (
    candidate_stats,
    extrapolate_column,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    revenue_alpha: float
    cogs_alpha: float
    risk: str
    note: str


CANDIDATES = [
    Candidate("v22_b39_alpha_all_100", 1.00, 1.00, "low", "b39-only version of v20"),
    Candidate("v22_b39_alpha_all_160", 1.60, 1.60, "medium", "balanced extrapolation"),
    Candidate("v22_b39_alpha_all_220", 2.20, 2.20, "medium-high", "strong all-column extrapolation"),
    Candidate("v22_b39_alpha_all_300", 3.00, 3.00, "high", "large all-column extrapolation"),
    Candidate("v22_b39_rev220_cogs120", 2.20, 1.20, "medium", "Revenue-forward, COGS conservative"),
    Candidate("v22_b39_rev300_cogs140", 3.00, 1.40, "high", "aggressive Revenue, modest COGS"),
    Candidate("v22_b39_rev300_cogs000", 3.00, 0.00, "high", "Revenue-only stress test"),
    Candidate("v22_b39_rev400_cogs160", 4.00, 1.60, "very-high", "high-upside Revenue extrapolation"),
]


def b39_anchor(sample: pd.DataFrame) -> pd.DataFrame:
    path = v20.ROOT / "submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv"
    anchor = v20.read_csv(path)
    v20.validate_forecast_frame(anchor, path.name)
    anchor = sample[["Date"]].merge(anchor, on="Date", how="left", validate="one_to_one")
    if anchor[["Revenue", "COGS"]].isna().any().any():
        raise ValueError("b39 anchor does not align with sample_submission dates")
    return anchor


def build_base(
    sales: pd.DataFrame,
    sample: pd.DataFrame,
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    base = sample[["Date"]].copy()
    for col in ["Revenue", "COGS"]:
        lag_values = v20.historical_lag_prior(sales, sample["Date"], col)
        preds, _ = v20.calibrate_one_column(
            dates=sample["Date"],
            sample_values=sample[col].to_numpy(dtype=float),
            anchor_values=anchor[col].to_numpy(dtype=float),
            lag_values=lag_values,
            col=col,
        )
        base[col] = preds
    return base


def main() -> None:
    sales = v20.read_csv(v20.DATA_DIR / "sales.csv")
    sample = v20.read_csv(v20.DATA_DIR / "sample_submission.csv")
    v20.validate_forecast_frame(sample, "sample_submission.csv")
    anchor = b39_anchor(sample)
    base = build_base(sales, sample, anchor)

    rows = []
    print("Generating V22 b39-only anchor candidates...")
    for cfg in CANDIDATES:
        out = sample[["Date"]].copy()
        out["Revenue"] = extrapolate_column(
            sample["Date"],
            anchor["Revenue"].to_numpy(dtype=float),
            base["Revenue"].to_numpy(dtype=float),
            cfg.revenue_alpha,
            "Revenue",
        )
        out["COGS"] = extrapolate_column(
            sample["Date"],
            anchor["COGS"].to_numpy(dtype=float),
            base["COGS"].to_numpy(dtype=float),
            cfg.cogs_alpha,
            "COGS",
        )
        for col in ["Revenue", "COGS"]:
            if not np.isfinite(out[col]).all():
                raise ValueError(f"{cfg.name} {col} contains non-finite values")
            if (out[col] < 0).any():
                raise ValueError(f"{cfg.name} {col} contains negative values")
            out[col] = out[col].round(2)

        formatted = out.copy()
        formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
        out_path = v20.OUT_DIR / f"submission_{cfg.name}.csv"
        formatted.to_csv(out_path, index=False)

        # Reuse the manifest stats helper. It only needs an object with these
        # attributes, so the local Candidate dataclass is intentionally aligned.
        row = candidate_stats(cfg.name, out, anchor, base, cfg)  # type: ignore[arg-type]
        rows.append(row)
        print(
            f"  {cfg.name:27s} risk={cfg.risk:11s} "
            f"RevMove={row['Revenue_mean_abs_move_vs_anchor']:,.0f} "
            f"COGSMove={row['COGS_mean_abs_move_vs_anchor']:,.0f}"
        )

    manifest = pd.DataFrame(rows)
    manifest_path = v20.OUT_DIR / "v22_b39_candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    print("\nRecommended b39-only submissions:")
    print("  1. submission_v22_b39_alpha_all_100.csv")
    print("  2. submission_v22_b39_alpha_all_220.csv")
    print("  3. submission_v22_b39_rev220_cogs120.csv")
    print("  4. submission_v22_b39_alpha_all_300.csv")


if __name__ == "__main__":
    main()
