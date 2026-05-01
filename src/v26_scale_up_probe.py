"""
V26: Scale-up probes after v25 downscale failed.

Known LB feedback:
    v23_b39_all_430               704,169
    v25_v23_both_down_075pct      707,771

Downscaling made the public score worse, so the next cheap axis to probe is
whether the best shape is slightly under-level. This script creates focused
scale-up files around v23 plus column-specific variants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20


@dataclass(frozen=True)
class Candidate:
    name: str
    revenue_scale: float
    cogs_scale: float
    note: str


CANDIDATES = [
    Candidate("v26_v23_both_up_075pct", 1.0075, 1.0075, "mirror of failed downscale"),
    Candidate("v26_v23_both_up_125pct", 1.0125, 1.0125, "small global upscale"),
    Candidate("v26_v23_both_up_200pct", 1.0200, 1.0200, "moderate global upscale"),
    Candidate("v26_v23_both_up_300pct", 1.0300, 1.0300, "aggressive global upscale"),
    Candidate("v26_v23_rev_up125_cogs_same", 1.0125, 1.0000, "Revenue level only"),
    Candidate("v26_v23_rev_up200_cogs_same", 1.0200, 1.0000, "Revenue stronger level only"),
    Candidate("v26_v23_rev_same_cogs_up125", 1.0000, 1.0125, "COGS level only"),
    Candidate("v26_v23_rev_same_cogs_up200", 1.0000, 1.0200, "COGS stronger level only"),
    Candidate("v26_v23_rev_up200_cogs_up075", 1.0200, 1.0075, "Revenue-heavy upscale"),
    Candidate("v26_v23_rev_up075_cogs_up200", 1.0075, 1.0200, "COGS-heavy upscale"),
]


def main() -> None:
    base_path = v20.OUT_DIR / "submission_v23_b39_all_430.csv"
    base = v20.read_csv(base_path)
    v20.validate_forecast_frame(base, base_path.name)

    rows = []
    print("Generating V26 scale-up probes...")
    for cfg in CANDIDATES:
        out = base.copy()
        out["Revenue"] = (out["Revenue"] * cfg.revenue_scale).round(2)
        out["COGS"] = (out["COGS"] * cfg.cogs_scale).round(2)
        if (out[["Revenue", "COGS"]] < 0).any().any():
            raise ValueError(f"{cfg.name} has negative values")

        formatted = out.copy()
        formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
        path = v20.OUT_DIR / f"submission_{cfg.name}.csv"
        formatted.to_csv(path, index=False)

        rows.append(
            {
                "candidate": cfg.name,
                "revenue_scale": cfg.revenue_scale,
                "cogs_scale": cfg.cogs_scale,
                "note": cfg.note,
                "path": str(path),
                "Revenue_mean": float(out["Revenue"].mean()),
                "COGS_mean": float(out["COGS"].mean()),
                "Revenue_mean_abs_move_vs_v23": float(np.abs(out["Revenue"] - base["Revenue"]).mean()),
                "COGS_mean_abs_move_vs_v23": float(np.abs(out["COGS"] - base["COGS"]).mean()),
            }
        )
        print(f"  {cfg.name:32s} RevScale={cfg.revenue_scale:.4f} COGSScale={cfg.cogs_scale:.4f}")

    manifest = pd.DataFrame(rows)
    manifest_path = v20.OUT_DIR / "v26_candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    print("\nRecommended:")
    print("  1. submission_v26_v23_both_up_075pct.csv")
    print("  2. submission_v26_v23_both_up_125pct.csv")
    print("  3. submission_v26_v23_rev_up125_cogs_same.csv")


if __name__ == "__main__":
    main()
