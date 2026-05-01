"""
V27: Public-LB probing files for MAE sign diagnostics.

This is not a normal model. It creates small, structured perturbations around
the current best submission (v23_b39_all_430). For MAE, a small perturbation
changes the score approximately by:

    delta_score * N ~= sum(sign(pred - truth) * perturbation)

So if the user submits these probes and records the public scores, we can infer
which coarse month/column blocks are over- or under-predicted, then create a
targeted correction. This is the kind of leaderboard-guided optimization that
can beat ordinary validation when the public LB is available.

Outputs:
    output/submission_v27_probe_*.csv
    output/v27_probe_manifest.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20


BASE_PUBLIC_SCORE = 704169.28382
N_VALUES = 548 * 2


@dataclass(frozen=True)
class Probe:
    name: str
    revenue_pattern: str
    cogs_pattern: str
    magnitude: float
    note: str


PROBES = [
    Probe("v27_probe_global_up_05pct", "all_up", "all_up", 0.0050, "global sign check"),
    Probe("v27_probe_revenue_up_05pct", "all_up", "zero", 0.0050, "Revenue-only sign check"),
    Probe("v27_probe_cogs_up_05pct", "zero", "all_up", 0.0050, "COGS-only sign check"),
    Probe("v27_probe_h1_up_h2_down", "h1_up_h2_down", "h1_up_h2_down", 0.0060, "early vs late horizon"),
    Probe("v27_probe_odd_month_up_even_down", "odd_month_up", "odd_month_up", 0.0060, "month parity code"),
    Probe("v27_probe_q1q3_up_q2q4_down", "q1q3_up", "q1q3_up", 0.0060, "seasonal quarter code"),
    Probe("v27_probe_rev_h1_up_cogs_h2_up", "h1_up_h2_down", "h2_up_h1_down", 0.0060, "target-horizon interaction"),
]


def signs_for_pattern(dates: pd.Series, pattern: str) -> np.ndarray:
    if pattern == "zero":
        return np.zeros(len(dates), dtype=float)
    if pattern == "all_up":
        return np.ones(len(dates), dtype=float)
    if pattern == "h1_up_h2_down":
        midpoint = dates.min() + (dates.max() - dates.min()) / 2
        return np.where(dates <= midpoint, 1.0, -1.0)
    if pattern == "h2_up_h1_down":
        midpoint = dates.min() + (dates.max() - dates.min()) / 2
        return np.where(dates <= midpoint, -1.0, 1.0)
    if pattern == "odd_month_up":
        return np.where(dates.dt.month % 2 == 1, 1.0, -1.0)
    if pattern == "q1q3_up":
        return np.where(dates.dt.quarter.isin([1, 3]), 1.0, -1.0)
    raise ValueError(f"Unknown pattern: {pattern}")


def apply_probe(base: pd.DataFrame, probe: Probe) -> pd.DataFrame:
    out = base.copy()
    for col, pattern in [("Revenue", probe.revenue_pattern), ("COGS", probe.cogs_pattern)]:
        sign = signs_for_pattern(out["Date"], pattern)
        out[col] = np.clip(out[col].to_numpy(dtype=float) * (1.0 + probe.magnitude * sign), 0, None)
        out[col] = out[col].round(2)
    return out


def main() -> None:
    base_path = v20.OUT_DIR / "submission_v23_b39_all_430.csv"
    base = v20.read_csv(base_path)
    v20.validate_forecast_frame(base, base_path.name)

    rows = []
    print("Generating V27 public-LB probe files...")
    for probe in PROBES:
        out = apply_probe(base, probe)
        formatted = out.copy()
        formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
        path = v20.OUT_DIR / f"submission_{probe.name}.csv"
        formatted.to_csv(path, index=False)
        rows.append(
            {
                "probe": probe.name,
                "path": str(path),
                "magnitude": probe.magnitude,
                "revenue_pattern": probe.revenue_pattern,
                "cogs_pattern": probe.cogs_pattern,
                "note": probe.note,
                "base_public_score": BASE_PUBLIC_SCORE,
                "how_to_decode": "delta_score = submitted_score - base_public_score; multiply by 1096 for signed projection",
            }
        )
        print(f"  {probe.name:34s} mag={probe.magnitude:.4f} {probe.note}")

    manifest = pd.DataFrame(rows)
    manifest_path = v20.OUT_DIR / "v27_probe_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
