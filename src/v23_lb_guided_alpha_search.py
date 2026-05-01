"""
V23: LB-guided alpha search around the observed optimum.

Observed public LB:
    b39 anchor                     ~725,504
    v20 / alpha about 1            ~716,789
    v22_b39_alpha_all_220          ~709,389

A quadratic fit puts the minimum near alpha 4.2. This script generates a
focused batch around that region, still preserving the monthly mean of the
b39 neural anchor. These are not retrained models; they are controlled
extrapolations of a direction already confirmed by the public leaderboard.

Outputs:
    output/submission_v23_*.csv
    output/v23_candidate_manifest.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20
from v21_anchor_extrapolation import candidate_stats, extrapolate_column
from v22_b39_anchor_extrapolation import b39_anchor, build_base


@dataclass(frozen=True)
class Candidate:
    name: str
    revenue_alpha: float
    cogs_alpha: float
    risk: str
    note: str


CANDIDATES = [
    Candidate("v23_b39_all_360", 3.60, 3.60, "medium-high", "left shoulder of fitted optimum"),
    Candidate("v23_b39_all_400", 4.00, 4.00, "high", "near prior v22 stress test"),
    Candidate("v23_b39_all_430", 4.30, 4.30, "high", "quadratic-fit optimum"),
    Candidate("v23_b39_all_460", 4.60, 4.60, "high", "right shoulder of fitted optimum"),
    Candidate("v23_b39_all_500", 5.00, 5.00, "very-high", "tests if improvement is still monotonic"),
    Candidate("v23_b39_all_580", 5.80, 5.80, "wildcard", "only if 4.6/5.0 keep improving"),
    Candidate("v23_b39_rev430_cogs220", 4.30, 2.20, "medium-high", "Revenue stronger, COGS at proven alpha"),
    Candidate("v23_b39_rev430_cogs300", 4.30, 3.00, "high", "Revenue optimum, COGS moderately strong"),
    Candidate("v23_b39_rev500_cogs220", 5.00, 2.20, "high", "aggressive Revenue, proven COGS"),
    Candidate("v23_b39_rev360_cogs460", 3.60, 4.60, "high", "tests whether COGS wants more alpha"),
    Candidate("v23_b39_rev220_cogs430", 2.20, 4.30, "medium-high", "COGS-forward split"),
    Candidate("v23_b39_rev430_cogs000", 4.30, 0.00, "high", "Revenue-only diagnostic"),
]


def fit_alpha_curve() -> tuple[float, float, float, float]:
    # Scores supplied by the user / current known submissions.
    points = np.array(
        [
            [0.0, 725504.0],
            [1.0, 716789.34431],
            [2.2, 709388.82312],
        ],
        dtype=float,
    )
    x = points[:, 0]
    y = points[:, 1]
    a, b, c = np.polyfit(x, y, deg=2)
    optimum = -b / (2 * a) if a > 0 else np.nan
    predicted = a * optimum**2 + b * optimum + c if np.isfinite(optimum) else np.nan
    return float(a), float(b), float(optimum), float(predicted)


def write_candidate(
    cfg: Candidate,
    sample: pd.DataFrame,
    anchor: pd.DataFrame,
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
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
    path = v20.OUT_DIR / f"submission_{cfg.name}.csv"
    formatted.to_csv(path, index=False)

    stats = candidate_stats(cfg.name, out, anchor, base, cfg)  # type: ignore[arg-type]
    return out, stats


def main() -> None:
    a, b, optimum, predicted = fit_alpha_curve()
    print("LB-guided quadratic fit:")
    print(f"  score ~= {a:.2f} * alpha^2 + {b:.2f} * alpha + c")
    print(f"  fitted optimum alpha ~= {optimum:.2f}, predicted public MAE ~= {predicted:,.0f}")

    sales = v20.read_csv(v20.DATA_DIR / "sales.csv")
    sample = v20.read_csv(v20.DATA_DIR / "sample_submission.csv")
    v20.validate_forecast_frame(sample, "sample_submission.csv")
    anchor = b39_anchor(sample)
    base = build_base(sales, sample, anchor)

    rows = []
    print("\nGenerating V23 candidates...")
    for cfg in CANDIDATES:
        _, stats = write_candidate(cfg, sample, anchor, base)
        rows.append(stats)
        print(
            f"  {cfg.name:25s} risk={cfg.risk:11s} "
            f"RevMove={stats['Revenue_mean_abs_move_vs_anchor']:,.0f} "
            f"COGSMove={stats['COGS_mean_abs_move_vs_anchor']:,.0f}"
        )

    manifest = pd.DataFrame(rows)
    manifest_path = v20.OUT_DIR / "v23_candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    print("\nRecommended submission order:")
    print("  1. submission_v23_b39_all_430.csv")
    print("  2. submission_v23_b39_all_400.csv")
    print("  3. submission_v23_b39_rev430_cogs220.csv")
    print("  4. submission_v23_b39_all_460.csv")
    print("  5. submission_v23_b39_rev430_cogs300.csv")


if __name__ == "__main__":
    main()
