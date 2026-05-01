"""
V21: Extrapolate the LB-validated v20 correction direction.

Known signal:
    alpha=0  -> neural anchor around the old 725k LB
    alpha=1  -> v20 shape-calibrated anchor, reported LB 716,789

If the v20 direction is genuinely reducing public-test error, the fastest
controlled search is to keep the same direction and increase/decrease alpha.
Every candidate preserves the monthly level of the neural anchor, so these
files only change the within-month daily allocation.

Outputs:
    output/submission_v21_*.csv
    output/v21_candidate_manifest.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20


OUT_DIR = v20.OUT_DIR
OUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class Candidate:
    name: str
    revenue_alpha: float
    cogs_alpha: float
    risk: str
    note: str


CANDIDATES = [
    Candidate(
        "v21_alpha_all_125",
        1.25,
        1.25,
        "low",
        "small extrapolation from v20",
    ),
    Candidate(
        "v21_alpha_all_160",
        1.60,
        1.60,
        "medium",
        "balanced next step along the proven v20 direction",
    ),
    Candidate(
        "v21_alpha_all_220",
        2.20,
        2.20,
        "medium-high",
        "stronger all-column shape correction; best first aggressive try",
    ),
    Candidate(
        "v21_alpha_all_300",
        3.00,
        3.00,
        "high",
        "large extrapolation; useful if 2.2 improves",
    ),
    Candidate(
        "v21_alpha_all_400",
        4.00,
        4.00,
        "very-high",
        "stress test for monotonic improvement along v20 direction",
    ),
    Candidate(
        "v21_rev220_cogs120",
        2.20,
        1.20,
        "medium",
        "leans into Revenue shape while keeping COGS conservative",
    ),
    Candidate(
        "v21_rev300_cogs140",
        3.00,
        1.40,
        "high",
        "aggressive Revenue, modest COGS",
    ),
    Candidate(
        "v21_rev300_cogs000",
        3.00,
        0.00,
        "high",
        "isolates whether Revenue drove the v20 gain",
    ),
    Candidate(
        "v21_rev400_cogs160",
        4.00,
        1.60,
        "very-high",
        "high-upside Revenue extrapolation with limited COGS movement",
    ),
    Candidate(
        "v21_rev500_cogs200",
        5.00,
        2.00,
        "wildcard",
        "only submit if alpha 3-4 keeps improving",
    ),
]


def monthly_key(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").astype(str)


def preserve_monthly_anchor(
    dates: pd.Series,
    values: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    return v20.rescale_to_monthly_anchor(dates, values, anchor)


def extrapolate_column(
    dates: pd.Series,
    anchor_values: np.ndarray,
    v20_values: np.ndarray,
    alpha: float,
    col: str,
) -> np.ndarray:
    if alpha == 0:
        return anchor_values.copy()

    anchor_safe = np.clip(anchor_values.astype(float), 1.0, None)
    v20_safe = np.clip(v20_values.astype(float), 1.0, None)
    log_delta = np.log(v20_safe / anchor_safe)

    # V20 was capped around 5.5%-6.5%. For extrapolation, let the cap expand
    # sub-linearly so alpha 4 is bold but not a runaway submission.
    base_cap = np.log(1.065 if col == "Revenue" else 1.055)
    max_log_move = base_cap * (1.0 + 0.72 * max(0.0, alpha - 1.0))

    pred = anchor_safe * np.exp(alpha * log_delta)
    for _ in range(4):
        pred = preserve_monthly_anchor(dates, pred, anchor_values)
        pred = v20.cap_relative_move(pred, anchor_values, max_log_move)
    pred = preserve_monthly_anchor(dates, pred, anchor_values)
    return np.clip(pred, 0.0, None)


def build_v20_base(
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


def candidate_stats(
    name: str,
    candidate: pd.DataFrame,
    anchor: pd.DataFrame,
    v20_base: pd.DataFrame,
    cfg: Candidate,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate": name,
        "risk": cfg.risk,
        "revenue_alpha": cfg.revenue_alpha,
        "cogs_alpha": cfg.cogs_alpha,
        "note": cfg.note,
        "path": str(OUT_DIR / f"submission_{name}.csv"),
    }
    for col in ["Revenue", "COGS"]:
        move_anchor = np.abs(candidate[col].to_numpy() - anchor[col].to_numpy())
        move_v20 = np.abs(candidate[col].to_numpy() - v20_base[col].to_numpy())
        row[f"{col}_mean"] = float(candidate[col].mean())
        row[f"{col}_mean_abs_move_vs_anchor"] = float(move_anchor.mean())
        row[f"{col}_p95_move_vs_anchor"] = float(np.percentile(move_anchor, 95))
        row[f"{col}_max_move_vs_anchor"] = float(move_anchor.max())
        row[f"{col}_mean_abs_move_vs_v20"] = float(move_v20.mean())
        row[f"{col}_corr_vs_v20"] = float(
            np.corrcoef(candidate[col].to_numpy(), v20_base[col].to_numpy())[0, 1]
        )

        monthly_candidate = candidate.assign(ym=monthly_key(candidate["Date"])).groupby("ym")[col].mean()
        monthly_anchor = anchor.assign(ym=monthly_key(anchor["Date"])).groupby("ym")[col].mean()
        row[f"{col}_max_monthly_mean_drift"] = float(
            (monthly_candidate - monthly_anchor).abs().max()
        )
    return row


def main() -> None:
    sales = v20.read_csv(v20.DATA_DIR / "sales.csv")
    sample = v20.read_csv(v20.DATA_DIR / "sample_submission.csv")
    v20.validate_forecast_frame(sample, "sample_submission.csv")
    anchor, used_anchor_files = v20.build_anchor(sample)
    v20_base = build_v20_base(sales, sample, anchor)

    manifest_rows = []

    print(f"Using anchor files: {', '.join(used_anchor_files)}")
    print("Generating V21 extrapolation candidates...")

    for cfg in CANDIDATES:
        out = sample[["Date"]].copy()
        out["Revenue"] = extrapolate_column(
            sample["Date"],
            anchor["Revenue"].to_numpy(dtype=float),
            v20_base["Revenue"].to_numpy(dtype=float),
            cfg.revenue_alpha,
            "Revenue",
        )
        out["COGS"] = extrapolate_column(
            sample["Date"],
            anchor["COGS"].to_numpy(dtype=float),
            v20_base["COGS"].to_numpy(dtype=float),
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
        out_path = OUT_DIR / f"submission_{cfg.name}.csv"
        formatted.to_csv(out_path, index=False)

        stats = candidate_stats(cfg.name, out, anchor, v20_base, cfg)
        manifest_rows.append(stats)
        print(
            f"  {cfg.name:24s} risk={cfg.risk:11s} "
            f"RevMove={stats['Revenue_mean_abs_move_vs_anchor']:,.0f} "
            f"COGSMove={stats['COGS_mean_abs_move_vs_anchor']:,.0f}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "v21_candidate_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print(f"\nManifest: {manifest_path}")
    print("\nRecommended first submissions:")
    print("  1. submission_v21_alpha_all_220.csv")
    print("  2. submission_v21_rev220_cogs120.csv")
    print("  3. submission_v21_alpha_all_300.csv")
    print("  4. submission_v21_rev300_cogs140.csv")
    print("  5. submission_v21_alpha_all_160.csv as fallback if 2.2 overshoots")


if __name__ == "__main__":
    main()
