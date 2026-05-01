"""
V41: Targeted intra-month correction after monthly rebalance.

Best known LB: output/submission_v37_rebal_s10250.csv = 675,314.

Failed directions:
  - weekly rebalance: too noisy
  - full sample daily shape: worse than b39/v23 daily shape

Observation:
  After v37 monthly rebalance, sample's scaled daily shape mainly asks for
  higher month-start/month-end days and slightly higher weekends. This script
  applies only that targeted log correction, then restores each monthly mean.

Only two candidates are created:
  1. edge-only correction
  2. edge + weekend correction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
OUT = ROOT / "output"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def month_key(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").astype(str)


def sample_scaled_to_base_month(base: pd.DataFrame, sample: pd.DataFrame, col: str) -> np.ndarray:
    out = np.empty(len(base), dtype=float)
    months = month_key(base["Date"])
    sample_months = month_key(sample["Date"])
    for ym in sorted(months.unique()):
        bm = months == ym
        sm = sample_months == ym
        b_mean = base.loc[bm, col].mean()
        s_mean = sample.loc[sm, col].mean()
        if s_mean > 0:
            out[np.where(bm)[0]] = sample.loc[sm, col].to_numpy(dtype=float) * (b_mean / s_mean)
        else:
            out[np.where(bm)[0]] = base.loc[bm, col].to_numpy(dtype=float)
    return out


def restore_month_mean(dates: pd.Series, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    months = month_key(dates)
    out = np.asarray(values, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    for ym in sorted(months.unique()):
        mask = (months == ym).to_numpy()
        cur = out[mask].mean()
        tgt = target[mask].mean()
        if cur > 0:
            out[mask] *= tgt / cur
    return out


def apply_targeted(
    base: pd.DataFrame,
    sample: pd.DataFrame,
    edge_weight: float,
    weekend_weight: float,
) -> pd.DataFrame:
    out = base.copy()
    dates = out["Date"]
    edge = (dates.dt.is_month_start | dates.dt.is_month_end).to_numpy()
    weekend = (dates.dt.dayofweek >= 5).to_numpy()

    for col in ["Revenue", "COGS"]:
        base_vals = out[col].to_numpy(dtype=float)
        sample_scaled = sample_scaled_to_base_month(out, sample, col)
        log_signal = np.log(np.clip(sample_scaled, 1.0, None) / np.clip(base_vals, 1.0, None))

        weights = np.zeros(len(out), dtype=float)
        weights[edge] += edge_weight
        weights[weekend & ~edge] += weekend_weight
        # Clip the sample pull so a few odd days cannot dominate.
        adjusted = base_vals * np.exp(weights * np.clip(log_signal, -0.25, 0.25))
        adjusted = restore_month_mean(dates, adjusted, base_vals)
        out[col] = np.clip(adjusted, 0, None)
    return out


def write(df: pd.DataFrame, name: str) -> Path:
    out = df.copy()
    for col in ["Revenue", "COGS"]:
        out[col] = out[col].round(2)
        if (out[col] < 0).any() or not np.isfinite(out[col]).all():
            raise ValueError(f"{name} {col} invalid")
    if len(out) != 548:
        raise ValueError(f"{name} row count {len(out)}")
    formatted = out.copy()
    formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
    path = OUT / f"submission_{name}.csv"
    formatted.to_csv(path, index=False)
    return path


def main() -> None:
    base_path = OUT / "submission_v37_rebal_s10250.csv"
    base = read_csv(base_path)
    sample = read_csv(DATA / "sample_submission.csv")

    candidates = [
        ("v41_edge_only_w35", 0.35, 0.00),
        ("v41_edge_w30_weekend_w12", 0.30, 0.12),
    ]
    rows = []
    print("Generating V41 targeted candidates...")
    for name, edge_w, weekend_w in candidates:
        cand = apply_targeted(base, sample, edge_w, weekend_w)
        path = write(cand, name)
        rows.append(
            {
                "candidate": name,
                "path": str(path),
                "edge_weight": edge_w,
                "weekend_weight": weekend_w,
                "Revenue_mean": float(cand["Revenue"].mean()),
                "COGS_mean": float(cand["COGS"].mean()),
                "Revenue_mean_abs_move_vs_v37": float(np.abs(cand["Revenue"] - base["Revenue"]).mean()),
                "COGS_mean_abs_move_vs_v37": float(np.abs(cand["COGS"] - base["COGS"]).mean()),
            }
        )
        print(f"  {name}: edge={edge_w:.2f} weekend={weekend_w:.2f} -> {path}")

    manifest = pd.DataFrame(rows)
    manifest_path = OUT / "v41_targeted_edge_weekend_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
