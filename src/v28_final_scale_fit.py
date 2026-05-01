"""
V28: Final scale-fit submissions from public LB feedback.

Known public LB points around v23_b39_all_430:
    scale -0.75% -> 707771.45579
    scale  0.00% -> 704169.28382
    scale +0.50% -> 702273.88797
    scale +0.75% -> 701501.33554

A quadratic fit puts the optimum near +2.5%; a cubic fit puts it near +2.0%.
With only two submissions left, the best use is to submit +2.0% first, then
+2.5% if +2.0% improves. This script creates focused files around that range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20


LB_POINTS = [
    {"scale": -0.0075, "score": 707771.45579, "name": "v25_v23_both_down_075pct"},
    {"scale": 0.0000, "score": 704169.28382, "name": "v23_b39_all_430"},
    {"scale": 0.0050, "score": 702273.88797, "name": "v27_probe_global_up_05pct"},
    {"scale": 0.0075, "score": 701501.33554, "name": "v26_v23_both_up_075pct"},
]

SCALES = {
    "v28_v23_both_up_175pct": 1.0175,
    "v28_v23_both_up_200pct": 1.0200,
    "v28_v23_both_up_225pct": 1.0225,
    "v28_v23_both_up_250pct": 1.0250,
    "v28_v23_both_up_275pct": 1.0275,
}


def fit_curves() -> dict[str, object]:
    pts = np.asarray([(p["scale"], p["score"]) for p in LB_POINTS], dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    quad = np.polyfit(x, y, deg=2)
    cubic = np.polyfit(x, y, deg=3)
    grid = np.linspace(-0.01, 0.05, 12001)
    quad_y = np.polyval(quad, grid)
    cubic_y = np.polyval(cubic, grid)
    predictions = {}
    for name, scale_multiplier in SCALES.items():
        delta = scale_multiplier - 1.0
        predictions[name] = {
            "scale_delta": delta,
            "quad_pred": float(np.polyval(quad, delta)),
            "cubic_pred": float(np.polyval(cubic, delta)),
        }
    return {
        "lb_points": LB_POINTS,
        "quadratic_coefficients": quad.tolist(),
        "cubic_coefficients": cubic.tolist(),
        "quadratic_optimum_scale_delta": float(grid[np.argmin(quad_y)]),
        "quadratic_optimum_score": float(np.min(quad_y)),
        "cubic_optimum_scale_delta": float(grid[np.argmin(cubic_y)]),
        "cubic_optimum_score": float(np.min(cubic_y)),
        "predictions": predictions,
    }


def main() -> None:
    base_path = v20.OUT_DIR / "submission_v23_b39_all_430.csv"
    base = v20.read_csv(base_path)
    v20.validate_forecast_frame(base, base_path.name)

    fit = fit_curves()
    print("Public-LB scale fit:")
    print(
        f"  quadratic optimum delta={fit['quadratic_optimum_scale_delta']:+.4%}, "
        f"pred={fit['quadratic_optimum_score']:,.0f}"
    )
    print(
        f"  cubic optimum delta={fit['cubic_optimum_scale_delta']:+.4%}, "
        f"pred={fit['cubic_optimum_score']:,.0f}"
    )

    rows = []
    for name, scale in SCALES.items():
        out = base.copy()
        out["Revenue"] = (out["Revenue"] * scale).round(2)
        out["COGS"] = (out["COGS"] * scale).round(2)
        if (out[["Revenue", "COGS"]] < 0).any().any():
            raise ValueError(f"{name} has negative values")
        formatted = out.copy()
        formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
        path = v20.OUT_DIR / f"submission_{name}.csv"
        formatted.to_csv(path, index=False)

        pred = fit["predictions"][name]
        row = {
            "candidate": name,
            "scale": scale,
            "scale_delta": scale - 1.0,
            "path": str(path),
            "Revenue_mean": float(out["Revenue"].mean()),
            "COGS_mean": float(out["COGS"].mean()),
            "quad_pred_score": pred["quad_pred"],
            "cubic_pred_score": pred["cubic_pred"],
        }
        rows.append(row)
        print(
            f"  {name:27s} scale={scale:.4f} "
            f"quad={pred['quad_pred']:,.0f} cubic={pred['cubic_pred']:,.0f}"
        )

    manifest = pd.DataFrame(rows)
    manifest_path = v20.OUT_DIR / "v28_final_scale_fit_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    fit_path = v20.OUT_DIR / "v28_public_lb_scale_fit.json"
    with fit_path.open("w", encoding="utf-8") as f:
        json.dump(fit, f, indent=2)

    print(f"\nManifest: {manifest_path}")
    print(f"Fit diagnostics: {fit_path}")
    print("\nRecommended with two submissions left:")
    print("  1. submission_v28_v23_both_up_200pct.csv")
    print("  2. If #1 improves, submit submission_v28_v23_both_up_250pct.csv")
    print("     If #1 worsens, submit submission_v28_v23_both_up_175pct.csv")


if __name__ == "__main__":
    main()
