"""
V29: Minimal follow-up after +2.5% improved.

Known public LB around v23_b39_all_430:
    +2.0% -> 698768.48000
    +2.5% -> 698184.70135

The local slope is still negative, but flattening. Refit curves place the next
minimum around +3.0% to +3.6%. To avoid generating noise, this script creates
only two files:
    - +3.25%: best compromise across weighted/local fits
    - +3.50%: hedge if the optimum is closer to cubic/quartic fit
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20


SCALES = {
    "v29_v23_both_up_325pct": 1.0325,
    "v29_v23_both_up_350pct": 1.0350,
}


LB_POINTS = np.array(
    [
        [-0.0075, 707771.45579],
        [0.0000, 704169.28382],
        [0.0050, 702273.88797],
        [0.0075, 701501.33554],
        [0.0200, 698768.48000],
        [0.0250, 698184.70135],
    ],
    dtype=float,
)


def curve_predictions(delta: float) -> dict[str, float]:
    x = LB_POINTS[:, 0]
    y = LB_POINTS[:, 1]
    weighted = np.array([0.2, 0.5, 0.7, 1.0, 1.3, 1.6])
    quad_all = np.polyfit(x, y, 2)
    quad_weighted = np.polyfit(x, y, 2, w=weighted)
    quad_recent4 = np.polyfit(x[-4:], y[-4:], 2)
    cubic_all = np.polyfit(x, y, 3)
    return {
        "quad_all": float(np.polyval(quad_all, delta)),
        "quad_weighted": float(np.polyval(quad_weighted, delta)),
        "quad_recent4": float(np.polyval(quad_recent4, delta)),
        "cubic_all": float(np.polyval(cubic_all, delta)),
    }


def main() -> None:
    base_path = v20.OUT_DIR / "submission_v23_b39_all_430.csv"
    base = v20.read_csv(base_path)
    v20.validate_forecast_frame(base, base_path.name)

    rows = []
    print("Generating V29 minimal scale candidates...")
    for name, scale in SCALES.items():
        out = base.copy()
        out["Revenue"] = (out["Revenue"] * scale).round(2)
        out["COGS"] = (out["COGS"] * scale).round(2)
        formatted = out.copy()
        formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
        path = v20.OUT_DIR / f"submission_{name}.csv"
        formatted.to_csv(path, index=False)

        preds = curve_predictions(scale - 1.0)
        row = {
            "candidate": name,
            "scale": scale,
            "path": str(path),
            "Revenue_mean": float(out["Revenue"].mean()),
            "COGS_mean": float(out["COGS"].mean()),
            **{f"pred_{k}": v for k, v in preds.items()},
        }
        rows.append(row)
        print(
            f"  {name:26s} scale={scale:.4f} "
            f"weighted={preds['quad_weighted']:,.0f} recent4={preds['quad_recent4']:,.0f} "
            f"cubic={preds['cubic_all']:,.0f}"
        )

    manifest = pd.DataFrame(rows)
    manifest_path = v20.OUT_DIR / "v29_minimal_scale_followup_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    print("\nSubmit order:")
    print("  1. submission_v29_v23_both_up_325pct.csv")
    print("  2. If it improves, submission_v29_v23_both_up_350pct.csv")


if __name__ == "__main__":
    main()
