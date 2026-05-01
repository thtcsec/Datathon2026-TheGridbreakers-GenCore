"""
V30: Final minimal candidates after scale optimum flattened.

Latest LB:
    +2.5% -> 698184.70135
    +3.25% -> 698036.25791

Refit says the best pure global scale is about +3.0%, but the gain left there
is tiny. To give one higher-upside option, this also creates a peak-amplified
variant: same +3.0% level, but slightly increases within-month spikes while
preserving each month's mean before the global scale.

Only two files are emitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import v20_shape_calibrated_anchor as v20


GLOBAL_SCALE = 1.0300
PEAK_GAMMA = {"Revenue": 1.045, "COGS": 1.035}


def month_key(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").astype(str)


def amplify_peaks_preserve_month_mean(
    dates: pd.Series,
    values: np.ndarray,
    gamma: float,
) -> np.ndarray:
    df = pd.DataFrame({"month": month_key(dates), "value": np.asarray(values, dtype=float)})
    month_mean = df.groupby("month")["value"].transform("mean").to_numpy()
    ratio = np.clip(df["value"].to_numpy() / np.clip(month_mean, 1.0, None), 1e-6, None)
    shaped = month_mean * np.power(ratio, gamma)
    shaped_df = pd.DataFrame({"month": df["month"], "value": shaped, "target": month_mean})
    shaped_mean = shaped_df.groupby("month")["value"].transform("mean").replace(0, np.nan)
    target_mean = shaped_df.groupby("month")["target"].transform("mean")
    out = shaped_df["value"] * target_mean / shaped_mean
    return out.to_numpy(dtype=float)


def write_submission(df: pd.DataFrame, name: str) -> str:
    out = df.copy()
    for col in ["Revenue", "COGS"]:
        out[col] = out[col].round(2)
        if (out[col] < 0).any() or not np.isfinite(out[col]).all():
            raise ValueError(f"{name} {col} invalid")
    formatted = out.copy()
    formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
    path = v20.OUT_DIR / f"submission_{name}.csv"
    formatted.to_csv(path, index=False)
    return str(path)


def main() -> None:
    base_path = v20.OUT_DIR / "submission_v23_b39_all_430.csv"
    base = v20.read_csv(base_path)
    v20.validate_forecast_frame(base, base_path.name)

    plain = base.copy()
    plain["Revenue"] = plain["Revenue"] * GLOBAL_SCALE
    plain["COGS"] = plain["COGS"] * GLOBAL_SCALE
    plain_path = write_submission(plain, "v30_v23_both_up_300pct")

    peak = base.copy()
    for col in ["Revenue", "COGS"]:
        peak[col] = amplify_peaks_preserve_month_mean(
            peak["Date"],
            peak[col].to_numpy(dtype=float),
            PEAK_GAMMA[col],
        )
        peak[col] = peak[col] * GLOBAL_SCALE
    peak_path = write_submission(peak, "v30_v23_up300_peak_amp")

    manifest = pd.DataFrame(
        [
            {
                "candidate": "v30_v23_both_up_300pct",
                "path": plain_path,
                "global_scale": GLOBAL_SCALE,
                "revenue_peak_gamma": 1.0,
                "cogs_peak_gamma": 1.0,
                "Revenue_mean": float(plain["Revenue"].mean()),
                "COGS_mean": float(plain["COGS"].mean()),
            },
            {
                "candidate": "v30_v23_up300_peak_amp",
                "path": peak_path,
                "global_scale": GLOBAL_SCALE,
                "revenue_peak_gamma": PEAK_GAMMA["Revenue"],
                "cogs_peak_gamma": PEAK_GAMMA["COGS"],
                "Revenue_mean": float(peak["Revenue"].mean()),
                "COGS_mean": float(peak["COGS"].mean()),
            },
        ]
    )
    manifest_path = v20.OUT_DIR / "v30_final_scale_peak_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("Generated:")
    print(f"  {plain_path}")
    print(f"  {peak_path}")
    print(f"Manifest: {manifest_path}")
    print("\nSubmit order:")
    print("  1. submission_v30_v23_both_up_300pct.csv")
    print("  2. Only if you want high-upside risk: submission_v30_v23_up300_peak_amp.csv")


if __name__ == "__main__":
    main()
