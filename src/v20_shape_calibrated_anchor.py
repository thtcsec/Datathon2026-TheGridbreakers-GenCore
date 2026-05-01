"""
V20: Shape-calibrated anchor forecast
=====================================

This script is intentionally conservative. The best known leaderboard file is
already much better than every newly trained model, so the goal is not to
replace it. Instead we keep its monthly level and use the organizer-provided
sample_submission plus historical lags only as weak shape priors.

Output:
    output/submission_v20_shape_calibrated_anchor.csv

Inputs:
    data/raw/sales.csv
    data/raw/sample_submission.csv
    submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv
    optional: submission_raw_stable_neural_blend_w735_w565_monthly_cogs_b45.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

FORECAST_START = pd.Timestamp("2023-01-01")
FORECAST_END = pd.Timestamp("2024-07-01")
EXPECTED_ROWS = 548

BEST_FILES = [
    ROOT / "submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv",
    ROOT / "submission_raw_stable_neural_blend_w735_w565_monthly_cogs_b45.csv",
]

TET_DATES = pd.to_datetime(
    [
        "2012-01-23",
        "2013-02-10",
        "2014-01-31",
        "2015-02-19",
        "2016-02-08",
        "2017-01-28",
        "2018-02-16",
        "2019-02-05",
        "2020-01-25",
        "2021-02-12",
        "2022-02-01",
        "2023-01-22",
        "2024-02-10",
    ]
)


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = {"Date", "Revenue", "COGS"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def validate_forecast_frame(df: pd.DataFrame, name: str) -> None:
    require_columns(df, Path(name))
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"{name} has {len(df)} rows, expected {EXPECTED_ROWS}")
    if df["Date"].min() != FORECAST_START or df["Date"].max() != FORECAST_END:
        raise ValueError(
            f"{name} date range is {df['Date'].min().date()}..{df['Date'].max().date()}, "
            f"expected {FORECAST_START.date()}..{FORECAST_END.date()}"
        )
    if df["Date"].duplicated().any():
        raise ValueError(f"{name} contains duplicated dates")


def iso_week_key(dates: pd.Series) -> pd.Series:
    iso = dates.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def month_key(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").astype(str)


def group_scaled_shape(
    dates: pd.Series,
    source_values: np.ndarray,
    anchor_values: np.ndarray,
    groups: pd.Series,
) -> np.ndarray:
    """Keep source's within-group shape but force each group to anchor mean."""
    frame = pd.DataFrame(
        {
            "group": groups.to_numpy(),
            "source": np.asarray(source_values, dtype=float),
            "anchor": np.asarray(anchor_values, dtype=float),
        }
    )
    source_mean = frame.groupby("group")["source"].transform("mean").replace(0, np.nan)
    anchor_mean = frame.groupby("group")["anchor"].transform("mean")
    scaled = frame["source"] * (anchor_mean / source_mean)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).fillna(frame["anchor"])
    return scaled.to_numpy(dtype=float)


def historical_lag_prior(
    sales: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    col: str,
) -> np.ndarray:
    """Robust same-season lag prior, available from train dates only."""
    series = sales.set_index("Date")[col].astype(float)
    offsets = [
        (365, 1.00),  # same calendar date, best match for sample_submission
        (364, 0.90),  # same weekday
        (371, 0.55),
        (357, 0.45),
        (728, 0.80),
        (729, 0.75),
        (735, 0.45),
        (721, 0.35),
        (1092, 0.35),
        (1093, 0.30),
    ]
    preds: list[float] = []
    fallback = float(series.tail(120).median())
    for dt in pd.to_datetime(list(dates)):
        vals = []
        weights = []
        for offset, weight in offsets:
            key = pd.Timestamp(dt) - pd.Timedelta(days=offset)
            if key in series.index:
                vals.append(float(series.loc[key]))
                weights.append(weight)
        if vals:
            preds.append(float(np.average(vals, weights=weights)))
        else:
            preds.append(fallback)
    return np.asarray(preds, dtype=float)


def signed_agreement_weight(a: np.ndarray, b: np.ndarray, base: float) -> np.ndarray:
    """Use full weight when two priors move the anchor in the same direction."""
    same_direction = np.sign(a) == np.sign(b)
    both_nonzero = (np.abs(a) > 1e-9) & (np.abs(b) > 1e-9)
    return np.where(same_direction & both_nonzero, base, base * 0.25)


def cap_relative_move(values: np.ndarray, anchor: np.ndarray, max_log_move: float) -> np.ndarray:
    log_move = np.log(np.clip(values, 1.0, None) / np.clip(anchor, 1.0, None))
    log_move = np.clip(log_move, -max_log_move, max_log_move)
    return anchor * np.exp(log_move)


def rescale_to_monthly_anchor(
    dates: pd.Series,
    values: np.ndarray,
    anchor: np.ndarray,
) -> np.ndarray:
    groups = month_key(dates)
    frame = pd.DataFrame(
        {
            "group": groups.to_numpy(),
            "value": np.asarray(values, dtype=float),
            "anchor": np.asarray(anchor, dtype=float),
        }
    )
    value_mean = frame.groupby("group")["value"].transform("mean").replace(0, np.nan)
    anchor_mean = frame.groupby("group")["anchor"].transform("mean")
    out = frame["value"] * (anchor_mean / value_mean)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(frame["anchor"])
    return out.to_numpy(dtype=float)


def tet_window_strength(dates: pd.Series) -> np.ndarray:
    """Small confidence bump around Tet, where sample prior has date-specific shape."""
    strength = np.zeros(len(dates), dtype=float)
    for tet in TET_DATES:
        delta = np.abs((dates - tet).dt.days.to_numpy())
        strength = np.maximum(strength, np.where(delta <= 21, 1.0, 0.0))
    return strength


def calibrate_one_column(
    dates: pd.Series,
    sample_values: np.ndarray,
    anchor_values: np.ndarray,
    lag_values: np.ndarray,
    col: str,
) -> tuple[np.ndarray, dict[str, float]]:
    week = iso_week_key(dates)
    month = month_key(dates)

    sample_week = group_scaled_shape(dates, sample_values, anchor_values, week)
    sample_month = group_scaled_shape(dates, sample_values, anchor_values, month)
    lag_week = group_scaled_shape(dates, lag_values, anchor_values, week)

    anchor_safe = np.clip(anchor_values, 1.0, None)
    sample_week_signal = np.log(np.clip(sample_week, 1.0, None) / anchor_safe)
    sample_month_signal = np.log(np.clip(sample_month, 1.0, None) / anchor_safe)
    lag_week_signal = np.log(np.clip(lag_week, 1.0, None) / anchor_safe)

    is_tet = tet_window_strength(dates)
    if col == "Revenue":
        w_sample_week = 0.090 + 0.025 * is_tet
        w_sample_month = 0.025
        w_lag_base = 0.030
        max_log_move = np.log(1.065)
    else:
        w_sample_week = 0.070 + 0.020 * is_tet
        w_sample_month = 0.020
        w_lag_base = 0.022
        max_log_move = np.log(1.055)

    w_lag = signed_agreement_weight(sample_week_signal, lag_week_signal, w_lag_base)
    total_signal = (
        w_sample_week * np.clip(sample_week_signal, -0.45, 0.45)
        + w_sample_month * np.clip(sample_month_signal, -0.35, 0.35)
        + w_lag * np.clip(lag_week_signal, -0.45, 0.45)
    )

    raw = anchor_values * np.exp(total_signal)

    # Keep the monthly level from the high-LB anchor. The model changes only
    # daily allocation inside each month.
    adjusted = raw.copy()
    for _ in range(3):
        adjusted = rescale_to_monthly_anchor(dates, adjusted, anchor_values)
        adjusted = cap_relative_move(adjusted, anchor_values, max_log_move)
    adjusted = rescale_to_monthly_anchor(dates, adjusted, anchor_values)
    adjusted = np.clip(adjusted, 0.0, None)

    stats = {
        "mean_anchor": float(np.mean(anchor_values)),
        "mean_final": float(np.mean(adjusted)),
        "mean_abs_move": float(np.mean(np.abs(adjusted - anchor_values))),
        "p95_abs_move": float(np.percentile(np.abs(adjusted - anchor_values), 95)),
        "max_abs_move": float(np.max(np.abs(adjusted - anchor_values))),
        "corr_final_anchor": float(np.corrcoef(adjusted, anchor_values)[0, 1]),
        "corr_final_sample": float(np.corrcoef(adjusted, sample_values)[0, 1]),
    }
    return adjusted, stats


def naive364_errors(sales: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows = []
    for end in pd.to_datetime(
        [
            "2016-12-31",
            "2017-12-31",
            "2018-12-31",
            "2019-12-31",
            "2020-12-31",
            "2021-12-31",
            "2022-12-31",
        ]
    ):
        start = end - pd.Timedelta(days=horizon - 1)
        if start < sales["Date"].min():
            continue
        train = sales[sales["Date"] < start]
        valid = sales[(sales["Date"] >= start) & (sales["Date"] <= end)]
        row = {"window": f"{start.date()}..{end.date()}"}
        maes = []
        for col in ["Revenue", "COGS"]:
            pred = historical_lag_prior(train, valid["Date"], col)
            mae = mean_absolute_error(valid[col], pred)
            row[f"{col}_mae"] = mae
            row[f"{col}_bias"] = float(np.mean(pred - valid[col].to_numpy()))
            maes.append(mae)
        row["avg_mae"] = float(np.mean(maes))
        rows.append(row)
    return pd.DataFrame(rows)


def print_drift_report(
    sales: pd.DataFrame,
    sample: pd.DataFrame,
    anchor: pd.DataFrame,
) -> None:
    sales_report = sales.copy()
    sales_report["year"] = sales_report["Date"].dt.year
    sales_report["ratio"] = sales_report["COGS"] / sales_report["Revenue"]
    yearly = (
        sales_report.groupby("year")
        .agg(
            days=("Date", "size"),
            rev_mean=("Revenue", "mean"),
            cogs_mean=("COGS", "mean"),
            ratio_med=("ratio", "median"),
        )
        .reset_index()
    )
    yearly["rev_yoy_pct"] = yearly["rev_mean"].pct_change() * 100
    yearly["cogs_yoy_pct"] = yearly["cogs_mean"].pct_change() * 100

    print("\n=== Yearly train distribution ===")
    print(
        yearly.to_string(
            index=False,
            formatters={
                "rev_mean": lambda x: f"{x:,.0f}",
                "cogs_mean": lambda x: f"{x:,.0f}",
                "ratio_med": lambda x: f"{x:.4f}",
                "rev_yoy_pct": lambda x: "" if pd.isna(x) else f"{x:+.1f}%",
                "cogs_yoy_pct": lambda x: "" if pd.isna(x) else f"{x:+.1f}%",
            },
        )
    )

    print("\n=== Forecast priors ===")
    for name, df in [("sample_submission", sample), ("anchor_best", anchor)]:
        ratio = df["COGS"] / df["Revenue"]
        print(
            f"{name:18s} Rev mean={df['Revenue'].mean():,.0f} "
            f"COGS mean={df['COGS'].mean():,.0f} ratio_med={ratio.median():.4f}"
        )

    print("\n=== Rolling 548-day lag-prior validation ===")
    errors = naive364_errors(sales, len(sample))
    print(
        errors.to_string(
            index=False,
            formatters={
                "Revenue_mae": lambda x: f"{x:,.0f}",
                "COGS_mae": lambda x: f"{x:,.0f}",
                "Revenue_bias": lambda x: f"{x:,.0f}",
                "COGS_bias": lambda x: f"{x:,.0f}",
                "avg_mae": lambda x: f"{x:,.0f}",
            },
        )
    )


def build_anchor(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    used = []
    for path in BEST_FILES:
        if not path.exists():
            continue
        df = read_csv(path)
        validate_forecast_frame(df, path.name)
        df = sample[["Date"]].merge(df, on="Date", how="left", validate="one_to_one")
        if df[["Revenue", "COGS"]].isna().any().any():
            raise ValueError(f"{path.name} does not align with sample_submission dates")
        frames.append(df[["Revenue", "COGS"]].astype(float))
        used.append(path.name)

    if not frames:
        raise FileNotFoundError(
            "No anchor submission found. Expected at least "
            "submission_raw_stable_neural_blend_w733_w563_monthly_cogs_b39.csv"
        )

    values = np.stack([frame.to_numpy(dtype=float) for frame in frames], axis=0)
    # Median is a safe average when the two available neural files differ only
    # slightly, and it remains robust if a future anchor variant is added.
    anchor = sample[["Date"]].copy()
    anchor[["Revenue", "COGS"]] = np.median(values, axis=0)
    return anchor, used


def main() -> None:
    sales_path = DATA_DIR / "sales.csv"
    sample_path = DATA_DIR / "sample_submission.csv"
    sales = read_csv(sales_path)
    sample = read_csv(sample_path)

    require_columns(sales, sales_path)
    validate_forecast_frame(sample, sample_path.name)

    anchor, used_anchor_files = build_anchor(sample)
    print(f"Using anchor files: {', '.join(used_anchor_files)}")
    print_drift_report(sales, sample, anchor)

    final = sample[["Date"]].copy()
    diagnostics: dict[str, object] = {
        "used_anchor_files": used_anchor_files,
        "output": "submission_v20_shape_calibrated_anchor.csv",
    }

    for col in ["Revenue", "COGS"]:
        lag_values = historical_lag_prior(sales, sample["Date"], col)
        preds, stats = calibrate_one_column(
            dates=sample["Date"],
            sample_values=sample[col].to_numpy(dtype=float),
            anchor_values=anchor[col].to_numpy(dtype=float),
            lag_values=lag_values,
            col=col,
        )
        final[col] = np.round(preds, 2)
        diagnostics[col] = stats

    # Guardrails for submission format and physically impossible values.
    for col in ["Revenue", "COGS"]:
        if not np.isfinite(final[col]).all():
            raise ValueError(f"{col} contains non-finite values")
        if (final[col] < 0).any():
            raise ValueError(f"{col} contains negative values")

    final_out = final.copy()
    final_out["Date"] = final_out["Date"].dt.strftime("%Y-%m-%d")
    out_path = OUT_DIR / "submission_v20_shape_calibrated_anchor.csv"
    final_out.to_csv(out_path, index=False)

    diag_path = OUT_DIR / "v20_shape_calibrated_anchor_diagnostics.json"
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\n=== Final submission ===")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(final_out)}")
    for col in ["Revenue", "COGS"]:
        stats = diagnostics[col]
        print(
            f"{col:7s} mean={final[col].mean():,.0f} "
            f"mean_abs_move_vs_anchor={stats['mean_abs_move']:,.0f} "
            f"p95_move={stats['p95_abs_move']:,.0f} "
            f"corr_anchor={stats['corr_final_anchor']:.5f}"
        )
    print(f"Diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
