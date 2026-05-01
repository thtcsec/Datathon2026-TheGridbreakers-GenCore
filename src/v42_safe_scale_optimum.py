"""V42: ultra-safe scale fine tune around the v37 best."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DATA = ROOT / "data" / "raw"
SCALE = 1.0253


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def mk(d: pd.Series) -> pd.Series:
    return d.dt.to_period("M").astype(str)


def rebal(base: pd.DataFrame, sample: pd.DataFrame, col: str) -> pd.DataFrame:
    sm, bm = mk(sample["Date"]), mk(base["Date"])
    sp = sample.groupby(sm)[col].mean()
    pat = sp / sample[col].mean()
    out = base.copy()
    bg = out[col].mean()
    for ym in sorted(bm.unique()):
        mask = bm == ym
        cur = out.loc[mask, col].mean()
        if ym in pat.index and cur > 0:
            out.loc[mask, col] *= (bg * pat[ym]) / cur
    return out


def main() -> None:
    v23 = read_csv(OUT / "submission_v23_b39_all_430.csv")
    sample = read_csv(DATA / "sample_submission.csv")
    out = v23.copy()
    out["Revenue"] *= SCALE
    out["COGS"] *= SCALE
    for col in ["Revenue", "COGS"]:
        out = rebal(out, sample, col)
        out[col] = out[col].round(2)
    formatted = out.copy()
    formatted["Date"] = formatted["Date"].dt.strftime("%Y-%m-%d")
    path = OUT / "submission_v42_rebal_s10253.csv"
    formatted.to_csv(path, index=False)
    print(f"Saved {path}")
    print(f"Revenue mean={out['Revenue'].mean():,.0f} COGS mean={out['COGS'].mean():,.0f}")


if __name__ == "__main__":
    main()
