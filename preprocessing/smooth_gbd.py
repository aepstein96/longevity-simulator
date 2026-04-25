"""
PCHIP-interpolate GBD rate tables from age bands to single-year ages.

Runs over every (sex, cause_id) series independently. Preserves the
hierarchy columns (parent_id, level) so downstream code doesn't need a join.

Inputs
------
data/GBD/{incidence,deaths,prevalence}_by_age_sex_cause.csv

Outputs
-------
data/GBD/{incidence,deaths,prevalence}_smoothed_single_year.csv
    Long-format: cause_id, cause_name, parent_id, level, sex, age (int),
                 val, upper, lower

Two banding schemes are supported:
- The fine-grained finegrained export uses '<1 year' + '2-4 years' as the
  youngest bands (FINEGRAINED_MIDPOINTS).
- The level-2/level-1 prevalence export uses '<5 years' as the youngest
  band (FIVE_YEAR_MIDPOINTS).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# Used for files that split the youngest age into '<1 year' + '2-4 years'.
# '<5 years' / '1-4 years' are dropped because they overlap.
FINEGRAINED_MIDPOINTS = {
    "<1 year":      0.5,
    "2-4 years":    3.0,
    "5-9 years":    7.0,
    "10-14 years":  12.0,
    "15-19 years":  17.0,
    "20-24 years":  22.0,
    "25-29 years":  27.0,
    "30-34 years":  32.0,
    "35-39 years":  37.0,
    "40-44 years":  42.0,
    "45-49 years":  47.0,
    "50-54 years":  52.0,
    "55-59 years":  57.0,
    "60-64 years":  62.0,
    "65-69 years":  67.0,
    "70-74 years":  72.0,
    "75-79 years":  77.0,
    "80-84 years":  82.0,
    "85-89 years":  87.0,
    "90-94 years":  92.0,
    "95+ years":    97.0,
    "0-6 days":     None,
    "7-27 days":    None,
    "1-5 months":   None,
    "6-11 months":  None,
    "<5 years":     None,
    "1-4 years":    None,
}

# Used for files where the youngest band is '<5 years' (no <1 / 2-4 split).
FIVE_YEAR_MIDPOINTS = {
    **{k: v for k, v in FINEGRAINED_MIDPOINTS.items() if v is not None
       and k not in ("<1 year", "2-4 years")},
    "<5 years": 2.0,
}


def smooth_long(df: pd.DataFrame, ages: np.ndarray,
                age_midpoints: dict | None = None) -> pd.DataFrame:
    if age_midpoints is None:
        age_midpoints = FINEGRAINED_MIDPOINTS
    df = df.copy()
    df["age_mid"] = df["age"].map(age_midpoints)
    df = df.dropna(subset=["age_mid"])

    # Static per-cause columns (not interpolated) are carried forward
    meta_cols = ["cause_name", "parent_id", "level"]
    out_rows = []
    for (sex, cause_id), grp in df.groupby(["sex", "cause_id"]):
        grp = (grp.sort_values("age_mid")
                  .drop_duplicates("age_mid", keep="first"))
        if len(grp) < 2:
            continue
        xs = grp["age_mid"].to_numpy(dtype=float)
        meta = {c: grp.iloc[0][c] for c in meta_cols}

        interpolated: dict[str, np.ndarray] = {}
        for col in ("val", "upper", "lower"):
            ys = grp[col].to_numpy(dtype=float)
            fit = PchipInterpolator(xs, ys, extrapolate=False)
            vals = fit(ages)
            vals[ages < xs[0]]  = ys[0]
            vals[ages > xs[-1]] = ys[-1]
            interpolated[col] = np.maximum(vals, 0.0)

        for i, age in enumerate(ages):
            row = {"cause_id": cause_id, "sex": sex, "age": int(age),
                   "val": interpolated["val"][i],
                   "upper": interpolated["upper"][i],
                   "lower": interpolated["lower"][i]}
            row.update(meta)
            out_rows.append(row)

    cols = ["cause_id", "cause_name", "parent_id", "level",
            "sex", "age", "val", "upper", "lower"]
    return pd.DataFrame(out_rows)[cols]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    gbd_dir = root / "data" / "GBD"
    ages = np.arange(0, 101)

    pairs = [
        ("incidence",  "incidence_by_age_sex_cause.csv",
                       "incidence_smoothed_single_year.csv",
                       FINEGRAINED_MIDPOINTS),
        ("deaths",     "deaths_by_age_sex_cause.csv",
                       "deaths_smoothed_single_year.csv",
                       FINEGRAINED_MIDPOINTS),
        ("prevalence", "prevalence_by_age_sex_cause.csv",
                       "prevalence_smoothed_single_year.csv",
                       FIVE_YEAR_MIDPOINTS),
    ]
    for label, src_name, dst_name, midpoints in pairs:
        src = gbd_dir / src_name
        if not src.exists():
            print(f"SKIP {label}: {src} missing")
            continue
        df = pd.read_csv(src)
        smooth = smooth_long(df, ages, age_midpoints=midpoints)
        dst = gbd_dir / dst_name
        smooth.to_csv(dst, index=False)
        print(f"Wrote {dst}: {len(smooth):,} rows, "
              f"{smooth['cause_id'].nunique()} causes, "
              f"sexes {sorted(smooth['sex'].unique())}, "
              f"ages {smooth['age'].min()}-{smooth['age'].max()}")


if __name__ == "__main__":
    main()
