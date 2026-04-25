"""Convert the GBD prevalence export into the project's tidy long schema.

Input
-----
data/GBD/final_GBD_raw_data.csv
    GBD Results Tool export with measure_name in {'Deaths', 'Prevalence'},
    metric_name='Rate', United States, 2023, 5-year age bands (plus
    '<5 years') by sex (Male/Female/Both), at level-1 / level-2 cause
    aggregation. Same wide column set as IHME exports.

data/GBD/cause_hierarchy.csv
    Used to attach parent_id and level.

Output
------
data/GBD/prevalence_by_age_sex_cause.csv
    Long format: cause_id, cause_name, parent_id, level, sex, age (band str),
                 val, upper, lower
    Rate per 100,000 person-years (GBD native units).

Notes
-----
- This file's age bands are 5-year groupings starting with '<5 years' (i.e.
  no '<1 year' / '2-4 years' split). Downstream smoothing treats '<5 years'
  as midpoint 2.0 — see preprocessing/smooth_gbd.py.
- We keep the 'Both' sex (sex_id=3) alongside 'Male' and 'Female' so the
  consumer can either pick a sex directly or compute its own aggregate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
GBD_DIR = REPO_ROOT / "data" / "GBD"
SRC = GBD_DIR / "final_GBD_raw_data.csv"
HIER_PATH = GBD_DIR / "cause_hierarchy.csv"
OUT = GBD_DIR / "prevalence_by_age_sex_cause.csv"


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df):,} rows from {SRC.name}")

    # --- normalize column names -------------------------------------------
    df = df.rename(columns={
        "measure_name": "measure",
        "sex_name":     "sex",
        "age_name":     "age",
    })

    # --- sanity checks ----------------------------------------------------
    for col, want in [("metric_name", "Rate"),
                      ("location_name", "United States of America"),
                      ("year", 2023)]:
        u = df[col].unique()
        if len(u) != 1 or u[0] != want:
            print(f"  WARNING: {col} is not constant {want!r}: {u}")

    # --- filter to Prevalence ---------------------------------------------
    pre = df[df["measure"] == "Prevalence"].copy()
    print(f"Prevalence rows: {len(pre):,}")

    # --- join hierarchy ---------------------------------------------------
    hier = pd.read_csv(HIER_PATH)
    pre = pre.merge(
        hier[["cause_id", "parent_id", "level"]],
        on="cause_id", how="left", validate="many_to_one",
    )
    missing = pre["level"].isna().sum()
    if missing:
        bad = sorted(pre.loc[pre["level"].isna(), "cause_id"].unique())
        print(f"  WARNING: {missing} rows with cause_id not in hierarchy: {bad}")
    pre["parent_id"] = pre["parent_id"].astype("Int64")
    pre["level"]     = pre["level"].astype("Int64")

    # --- order by cause then sex then age band ----------------------------
    age_order = [
        "<5 years", "5-9 years", "10-14 years", "15-19 years", "20-24 years",
        "25-29 years", "30-34 years", "35-39 years", "40-44 years",
        "45-49 years", "50-54 years", "55-59 years", "60-64 years",
        "65-69 years", "70-74 years", "75-79 years", "80-84 years",
        "85-89 years", "90-94 years", "95+ years",
    ]
    age_idx = {a: i for i, a in enumerate(age_order)}
    pre["_age_sort"] = pre["age"].map(age_idx).fillna(999).astype(int)
    pre = (pre.sort_values(["cause_id", "sex", "_age_sort"])
              .drop(columns="_age_sort")
              .reset_index(drop=True))

    out = pre[["cause_id", "cause_name", "parent_id", "level",
               "sex", "age", "val", "upper", "lower"]]
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out):,} rows, "
          f"{out['cause_id'].nunique()} causes, "
          f"{out['age'].nunique()} age bands, "
          f"sexes {sorted(out['sex'].unique())}")


if __name__ == "__main__":
    main()
