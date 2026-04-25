"""
Consolidate the GBD 2023 fine-grained Results Tool export into a tidy CSV,
joined with the cause hierarchy so downstream code can walk parent/child.

Input
-----
data/GBD/raw/IHME-GBD_2023_DATA-*_finegrained.csv
    Must contain measure=Incidence (and/or Deaths), metric=Rate, a single
    location (US), a single year, most-detailed causes by cause_id.
data/GBD/cause_hierarchy.csv
    Built by preprocessing/process_gbd_hierarchy.py. Joined on cause_id.

Output
------
data/GBD/incidence_by_age_sex_cause.csv   (if Incidence rows present)
data/GBD/deaths_by_age_sex_cause.csv      (if Deaths rows present)

Tidy schema:
    cause_id, cause_name, parent_id, level, sex, age, val, upper, lower

Filters applied
---------------
1. Age band "<5 years" is dropped because it double-counts "<1 year"+"2-4 years".
2. Level-1 aggregate causes that are not part of the main CMNN/NCD/Injury
   hierarchy (Total cancers, Total burden related to hepatitis B, etc.) are
   dropped — they sum over leaves and would double-count if kept.
3. Rates are left as "per 100 000 person-years" (GBD's native units). Code
   that consumes this as a hazard should divide by 100 000 at use time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR   = REPO_ROOT / "data" / "GBD" / "raw"
OUT_DIR   = REPO_ROOT / "data" / "GBD"
HIER_PATH = OUT_DIR / "cause_hierarchy.csv"


AGE_BAND_ORDER = [
    "<1 year", "2-4 years", "5-9 years", "10-14 years", "15-19 years",
    "20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years",
    "45-49 years", "50-54 years", "55-59 years", "60-64 years", "65-69 years",
    "70-74 years", "75-79 years", "80-84 years", "85-89 years", "90-94 years",
    "95+ years",
]
# Cause IDs of "Total X" pseudo-chapters in the hierarchy (Level 1 but not
# part of the CMNN/NCD/Injury tree). These double-count their descendants.
TOTAL_PSEUDO_CHAPTERS = {1026, 1027, 1028, 1029, 1059}


def load_finegrained() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("IHME-GBD_*finegrained*.csv"))
    if not files:
        raise SystemExit(f"No fine-grained export found under {RAW_DIR}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} fine-grained file(s)")
    return df


def main() -> None:
    df = load_finegrained()
    hier = pd.read_csv(HIER_PATH)

    # --- normalize column names -------------------------------------------
    df = df.rename(columns={
        "measure_name": "measure",
        "sex_name":     "sex",
        "age_name":     "age",
        "cause_name":   "cause_name",
    })

    # --- sanity: expect constants -----------------------------------------
    for col, want in [("metric_name", "Rate"),
                      ("location_name", "United States of America"),
                      ("year", 2023)]:
        u = df[col].unique()
        if len(u) != 1 or u[0] != want:
            print(f"  WARNING: {col} is not constant {want!r}: {u}")

    # --- drop unwanted rows ------------------------------------------------
    before = len(df)
    df = df[df["age"] != "<5 years"]
    df = df[~df["cause_id"].isin(TOTAL_PSEUDO_CHAPTERS)]
    print(f"Dropped {before - len(df):,} rows "
          f"(<5 years aggregate + Total* pseudo-chapters)")

    # --- join hierarchy ---------------------------------------------------
    df = df.merge(
        hier[["cause_id", "parent_id", "level"]],
        on="cause_id", how="left", validate="many_to_one",
    )
    missing_hier = df["level"].isna().sum()
    if missing_hier:
        missing_ids = sorted(df.loc[df["level"].isna(), "cause_id"].unique())
        print(f"  WARNING: {missing_hier} rows with cause_id not in hierarchy: "
              f"{missing_ids}")
    df["parent_id"] = df["parent_id"].astype("Int64")
    df["level"]     = df["level"].astype("Int64")

    # --- order & write per measure ----------------------------------------
    age_order = {a: i for i, a in enumerate(AGE_BAND_ORDER)}
    df["_age_sort"] = df["age"].map(age_order).fillna(999).astype(int)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for measure, slug in [("Incidence", "incidence"), ("Deaths", "deaths")]:
        sub = df[df["measure"] == measure]
        if sub.empty:
            print(f"  (no {measure} rows — skipping)")
            continue
        sub = sub[["cause_id", "cause_name", "parent_id", "level",
                   "sex", "age", "_age_sort", "val", "upper", "lower"]]
        sub = (sub.sort_values(["cause_id", "sex", "_age_sort"])
                  .drop(columns="_age_sort")
                  .reset_index(drop=True))
        out = OUT_DIR / f"{slug}_by_age_sex_cause.csv"
        sub.to_csv(out, index=False)
        print(f"Wrote {out}: {len(sub):,} rows, "
              f"{sub['cause_id'].nunique()} causes, "
              f"ages {sub['age'].nunique()}, "
              f"sexes {sorted(sub['sex'].unique())}")


if __name__ == "__main__":
    main()
