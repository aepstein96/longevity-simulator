"""
Flatten the GBD 2023 cause hierarchy XLSX into a tidy CSV.

Input
-----
data/GBD/raw/IHME_GBD_2023_HIERARCHIES_Y2025M10D23.XLSX
    Sheet "Cause Hierarchy" with columns:
        Cause ID, Cause Name, Parent ID, Parent Name,
        Level, Cause Outline, Sort Order, YLL Only, YLD Only

Output
------
data/GBD/cause_hierarchy.csv
    One row per cause (381 rows). Columns:
        cause_id, cause_name, parent_id, level, cause_outline, sort_order

The root (Level 0, "All causes", cause_id=294) is its own parent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "GBD" / "raw" / "IHME_GBD_2023_HIERARCHIES_Y2025M10D23.XLSX"
OUT = REPO_ROOT / "data" / "GBD" / "cause_hierarchy.csv"


def main() -> None:
    df = pd.read_excel(SRC, sheet_name="Cause Hierarchy")
    df = df.rename(columns={
        "Cause ID":      "cause_id",
        "Cause Name":    "cause_name",
        "Parent ID":     "parent_id",
        "Level":         "level",
        "Cause Outline": "cause_outline",
        "Sort Order":    "sort_order",
    })[["cause_id", "cause_name", "parent_id", "level",
        "cause_outline", "sort_order"]]
    df["cause_id"]   = df["cause_id"].astype(int)
    df["parent_id"]  = df["parent_id"].astype(int)
    df["level"]      = df["level"].astype(int)
    df["sort_order"] = df["sort_order"].astype(int)
    df = df.sort_values("sort_order").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"Wrote {OUT}: {len(df)} causes")
    print("\nLevel distribution:")
    print(df["level"].value_counts().sort_index().to_string())
    print("\nLevel 1 (top chapters):")
    print(df.loc[df["level"] == 1, ["cause_id", "cause_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
