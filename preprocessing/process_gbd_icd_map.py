"""
Parse the GBD 2021 Nonfatal ICD-10 mapping XLSX into a tidy lookup.

The source file has one row per GBD cause (Levels 2-4) and a comma-separated
string of ICD-10 codes/ranges per row, e.g.

    C50-C50.9, D05-D05.9

Output
------
data/GBD/icd10_ranges_to_gbd.csv
    One row per ICD-10 range (not per code). Columns:
        cause_id, cause_name, level, icd10_start, icd10_end
    Ranges that are single codes have start == end.

data/GBD/icd10_prefix_to_gbd.csv
    One row per 3-character ICD-10 prefix -> cause_id. When a prefix is claimed
    by multiple causes, the most-detailed (highest level) wins. Ties within a
    level are broken by cause_id. Columns:
        icd10_prefix, cause_id, cause_name, level

The 3-char prefix file is what CDC cause-fraction preprocessing uses to
categorize deaths, since ICD-10 codes in CDC MMCD are reported at the 3-4
char level and the simulator's existing code groups them at that granularity.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "data" / "GBD" / "raw" / "IHME_GBD_2021_NONFATAL_CAUSE_ICD_CODE_MAP_Y2024M05D16_0.XLSX"
OUT_RANGES = REPO_ROOT / "data" / "GBD" / "icd10_ranges_to_gbd.csv"
OUT_PREFIX = REPO_ROOT / "data" / "GBD" / "icd10_prefix_to_gbd.csv"


# ICD-10 code = 1 letter + 2 digits, optionally ".digits"
_CODE_RE = re.compile(r"^([A-Z])(\d{2})(?:\.(\d+))?$")


def parse_icd_code(code: str) -> tuple[str, int, int]:
    """(letter, base_number, decimal_int) — decimal_int=0 if no decimal."""
    m = _CODE_RE.match(code.strip())
    if not m:
        raise ValueError(f"bad ICD-10 code: {code!r}")
    letter, base, dec = m.group(1), int(m.group(2)), m.group(3) or "0"
    # Normalize decimal to a single integer so B20.1 < B20.10 etc. — length-pad
    dec_int = int(dec.ljust(2, "0")[:2])
    return letter, base, dec_int


def parse_range_string(cell: str) -> list[tuple[str, str]]:
    """Split a cell like 'A50-A60.9, B63' into [(start, end), ...]."""
    if not isinstance(cell, str):
        return []
    out = []
    for part in cell.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [p.strip() for p in part.split("-", 1)]
            out.append((a, b))
        else:
            out.append((part, part))
    return out


def prefixes_in_range(start: str, end: str) -> list[str]:
    """3-char prefixes covered by an ICD-10 range [start, end] inclusive.

    Both endpoints must share the same letter. Returns prefixes like 'A50'.
    """
    ls, ns, _ = parse_icd_code(start)
    le, ne, _ = parse_icd_code(end)
    if ls != le:
        # cross-letter ranges would need alphabetical expansion; the GBD file
        # doesn't contain any, but fall back to [start] if it ever happens
        return [f"{ls}{ns:02d}"]
    return [f"{ls}{n:02d}" for n in range(ns, ne + 1)]


def main() -> None:
    df = pd.read_excel(SRC, sheet_name="Sheet1", header=1)
    df = df.rename(columns={
        "Cause ID":              "cause_id",
        "Cause Hierarchy Level": "level",
        "Cause Name":            "cause_name",
        "ICD10":                 "icd10",
    })[["cause_id", "level", "cause_name", "icd10"]].dropna(subset=["cause_id"])
    # Trailing "NOTE: ..." row slips in as a non-numeric cause_id
    df = df[pd.to_numeric(df["cause_id"], errors="coerce").notna()]
    df["cause_id"] = df["cause_id"].astype(int)
    df["level"]    = df["level"].astype(int)

    # ---- ranges table ------------------------------------------------------
    ranges_rows = []
    for _, r in df.iterrows():
        for start, end in parse_range_string(r["icd10"]):
            ranges_rows.append({
                "cause_id":    r["cause_id"],
                "cause_name":  r["cause_name"],
                "level":       r["level"],
                "icd10_start": start,
                "icd10_end":   end,
            })
    ranges = pd.DataFrame(ranges_rows)
    OUT_RANGES.parent.mkdir(parents=True, exist_ok=True)
    ranges.to_csv(OUT_RANGES, index=False)
    print(f"Wrote {OUT_RANGES}: {len(ranges)} ICD-10 ranges")

    # ---- 3-char prefix table: "most detailed wins" -------------------------
    # For each (prefix, cause), record (level, cause_id). Then for each prefix,
    # keep the row with max level (break ties by smaller cause_id).
    prefix_rows = []
    for _, r in ranges.iterrows():
        try:
            prefs = prefixes_in_range(r["icd10_start"], r["icd10_end"])
        except ValueError:
            continue
        for p in prefs:
            prefix_rows.append({
                "icd10_prefix": p,
                "cause_id":     r["cause_id"],
                "cause_name":   r["cause_name"],
                "level":        r["level"],
            })
    prefix_df = pd.DataFrame(prefix_rows)
    # Most detailed wins
    prefix_df = prefix_df.sort_values(["icd10_prefix", "level", "cause_id"],
                                       ascending=[True, False, True])
    prefix_df = prefix_df.drop_duplicates("icd10_prefix", keep="first")
    prefix_df = prefix_df.sort_values("icd10_prefix").reset_index(drop=True)
    prefix_df.to_csv(OUT_PREFIX, index=False)
    print(f"Wrote {OUT_PREFIX}: {len(prefix_df)} unique ICD-10 3-char prefixes")

    # Sanity: how many 3-char prefixes did we cover?
    print(f"\nLevel distribution of prefix claims:")
    print(prefix_df["level"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
