"""
Build a mapping from MorbiNet's ICPC-2 dictionary (data/morbinet/tables1_...)
to the 14 GBD Level-2 causes used in data/GBD/*.

Strategy
--------
1. Default: each ICPC-2 chapter maps to one GBD bucket.
2. Override: description keywords promote codes to the right specific bucket
   (e.g. any "neoplasm"/"malignant"/"cancer"/"leukaemia"/"lymphoma"/"hodgkin"
    text -> Neoplasms, regardless of chapter).
3. Hand-listed exceptions for codes that don't fit the chapter default
   (HIV -> Communicable; drug/alcohol use -> Substance use; TB -> Communicable).

Outputs
-------
- data/morbinet/icpc2_to_gbd_mapping.csv  one row per ICPC-2 code
- console summary of how the 14 GBD buckets are covered, and which codes in
  the OR edge-list (Table S5) fall in each bucket.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
MORBI_DIR = REPO_ROOT / "data" / "morbinet"
GBD_DIR = REPO_ROOT / "data" / "GBD"


# The 14 GBD Level-2 buckets that appear in our downloaded GBD data.
GBD_LEVEL2 = [
    "Cardiovascular diseases",
    "Chronic respiratory diseases",
    "Communicable, maternal, neonatal, and nutritional diseases",
    "Diabetes and kidney diseases",
    "Digestive diseases",
    "Injuries",
    "Mental disorders",
    "Musculoskeletal disorders",
    "Neoplasms",
    "Neurological disorders",
    "Other non-communicable diseases",
    "Sense organ diseases",
    "Skin and subcutaneous diseases",
    "Substance use disorders",
]


CHAPTER_TO_GBD = {
    "General and unspecified": "Other non-communicable diseases",
    "Blood, Blood Forming Organs and Immune Mechanism": "Other non-communicable diseases",
    "Digestive": "Digestive diseases",
    "Eye": "Sense organ diseases",
    "Ear": "Sense organ diseases",
    "Cardiovascular": "Cardiovascular diseases",
    "Musculoskeletal": "Musculoskeletal disorders",
    "Neurological": "Neurological disorders",
    "Psychological": "Mental disorders",
    "Respiratory": "Chronic respiratory diseases",
    "Skin": "Skin and subcutaneous diseases",
    "Endocrine/Metabolic and Nutritional": "Other non-communicable diseases",
    "Urological": "Diabetes and kidney diseases",
    "Pregnancy, Childbearing, Family Planning": "Communicable, maternal, neonatal, and nutritional diseases",
    "Female Genital": "Other non-communicable diseases",
    "Male Genital": "Other non-communicable diseases",
}


# Description keywords -> GBD bucket. Checked before the chapter default.
CANCER_KEYS = ("neoplasm", "malignant", "malignancy", "cancer",
               "leukaemia", "leukemia", "lymphoma", "hodgkin")


# Explicit per-code overrides.
EXPLICIT = {
    "A70": "Communicable, maternal, neonatal, and nutritional diseases",  # TB
    "B90": "Communicable, maternal, neonatal, and nutritional diseases",  # HIV
    "A90": "Communicable, maternal, neonatal, and nutritional diseases",  # congenital NOS
    "T89": "Diabetes and kidney diseases",  # Type 1 diabetes
    "T90": "Diabetes and kidney diseases",  # Type 2 diabetes
    "T85": "Other non-communicable diseases",  # hypothyroidism
    "T86": "Other non-communicable diseases",  # hyperthyroidism
    "T82": "Other non-communicable diseases",  # obesity
    "T83": "Other non-communicable diseases",  # overweight
    # ciap2_all_ALL_edges.txt includes decimal/aggregate codes absent from S1
    "T83/82":    "Other non-communicable diseases",   # Overweight/Obesity aggregate
    "D99.01":    "Digestive diseases",                # Coeliac disease
    "K94.01":    "Cardiovascular diseases",           # Phlebitis (superficial)
    "K94.02":    "Cardiovascular diseases",           # Phlebitis (deep)
    "L82.01":    "Musculoskeletal disorders",         # Congenital anomaly MSK
    "L86/84/83": "Musculoskeletal disorders",         # Neck/back syndrome aggregate
    "L88.01":    "Musculoskeletal disorders",         # Ankylosing spondylitis
    "N18":       "Neurological disorders",            # Paralysis/weakness
    "P79.01":    "Mental disorders",                  # OCD
    "P79.02":    "Mental disorders",                  # Phobia
    "YX15":      "Other non-communicable diseases",   # Infertility
    "YX99":      "Other non-communicable diseases",   # Genital disease, other
}


# Map "Psychological" codes P15/16/17/18/19 (alcohol/drug abuse) -> Substance use
PSYCH_SUBSTANCE_CODES = {"P15", "P16", "P17", "P18", "P19"}


def map_code(code: str, description: str, chapter: str) -> str:
    code = (code or "").strip()
    desc = (description or "").lower()
    chapter = chapter or ""

    if code in EXPLICIT:
        return EXPLICIT[code]

    if chapter == "Psychological" and code in PSYCH_SUBSTANCE_CODES:
        return "Substance use disorders"

    if any(k in desc for k in CANCER_KEYS):
        return "Neoplasms"

    return CHAPTER_TO_GBD.get(chapter, "Other non-communicable diseases")


def main() -> None:
    s1 = pd.read_csv(MORBI_DIR / "tables1_icpc2_codes.csv").fillna("")
    s1["gbd_level2"] = [
        map_code(c, d, ch)
        for c, d, ch in zip(s1["icpc2_code"], s1["description"], s1["chapter"])
    ]

    # Also include codes that appear in the full edge list but not in S1
    # (decimal codes like D99.01, K94.01, aggregates like T83/82).
    edges_path = MORBI_DIR / "ciap2_all_ALL_edges.txt"
    edges_df = pd.read_csv(edges_path, sep="\t")
    edge_codes_with_labels = (
        pd.concat([
            edges_df[["Disease_A", "label_A"]].rename(
                columns={"Disease_A": "icpc2_code", "label_A": "description"}),
            edges_df[["Disease_B", "label_B"]].rename(
                columns={"Disease_B": "icpc2_code", "label_B": "description"}),
        ])
        .drop_duplicates("icpc2_code")
    )
    known_codes = set(s1["icpc2_code"])
    extra_rows = []
    for _, r in edge_codes_with_labels.iterrows():
        if r["icpc2_code"] in known_codes:
            continue
        # Not in S1 — use EXPLICIT if we have it, else keyword fallback
        gbd = EXPLICIT.get(
            r["icpc2_code"],
            "Neoplasms" if any(k in r["description"].lower()
                               for k in CANCER_KEYS)
            else "Other non-communicable diseases",
        )
        extra_rows.append({
            "chapter": "(from edges file)",
            "icpc2_code": r["icpc2_code"],
            "description": r["description"],
            "grouped_codes": "",
            "gbd_level2": gbd,
        })
    if extra_rows:
        s1 = pd.concat([s1, pd.DataFrame(extra_rows)], ignore_index=True)

    s1_out = s1[["chapter", "icpc2_code", "description",
                 "grouped_codes", "gbd_level2"]]
    out_path = MORBI_DIR / "icpc2_to_gbd_mapping.csv"
    s1_out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(s1_out)} rows "
          f"({len(extra_rows)} added from edges file)\n")

    # Build a code-level lookup
    code_map = dict(zip(s1.loc[s1["icpc2_code"] != "", "icpc2_code"],
                        s1.loc[s1["icpc2_code"] != "", "gbd_level2"]))

    # Coverage vs the full edge list
    edge_nodes = set(edges_df["Disease_A"]).union(edges_df["Disease_B"])
    codes_by_bucket_all = s1.loc[s1["icpc2_code"] != ""] \
        .groupby("gbd_level2")["icpc2_code"].apply(list).to_dict()
    codes_by_bucket_edges: dict[str, list[str]] = {}
    for code in edge_nodes:
        bucket = code_map.get(code, "UNMAPPED")
        codes_by_bucket_edges.setdefault(bucket, []).append(code)

    print("GBD Level-2 bucket                                              "
          "| S1 codes | edge-file nodes")
    print("-" * 98)
    for bucket in GBD_LEVEL2:
        all_n = len(codes_by_bucket_all.get(bucket, []))
        edge_n = len(codes_by_bucket_edges.get(bucket, []))
        print(f"{bucket:<65s} | {all_n:>8d} | {edge_n:>9d}")

    unmapped = codes_by_bucket_edges.get("UNMAPPED", [])
    if unmapped:
        print(f"\nUNMAPPED edge nodes (need review): {sorted(unmapped)}")

    # Per-bucket edge count in the full 1277-edge network
    print("\n--- edges in full network by (A_bucket, B_bucket) pair ---")
    edges_df["bucket_A"] = edges_df["Disease_A"].map(code_map).fillna("UNMAPPED")
    edges_df["bucket_B"] = edges_df["Disease_B"].map(code_map).fillna("UNMAPPED")
    pair_counts = (
        edges_df.groupby(["bucket_A", "bucket_B"]).size().sort_values(ascending=False)
    )
    print(pair_counts.head(20).to_string())


if __name__ == "__main__":
    main()
