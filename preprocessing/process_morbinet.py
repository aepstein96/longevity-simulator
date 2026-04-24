"""
Parse the markitdown-ified MorbiNet supplement (data/morbinet/supplement.md)
and extract each of the 11 supplementary tables into its own CSV in
data/morbinet/.

The supplement is a mix of shapes:
- S1: ICPC-2 code dictionary, with italic chapter rows and merged "grouped
      codes" rows. We carry the chapter forward and keep the grouped-codes
      field as-is.
- S2: Two-level stacked header (sex x metric, then N/% per row). We flatten
      by concatenating the header rows.
- S3, S6: Simple rectangular tables with a single header row.
- S5, S7: Edge lists (pairwise ORs). Clean 5-column layout.
- S4, S8-S11: Mixed descriptive tables — extracted as-is.

No attempt is made to reshape the data beyond making each table a clean CSV.
Downstream modeling can pick and choose columns.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SUPP_PATH = REPO_ROOT / "data" / "morbinet" / "supplement.md"
OUT_DIR = REPO_ROOT / "data" / "morbinet"


TABLE_TITLES = {
    "S1":  "icpc2_codes",
    "S2":  "multimorbidity_prevalence_by_age_sex",
    "S3":  "network_parameters",
    "S4":  "diabetes_patient_characteristics",
    "S5":  "undirected_or_edges",
    "S6":  "node_degree_pagerank",
    "S7":  "directed_or_edges",
    "S8":  "temporal_associations",
    "S9":  "sensitivity_or_thresholds",
    "S10": "sensitivity_temporal_thresholds",
    "S11": "directional_associations_20_80",
}


def _strip_md(cell: str) -> str:
    """Strip bold/italic markers, trim whitespace."""
    t = cell.strip()
    t = re.sub(r"^\*{1,3}|\*{1,3}$", "", t)
    t = t.replace("​", "").strip()
    return t


def _split_tables(md_text: str) -> dict[str, list[str]]:
    """Split the supplement into per-table blocks keyed by 'S<n>'."""
    blocks: dict[str, list[str]] = {}
    header_re = re.compile(r"^\*\*Table\s+(S\d+)\.?\s*(.*?)\*\*\s*$", re.IGNORECASE)
    current = None
    for line in md_text.splitlines():
        m = header_re.match(line.strip())
        if m:
            current = m.group(1).upper()
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)
    return blocks


def _extract_rows(lines: list[str]) -> list[list[str]]:
    """Extract markdown table rows from a block.

    Some tables (notably S1) span multiple markdown sub-tables glued under
    one heading, separated by blank lines. We scan through the whole block,
    dropping divider rows and collapsing runs of all-empty rows.
    """
    rows: list[list[str]] = []
    for line in lines:
        s = line.rstrip()
        if not s.startswith("|"):
            continue
        cells_raw = [c.strip() for c in s.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c or "") for c in cells_raw):
            continue
        rows.append([_strip_md(c) for c in cells_raw])
    deduped: list[list[str]] = []
    prev_empty = False
    for r in rows:
        is_empty = not any(c for c in r)
        if is_empty and prev_empty:
            continue
        deduped.append(r)
        prev_empty = is_empty
    return deduped


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pad or trim rows to match header length
    width = len(header)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            if len(r) < width:
                r = r + [""] * (width - len(r))
            elif len(r) > width:
                r = r[:width]
            w.writerow(r)


def _carry_forward(values: list[str]) -> list[str]:
    """Forward-fill empty strings with the last non-empty neighbour."""
    out = []
    prev = ""
    for v in values:
        if v:
            prev = v
        out.append(prev)
    return out


# ------------------------------------------------------------------
# Table-specific shapers
# ------------------------------------------------------------------

def shape_s1(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """
    S1 is one logical table split across several markdown sub-tables, each
    with its own column-header row and empty placeholder row. Row shapes:

      italic chapter header  -> only first cell filled, text = chapter name
      code row               -> (code, description, maybe grouped_codes)
      aggregate row          -> (empty, aggregate_name, "code1, code2, ...")
      column header          -> "ICPC 2 code" / "Description" / "..."

    Emit: chapter, icpc2_code, description, grouped_codes.
    """
    header = ["chapter", "icpc2_code", "description", "grouped_codes"]
    out: list[list[str]] = []
    current_chapter = ""

    for r in rows:
        cells = (r + ["", "", ""])[:3]
        code, desc, grouped = cells[0], cells[1], cells[2]
        # Skip the recurring column-header rows
        if code.lower().startswith("icpc") and "code" in code.lower():
            continue
        # Chapter rows: only first cell set
        if code and not desc and not grouped:
            current_chapter = code
            continue
        # Completely empty row
        if not code and not desc and not grouped:
            continue
        out.append([current_chapter, code, desc, grouped])
    return header, out


def shape_s2(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Flatten the three-row stacked header of S2."""
    if len(rows) < 4:
        return (rows[0] if rows else []), rows[1:] if len(rows) > 1 else []
    h1 = _carry_forward(rows[0])
    h2 = _carry_forward(rows[1])
    h3 = rows[2]
    header = []
    for a, b, c in zip(h1, h2, h3):
        parts = [p for p in (a, b, c) if p]
        header.append("_".join(parts) if parts else "col")
    # Dedup
    seen: dict[str, int] = {}
    final = []
    for h in header:
        n = seen.get(h, 0)
        final.append(f"{h}_{n}" if n else h)
        seen[h] = n + 1
    return final, rows[3:]


def shape_generic(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """One header row, data rows below."""
    if not rows:
        return [], []
    # Some blocks start with an all-empty placeholder row before the header
    if rows and not any(c for c in rows[0]):
        rows = rows[1:]
    if not rows:
        return [], []
    header = [h or f"col{i}" for i, h in enumerate(rows[0])]
    return header, rows[1:]


SHAPERS = {
    "S1": shape_s1,
    "S2": shape_s2,
}


def main() -> None:
    md = SUPP_PATH.read_text()
    blocks = _split_tables(md)
    print(f"Found {len(blocks)} tables: {sorted(blocks.keys())}")

    for key, lines in blocks.items():
        rows = _extract_rows(lines)
        if not rows:
            print(f"  {key}: no rows parsed, skipping")
            continue
        shaper = SHAPERS.get(key, shape_generic)
        header, data = shaper(rows)
        slug = TABLE_TITLES.get(key, key.lower())
        out_path = OUT_DIR / f"table{key.lower()}_{slug}.csv"
        _write_csv(out_path, header, data)
        print(f"  {key:3s} -> {out_path.name}  ({len(data)} rows, {len(header)} cols)")


if __name__ == "__main__":
    main()
