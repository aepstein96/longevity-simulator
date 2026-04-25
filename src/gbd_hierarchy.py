"""
Tools for working with the GBD 2023 cause hierarchy.

The hierarchy is a tree rooted at `cause_id = 294` ("All causes"). Each node
has a `parent_id` (the root is its own parent). `level` goes 0..4.

All functions here operate on the flat CSV emitted by
preprocessing/process_gbd_hierarchy.py. Typical use:

    from src.gbd_hierarchy import Hierarchy
    h = Hierarchy.load()
    leaves = h.leaves_under(410)               # all cancer leaves
    all_cancer = h.descendants(410)            # cancer + every descendant
    rate = h.aggregate_leaf_rates(rates, 410)  # sum leaf rates -> Neoplasms rate

The "aggregate leaf rates" operation is the hazard-equivalent of logical-OR:
within a single (age, sex, year) cell all leaves share the same person-year
denominator, so rates sum directly to give the "any descendant event" rate
(ignoring within-cell co-occurrence, which is small when leaf rates are low).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterable

import pandas as pd


DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "GBD" / "cause_hierarchy.csv"


@dataclass
class Hierarchy:
    df: pd.DataFrame
    _parent: dict[int, int] = field(init=False)
    _children: dict[int, list[int]] = field(init=False)
    _level: dict[int, int] = field(init=False)
    _name: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self._parent   = dict(zip(self.df["cause_id"], self.df["parent_id"]))
        self._level    = dict(zip(self.df["cause_id"], self.df["level"]))
        self._name     = dict(zip(self.df["cause_id"], self.df["cause_name"]))
        self._children = defaultdict(list)
        for cid, pid in self._parent.items():
            if cid != pid:  # root is its own parent
                self._children[pid].append(cid)

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Hierarchy":
        return cls(pd.read_csv(path))

    # -----------------------------------------------------------------
    # Basic lookups
    # -----------------------------------------------------------------

    def name(self, cause_id: int) -> str:
        return self._name[cause_id]

    def level(self, cause_id: int) -> int:
        return self._level[cause_id]

    def parent(self, cause_id: int) -> int:
        return self._parent[cause_id]

    def children(self, cause_id: int) -> list[int]:
        return list(self._children.get(cause_id, []))

    # -----------------------------------------------------------------
    # Tree walks
    # -----------------------------------------------------------------

    def descendants(self, cause_id: int, include_self: bool = True) -> list[int]:
        """All causes reachable by descending from this node."""
        out: list[int] = [cause_id] if include_self else []
        queue = deque([cause_id])
        while queue:
            n = queue.popleft()
            for c in self._children.get(n, []):
                out.append(c)
                queue.append(c)
        return out

    def ancestors(self, cause_id: int, include_self: bool = False) -> list[int]:
        """All causes on the path to the root."""
        out: list[int] = [cause_id] if include_self else []
        n = cause_id
        while self._parent[n] != n:
            n = self._parent[n]
            out.append(n)
        return out

    def leaves_under(self, cause_id: int) -> list[int]:
        """Descendants that themselves have no children ("most detailed" under node)."""
        return [c for c in self.descendants(cause_id)
                if not self._children.get(c)]

    # -----------------------------------------------------------------
    # Data operations
    # -----------------------------------------------------------------

    def aggregate_leaf_rates(
        self,
        rates: pd.DataFrame,
        cause_id: int,
        rate_col: str = "val",
        cause_col: str = "cause_id",
    ) -> pd.DataFrame:
        """Sum the rates of all leaf descendants of `cause_id` across a
        long-format rate table, grouped by every non-cause column.

        Parameters
        ----------
        rates : long-format DataFrame with at least `cause_col`, `rate_col`,
                and one or more stratification columns (e.g. sex, age).
        cause_id : ancestor cause.
        rate_col : column with numeric rates to sum.
        cause_col : column containing cause_id.

        Returns a DataFrame with the same stratification columns plus
        `rate_col`, summed over leaves under `cause_id`. Within an (age, sex)
        cell, leaf rates share the same person-year denominator, so summing
        rates is equivalent to summing case counts.
        """
        leaves = set(self.leaves_under(cause_id))
        sub = rates[rates[cause_col].isin(leaves)]
        strat_cols = [c for c in rates.columns if c not in (cause_col, rate_col,
                                                             "cause_name",
                                                             "parent_id", "level",
                                                             "upper", "lower")]
        if not strat_cols:
            return pd.DataFrame({rate_col: [sub[rate_col].sum()]})
        return (sub.groupby(strat_cols, as_index=False)[rate_col].sum())

    def resolve(self, name_or_id: int | str) -> int:
        """Look up cause_id by either id or (case-insensitive) exact name."""
        if isinstance(name_or_id, (int,)):
            if name_or_id not in self._name:
                raise KeyError(f"Unknown cause_id {name_or_id}")
            return name_or_id
        target = name_or_id.strip().lower()
        for cid, nm in self._name.items():
            if nm.lower() == target:
                return cid
        raise KeyError(f"No cause named {name_or_id!r}")

    # -----------------------------------------------------------------
    # Human-friendly chapter groupings
    # -----------------------------------------------------------------

    # Level-2 chapters used by the simulator. These are a subset of the full
    # Level-2 list, chosen to align with the existing CDC cause categories
    # in src/causes.py. See CLAUDE.md for the 13 CDC categories.
    SIMULATOR_CHAPTERS: ClassVar[dict[str, int]] = {
        # human label shown in the UI          -> GBD Level-2 cause_id
        "Neoplasms":                          410,
        "Cardiovascular":                     491,
        "Chronic respiratory":                508,
        "Diabetes & kidney":                  974,
        "Digestive":                          526,
        "Neurological":                       542,
        "Mental":                             558,
        "Musculoskeletal":                    626,
        "Skin":                               653,
        "Sense organs":                       669,
        "Substance use":                      973,
        "Injuries":                           687,   # Level 1 in GBD, but treated as a chapter
        "Other NCDs":                         640,
    }
