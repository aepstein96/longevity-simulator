import pandas as pd
import pytest

from src.causes import (
    categorize_cause,
    remove_cause_from_lifetable,
)


def test_categorize_cause_returns_unknown_for_missing_input():
    assert categorize_cause(None) == "Unknown"
    assert categorize_cause([]) == "Unknown"


def test_categorize_cause_returns_other_for_unmapped_code():
    assert categorize_cause(["Z99"]) == "Other"


def test_remove_cause_reduces_mortality_by_cause_fraction():
    mortality = pd.Series(
        [0.02, 0.04],
        index=[50, 51],
    )

    fractions = pd.DataFrame(
        {"Cancer": [0.25, 0.50]},
        index=[50, 51],
    )

    result = remove_cause_from_lifetable(
        mortality,
        fractions,
        "Cancer",
    )

    assert result.loc[50] == pytest.approx(0.015)
    assert result.loc[51] == pytest.approx(0.02)


def test_remove_cause_rejects_unknown_cause():
    mortality = pd.Series([0.02], index=[50])
    fractions = pd.DataFrame({"Cancer": [0.25]}, index=[50])

    with pytest.raises(ValueError, match="Cardiovascular"):
        remove_cause_from_lifetable(
            mortality,
            fractions,
            "Cardiovascular",
        )
