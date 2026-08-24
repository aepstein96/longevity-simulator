import pandas as pd
import pytest

from src.survival import calculate_median_lifespan, calculate_survival_curve


def test_survival_curve_from_constant_mortality():
    mortality = pd.Series(
        [0.1, 0.1, 0.1],
        index=[0, 1, 2],
    )

    survival = calculate_survival_curve(mortality)

    expected = pd.Series(
        [0.9, 0.81, 0.729],
        index=[0, 1, 2],
    )

    pd.testing.assert_series_equal(survival, expected)


def test_survival_curve_supports_multiple_scenarios():
    mortality = pd.DataFrame(
        {
            "baseline": [0.1, 0.1],
            "intervention": [0.05, 0.05],
        },
        index=[0, 1],
    )

    survival = calculate_survival_curve(mortality)

    assert survival["baseline"].tolist() == pytest.approx([0.9, 0.81])
    assert survival["intervention"].tolist() == pytest.approx([0.95, 0.9025])


def test_median_lifespan_is_first_age_below_fifty_percent():
    survival = pd.Series(
        [1.0, 0.8, 0.6, 0.49, 0.3],
        index=[0, 1, 2, 3, 4],
    )

    assert calculate_median_lifespan(survival) == 3
