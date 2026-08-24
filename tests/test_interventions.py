import pandas as pd
import pytest

from src.interventions import slow_aging, stop_aging


@pytest.fixture
def mortality():
    return pd.Series(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        index=[0, 1, 2, 3, 4, 5],
    )


def test_stop_aging_keeps_rates_unchanged_before_final_age(mortality):
    result = stop_aging(mortality, final_age=2)

    assert result.loc[0] == 0.01
    assert result.loc[1] == 0.02
    assert result.loc[2] == 0.03


def test_stop_aging_keeps_final_rate_after_final_age(mortality):
    result = stop_aging(mortality, final_age=2)

    assert result.loc[3] == 0.03
    assert result.loc[4] == 0.03


def test_stop_aging_can_pad_to_later_age(mortality):
    result = stop_aging(mortality, final_age=2, pad_to_age=6)

    assert list(result.index) == list(range(7))
    # Padding uses the last value after the intervention has been applied.
    assert result.loc[5] == 0.03
    assert result.loc[6] == 0.03


def test_slow_aging_preserves_rates_before_start_age(mortality):
    result = slow_aging(mortality, slow_factor=0.5, start_age=2)

    assert result.loc[0] == 0.01
    assert result.loc[1] == 0.02
    assert result.loc[2] == 0.03


def test_slow_aging_maps_ages_to_effective_biological_age(mortality):
    result = slow_aging(mortality, slow_factor=0.5, start_age=2)

    # age 3 -> biological age 2.5, interpolated between ages 2 and 3
    # age 4 -> biological age 3
    # age 5 -> biological age 3.5, interpolated between ages 3 and 4
    assert result.loc[3] == pytest.approx(0.035)
    assert result.loc[4] == pytest.approx(0.04)
    assert result.loc[5] == pytest.approx(0.045)


def test_slow_aging_with_zero_factor_freezes_at_start_age(mortality):
    result = slow_aging(mortality, slow_factor=0.0, start_age=2)

    assert result.loc[0] == pytest.approx(0.01)
    assert result.loc[1] == pytest.approx(0.02)
    assert result.loc[2:].tolist() == pytest.approx(
        [0.03, 0.03, 0.03, 0.03]
    )


def test_slow_aging_padding_uses_interpolated_terminal_rate(mortality):
    result = slow_aging(mortality, slow_factor=0.5, start_age=2, pad_to_age=7)

    assert result.loc[6] == pytest.approx(0.045)
    assert result.loc[7] == pytest.approx(0.045)
