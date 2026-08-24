import numpy as np
import pandas as pd
import pytest

from src.scenarios import LongevityScenario


@pytest.fixture
def synthetic_scenario_data():
    mortality = pd.Series(
        [0.10, 0.20, 0.30, 0.40, 0.50],
        index=pd.Index(range(5), name="Age"),
        name="mx",
    )
    fractions = pd.DataFrame(
        {"Cancer": [0.20, 0.20, 0.20, 0.20, 0.20]},
        index=pd.Index(range(5), name="age_years"),
    )
    return mortality, fractions


@pytest.fixture
def patch_scenario_data(monkeypatch, synthetic_scenario_data):
    mortality, fractions = synthetic_scenario_data

    def load_data(self):
        self.baseline_mortality = mortality.copy()
        self.cause_fractions = fractions.copy()

    monkeypatch.setattr(LongevityScenario, "_load_data", load_data)


def test_no_intervention_preserves_mortality_and_survival(patch_scenario_data):
    scenario = LongevityScenario(aging_rate=1.0)
    data = scenario.get_data(pad_to=6)

    expected_mortality = pd.concat(
        [
            scenario.baseline_mortality,
            pd.Series(
                [0.50, 0.50],
                index=pd.Index([5, 6], name="Age"),
                name="mx",
            ),
        ]
    ).rename(None)
    pd.testing.assert_series_equal(
        data["baseline_mortality"], expected_mortality
    )
    pd.testing.assert_series_equal(
        data["intervention_mortality"], expected_mortality
    )
    pd.testing.assert_series_equal(
        data["baseline_survival"], data["intervention_survival"]
    )


def test_removed_cause_is_applied_to_intervention_only(patch_scenario_data):
    scenario = LongevityScenario(removed_causes=["Cancer"])
    data = scenario.get_data(pad_to=5)

    expected = pd.Series(
        [0.08, 0.16, 0.24, 0.32, 0.40, 0.40],
        index=pd.Index(range(6), name="Age"),
    ).rename(None)
    pd.testing.assert_series_equal(data["intervention_mortality"], expected)
    pd.testing.assert_series_equal(
        data["baseline_mortality"],
        pd.Series(
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.50],
            index=pd.Index(range(6), name="Age"),
        ),
    )
    assert np.all(
        data["intervention_survival"].to_numpy()
        >= data["baseline_survival"].to_numpy()
    )


def test_cause_removal_then_slow_aging_are_composed(patch_scenario_data):
    scenario = LongevityScenario(
        aging_rate=0.5,
        slow_aging_age=2,
        removed_causes=["Cancer"],
    )
    data = scenario.get_data(pad_to=6)

    # Cause removal happens first: [0.08, 0.16, 0.24, 0.32, 0.40].
    # Slow aging then maps chronological ages 3–6 to biological ages
    # 2.5 and 3 within the source range; ages beyond the source range use
    # the remapped terminal value during padding.
    expected = pd.Series(
        [0.08, 0.16, 0.24, 0.28, 0.32, 0.32, 0.32],
        index=pd.Index(range(7), name="Age"),
        name="mx",
    )
    pd.testing.assert_series_equal(
        data["intervention_mortality"],
        expected,
        check_exact=False,
        atol=1e-12,
    )
    assert data["intervention_survival"].iloc[-1] > 0
