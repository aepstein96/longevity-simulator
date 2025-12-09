"""Longevity Simulator - Tools for modeling mortality and longevity interventions."""

from .mortality import load_mortality_rates, load_life_table
from .interventions import stop_aging, slow_aging
from .causes import categorize_cause, remove_cause_from_lifetable, load_cause_fractions
from .survival import calculate_survival_curve, calculate_median_lifespan

__all__ = [
    'load_mortality_rates',
    'load_life_table',
    'stop_aging',
    'slow_aging',
    'categorize_cause',
    'remove_cause_from_lifetable',
    'load_cause_fractions',
    'calculate_survival_curve',
    'calculate_median_lifespan',
]

