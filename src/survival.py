"""Functions for calculating and analyzing survival curves."""

import numpy as np
import pandas as pd


def calculate_survival_curve(mortality_rates):
    """
    Calculate survival curve from mortality rates.
    
    The survival curve shows the fraction of people alive at each age,
    starting from a cohort of newborns.
    
    Parameters
    ----------
    mortality_rates : pd.Series or pd.DataFrame
        Mortality rates indexed by age. Can be a single Series or DataFrame
        with multiple scenarios as columns.
    
    Returns
    -------
    pd.Series or pd.DataFrame
        Survival probabilities at each age (fraction alive)
    
    Examples
    --------
    >>> survival = calculate_survival_curve(mortality_rates)
    >>> print(survival[50])  # Fraction alive at age 50
    0.96
    """
    if isinstance(mortality_rates, pd.Series):
        return np.cumprod(1 - mortality_rates)
    else:
        return pd.DataFrame({
            col: np.cumprod(1 - mortality_rates[col]) 
            for col in mortality_rates.columns
        })


def calculate_median_lifespan(survival_curve):
    """
    Calculate median lifespan from a survival curve.
    
    The median lifespan is the age at which 50% of the population has died.
    
    Parameters
    ----------
    survival_curve : pd.Series or pd.DataFrame
        Survival probabilities indexed by age
    
    Returns
    -------
    float or pd.Series
        Median lifespan in years. Returns Series if input is DataFrame.
    
    Examples
    --------
    >>> median = calculate_median_lifespan(survival_curve)
    >>> print(f"Median lifespan: {median} years")
    Median lifespan: 78.5 years
    """
    if isinstance(survival_curve, pd.Series):
        return survival_curve[survival_curve < 0.5].index.min()
    else:
        return pd.Series({
            col: survival_curve[survival_curve[col] < 0.5].index.min()
            for col in survival_curve.columns
        })

