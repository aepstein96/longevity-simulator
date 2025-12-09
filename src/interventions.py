"""Functions for modeling aging interventions."""

import numpy as np
import pandas as pd


def stop_aging(true_mortality_rate, final_age, pad_to_age=0):
    """
    Model stopping biological aging at a specific age.
    
    After the specified age, mortality rate remains constant instead of
    continuing to increase exponentially.
    
    Parameters
    ----------
    true_mortality_rate : pd.Series
        Mortality rates indexed by age
    final_age : int
        Age at which to stop aging
    pad_to_age : int, optional
        Extend the series to this age (default: 0, no padding)
    
    Returns
    -------
    pd.Series
        Adjusted mortality rates with aging stopped
    
    Examples
    --------
    >>> # Stop aging at 25 - mortality rate stays at age 25 level forever
    >>> adjusted_rates = stop_aging(mortality_rates, final_age=25, pad_to_age=200)
    """
    out = true_mortality_rate.copy()
    out.index = out.index.astype(int)
    out = out.sort_index()
    
    current_max_age = out.index[-1]
    
    if final_age < current_max_age:
        final_mortality_rate = out[final_age]
        out[final_age:] = final_mortality_rate
    else:
        final_mortality_rate = out[current_max_age]
    
    if pad_to_age > current_max_age:
        final_death_rate = out.loc[current_max_age]
        out_pad_index = np.arange(current_max_age + 1, pad_to_age)
        out_pad = pd.Series(final_death_rate, index=out_pad_index)
        out = pd.concat([out, out_pad])
    
    return out


def slow_aging(true_mortality_rate, slow_factor, start_age=0, pad_to_age=0):
    """
    Model slowing the rate of biological aging.
    
    After the start age, biological aging proceeds at a slower rate. For example,
    with slow_factor=0.5, after 10 years you've only aged 5 years biologically.
    
    Parameters
    ----------
    true_mortality_rate : pd.Series
        Mortality rates indexed by age
    slow_factor : float
        Factor to slow aging by (e.g., 0.5 means age half as fast)
    start_age : int, optional
        Age at which to begin slowing aging (default: 0)
    pad_to_age : int, optional
        Extend the series to this age (default: 0, no padding)
    
    Returns
    -------
    pd.Series
        Adjusted mortality rates with slowed aging
    
    Examples
    --------
    >>> # Slow aging by 20% starting at age 40
    >>> adjusted_rates = slow_aging(mortality_rates, slow_factor=0.8, 
    ...                              start_age=40, pad_to_age=200)
    
    Notes
    -----
    If start_age=30 and slow_factor=0.5:
    - Ages 0-29: normal mortality rates
    - Age 30: mortality rate of age 30
    - Age 40: mortality rate of age 30 + (40-30)*0.5 = age 35
    - Age 60: mortality rate of age 30 + (60-30)*0.5 = age 45
    """
    out = true_mortality_rate.copy()
    out.index = out.index.astype(int)
    out = out.sort_index()

    min_age = out.index.min()
    max_age = out.index.max()

    # Map each age to its effective biological age
    new_index = out.index
    mapped_ages = np.zeros(len(new_index), dtype=int)
    
    for i, age in enumerate(new_index):
        if age < start_age:
            mapped_ages[i] = age
        else:
            mapped_ages[i] = int(start_age + (age - start_age) * slow_factor)
    
    # Clip to valid range
    mapped_ages = np.clip(mapped_ages, min_age, max_age)
    slowed_rates = pd.Series([out.loc[age] for age in mapped_ages], index=new_index)

    # Pad if requested
    if pad_to_age > max_age:
        final_death_rate = slowed_rates.iloc[-1]
        pad_index = np.arange(max_age + 1, pad_to_age + 1)
        pad_vals = pd.Series(final_death_rate, index=pad_index)
        slowed_rates = pd.concat([slowed_rates, pad_vals])

    return slowed_rates

