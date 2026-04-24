"""Functions for analyzing causes of death and modeling disease interventions."""

import numpy as np
import pandas as pd


def categorize_cause(icd_list):
    """
    Categorize ICD-10 codes into major cause categories.
    
    Uses the first code in the list (underlying cause of death).
    
    Parameters
    ----------
    icd_list : list or None
        List of ICD-10 codes, e.g., ['C349', 'F179']
    
    Returns
    -------
    str
        Major cause category
    
    Categories
    ----------
    - Cancer: Malignant neoplasms (C00-C97, D00-D48)
    - Cardiovascular: Diseases of the circulatory system (I00-I99)
    - Respiratory: Diseases of the respiratory system (J00-J99)
    - Endocrine: Endocrine, nutritional and metabolic diseases (E00-E89)
    - COVID-19: COVID-19 (U07)
    - Digestive: Diseases of the digestive system (K00-K93)
    - Genitourinary: Diseases of the genitourinary system (N00-N99)
    - Neurological: Diseases of the nervous system (G00-G99)
    - Mental: Mental and behavioral disorders (F00-F99)
    - Infectious: Certain infectious and parasitic diseases (A00-B99)
    - External: External causes of morbidity and mortality (V01-Y98)
    - Other: All other causes
    - Unknown: Missing or invalid data
    """
    if icd_list is None:
        return 'Unknown'
    
    try:
        if pd.isna(icd_list):
            return 'Unknown'
    except (TypeError, ValueError):
        pass
    
    if not isinstance(icd_list, list) or len(icd_list) == 0:
        return 'Unknown'
    
    code = icd_list[0]
    
    if not code or code == 'None' or pd.isna(code):
        return 'Unknown'
    
    first_letter = str(code)[0].upper()
    
    if first_letter == 'C' or (first_letter == 'D' and len(code) > 1 and code[1] in '01234'):
        return 'Cancer'
    elif first_letter == 'I':
        return 'Cardiovascular'
    elif first_letter == 'J':
        return 'Respiratory'
    elif first_letter == 'E':
        return 'Endocrine'
    elif first_letter == 'U' and str(code).startswith('U07'):
        return 'COVID-19'
    elif first_letter == 'K':
        return 'Digestive'
    elif first_letter == 'N':
        return 'Genitourinary'
    elif first_letter == 'G':
        return 'Neurological'
    elif first_letter == 'F':
        return 'Mental'
    elif first_letter in ['A', 'B']:
        return 'Infectious'
    elif first_letter in ['V', 'W', 'X', 'Y']:
        return 'External'
    else:
        return 'Other'


def load_cause_fractions(filepath='data/CDC/cause_fractions_total.csv'):
    """
    Load preprocessed cause-of-death fractions from CSV file.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the processed cause fractions file
        (default: 'data/CDC/cause_fractions_total.csv' for both sexes)
    
    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with causes as columns and age as index.
        Each cell contains the fraction of deaths from that cause at that age.
    
    Examples
    --------
    >>> # Load total (both sexes) cause fractions
    >>> cause_fractions = load_cause_fractions()
    
    >>> # Load male-specific cause fractions
    >>> male_fractions = load_cause_fractions('data/CDC/cause_fractions_male.csv')
    
    >>> # Load female-specific cause fractions
    >>> female_fractions = load_cause_fractions('data/CDC/cause_fractions_female.csv')
    
    >>> # Show fractions at age 70
    >>> print(cause_fractions.loc[70])
    """
    cause_fractions = pd.read_csv(filepath, index_col='age_years')
    return cause_fractions


def load_cause_fractions_from_raw(mmcd_filepath, exclude_covid=True, group_old_ages=True):
    """
    Load and calculate cause-of-death fractions from raw CDC MMCD data.
    
    DEPRECATED: Use load_cause_fractions() instead for processed data.
    
    This function is kept for backward compatibility but requires the large
    raw MMCD file. The preferred approach is to use preprocessed data.
    
    Parameters
    ----------
    mmcd_filepath : str
        Path to MMCD parquet file
    exclude_covid : bool, optional
        Whether to exclude COVID-19 deaths (default: True)
    group_old_ages : bool, optional
        Whether to group ages 105+ together (default: True)
    
    Returns
    -------
    pd.DataFrame
        Wide format DataFrame with causes as columns and age as index.
        Each cell contains the fraction of deaths from that cause at that age.
    """
    df = pd.read_parquet(mmcd_filepath, engine='fastparquet')
    df['cause_category'] = df['record_axis_conditions'].apply(categorize_cause)
    
    age_years_float = df['age_lower_bound'] / 1000 / 60 / 60 / 24 / 365.25
    df['age_years'] = np.floor(age_years_float).astype('Int64')
    
    df = df.dropna(subset=['age_years'])
    df = df[df['age_years'] != 998]
    
    df_grouped = df.copy()
    
    if exclude_covid:
        df_grouped = df_grouped[df_grouped['cause_category'] != 'COVID-19']
    
    if group_old_ages:
        df_grouped.loc[df_grouped['age_years'] >= 105, 'age_years'] = 105
    
    cause_by_age = df_grouped.groupby(['age_years', 'cause_category']).size()
    total_by_age = df_grouped.groupby('age_years').size()
    
    cause_fractions_wide = cause_by_age.unstack(fill_value=0).div(total_by_age, axis=0)
    
    # Sort columns by overall importance
    column_order = cause_fractions_wide.mean().sort_values(ascending=False).index
    cause_fractions_wide = cause_fractions_wide[column_order]
    
    return cause_fractions_wide


def remove_cause_from_lifetable(life_table_series, cause_fractions_df, cause_code):
    """
    Adjust a life table by removing deaths from a specific cause.
    
    This models the effect of "curing" a disease by removing its contribution
    to mortality at each age.
    
    Parameters
    ----------
    life_table_series : pd.Series
        Mortality rates (mx) indexed by age
    cause_fractions_df : pd.DataFrame
        Wide format DataFrame with index=age and columns=causes.
        Each cell is the fraction of deaths from that cause at that age.
    cause_code : str
        The cause category to remove (e.g., 'Cancer', 'Cardiovascular')
    
    Returns
    -------
    pd.Series
        Adjusted mortality rates with the specified cause removed
    
    Examples
    --------
    >>> # Model "curing cancer"
    >>> adjusted_rates = remove_cause_from_lifetable(
    ...     mortality_rates, cause_fractions, 'Cancer')
    
    Notes
    -----
    If death rate at age 70 is 0.02 (2% die per year) and 'Cancer' 
    accounts for 25% of deaths at that age:
    - Deaths from cancer: 0.02 * 0.25 = 0.005
    - Adjusted death rate: 0.02 * (1 - 0.25) = 0.015
    """
    adjusted_rates = life_table_series.copy()
    
    if cause_code not in cause_fractions_df.columns:
        raise ValueError(f"Cause code '{cause_code}' not found in dataframe columns")
    
    cause_fractions = cause_fractions_df[cause_code]
    common_ages = adjusted_rates.index.intersection(cause_fractions.index)
    max_cause_age = cause_fractions.index.max() if len(cause_fractions) > 0 else 0
    
    for age in adjusted_rates.index:
        if age in common_ages:
            fraction_from_cause = cause_fractions.loc[age]
        elif age > max_cause_age and max_cause_age > 0:
            fraction_from_cause = cause_fractions.loc[max_cause_age]
        else:
            fraction_from_cause = 0
        
        adjusted_rates.loc[age] = adjusted_rates.loc[age] * (1 - fraction_from_cause)
    
    return adjusted_rates

