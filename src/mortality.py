"""Functions for loading and working with mortality data."""

import pandas as pd


def load_mortality_rates(filepath='data/CDC/mortality_rates_total.csv'):
    """
    Load preprocessed mortality rates from CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the processed mortality rates file
        (default: 'data/CDC/mortality_rates_total.csv')

    Returns
    -------
    pd.Series
        Mortality rates (mx) indexed by age

    Examples
    --------
    >>> # Load total (both sexes) mortality rates
    >>> mortality_rates = load_mortality_rates()

    >>> # Load male-specific rates
    >>> male_rates = load_mortality_rates('data/CDC/mortality_rates_male.csv')

    >>> # Load female-specific rates
    >>> female_rates = load_mortality_rates('data/CDC/mortality_rates_female.csv')
    """
    mortality = pd.read_csv(filepath, index_col='Age')
    return mortality['mx']


# Legacy function for backward compatibility
def load_life_table(filepath, year_range='2010-2019'):
    """
    Load US life table data from raw CDC file.
    
    DEPRECATED: Use load_mortality_rates() instead for processed data.
    
    This function is kept for backward compatibility but requires raw data files.
    The preferred approach is to use preprocessed data with load_mortality_rates().
    
    Parameters
    ----------
    filepath : str
        Path to the raw life table file
    year_range : str, optional
        Year range to filter (default: '2010-2019')
    
    Returns
    -------
    pd.Series
        Mortality rates (mx) indexed by age
    """
    life_table = pd.read_csv(filepath, skiprows=2, sep=r'\s+')
    life_table = life_table[life_table['Age'] != '110+']
    life_table['Age'] = life_table['Age'].astype(int)
    life_table = life_table[life_table['Year'] == year_range].set_index('Age')
    return life_table['mx'].astype('float')

