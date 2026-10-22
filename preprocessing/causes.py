"""Functions for analyzing causes of death and modeling disease interventions.

The bucket → ICD-10 → GBD mapping is data-driven: see
``data/CDC/cause_categories.csv``. Edit that file to change which ICD-10
codes flow into which app bucket; the lookup logic below just consumes it.
"""

import functools
import os

import pandas as pd


_CATEGORIES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'CDC', 'cause_categories.csv',
)


@functools.lru_cache(maxsize=1)
def _load_categories():
    """Read the bucket → ICD-10 → GBD mapping CSV."""
    return pd.read_csv(_CATEGORIES_PATH)


@functools.lru_cache(maxsize=1)
def _icd10_prefix_lookup():
    """Build a {3-char ICD-10 prefix → bucket} dict from the mapping CSV.

    Each row of the CSV defines an inclusive 3-character ICD-10 range
    (e.g. E10-E14). We expand each range into individual prefixes so the
    runtime lookup is a single dict access.
    """
    df = _load_categories()
    lookup = {}
    for _, row in df.iterrows():
        start, end = str(row['icd10_start']).upper(), str(row['icd10_end']).upper()
        bucket = row['bucket']
        if len(start) != 3 or len(end) != 3:
            raise ValueError(f"ICD-10 range bounds must be 3 chars: {start}-{end}")
        for letter_ord in range(ord(start[0]), ord(end[0]) + 1):
            letter = chr(letter_ord)
            num_lo = int(start[1:3]) if letter == start[0] else 0
            num_hi = int(end[1:3]) if letter == end[0] else 99
            for n in range(num_lo, num_hi + 1):
                prefix = f'{letter}{n:02d}'
                if prefix in lookup and lookup[prefix] != bucket:
                    raise ValueError(
                        f"ICD-10 prefix {prefix} mapped to both "
                        f"{lookup[prefix]!r} and {bucket!r}")
                lookup[prefix] = bucket
    return lookup


def categorize_cause(icd_list):
    """Map an ICD-10 code list to its app bucket.

    Uses the first code in the list (underlying cause of death). The
    bucket comes from ``data/CDC/cause_categories.csv``; codes that don't
    match any range fall through to ``'Other'``. Empty / missing input
    returns ``'Unknown'``.
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
    prefix = str(code).upper()[:3]
    return _icd10_prefix_lookup().get(prefix, 'Other')


def bucket_labels():
    """Return ``{bucket → display label}`` from the mapping CSV."""
    df = _load_categories()
    return df.drop_duplicates('bucket').set_index('bucket')['bucket_label'].to_dict()


def bucket_order():
    """Return buckets in display order (per ``display_order`` column)."""
    df = _load_categories()
    return (df.drop_duplicates('bucket')
              .sort_values('display_order')['bucket'].tolist())


def bucket_gbd_info(bucket):
    """Return ``{cause_id, outline, name, level}`` for a bucket's primary GBD match."""
    df = _load_categories()
    rows = df[df['bucket'] == bucket]
    if len(rows) == 0:
        return None
    r = rows.iloc[0]
    return {
        'cause_id': int(r['gbd_cause_id']),
        'outline': r['gbd_outline'],
        'name': r['gbd_cause_name'],
        'level': int(r['gbd_level']),
    }


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

