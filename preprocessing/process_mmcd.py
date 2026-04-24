"""
Preprocess MMCD (Multiple Cause of Death) data to extract cause fractions.

This reduces the large raw MMCD file (~hundreds of MB) to a small CSV (~few KB)
containing only the fraction of deaths from each cause category at each age.
"""

import numpy as np
import pandas as pd


def categorize_cause(icd_list):
    """Categorize ICD-10 codes into major cause categories."""
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


def process_mmcd(input_path, output_path, exclude_covid=True, group_old_ages=True, sex=None):
    """
    Process MMCD data to extract cause fractions by age.
    
    Parameters
    ----------
    input_path : str
        Path to raw MMCD parquet file
    output_path : str
        Path to save processed CSV file
    exclude_covid : bool
        Whether to exclude COVID-19 deaths
    group_old_ages : bool
        Whether to group ages 105+ together
    sex : str, optional
        Filter by sex: 'M' for male, 'F' for female, None for total
    """
    sex_label = {'M': 'male', 'F': 'female', None: 'total'}[sex]
    print(f"Loading MMCD data ({sex_label})...")
    df = pd.read_parquet(input_path, engine='fastparquet')
    print(f"  Loaded {len(df):,} death records")
    
    # Filter by sex if specified
    if sex is not None:
        df = df[df['sex'] == sex]
        print(f"  Filtered to {len(df):,} {sex_label} death records")
    
    print("\nCategorizing causes of death...")
    df['cause_category'] = df['record_axis_conditions'].apply(categorize_cause)
    
    print("Converting ages...")
    age_years_float = df['age_lower_bound'] / 1000 / 60 / 60 / 24 / 365.25
    df['age_years'] = np.floor(age_years_float).astype('Int64')
    
    df = df.dropna(subset=['age_years'])
    df = df[df['age_years'] != 998]
    
    df_grouped = df.copy()
    
    if exclude_covid:
        print("Excluding COVID-19 deaths...")
        df_grouped = df_grouped[df_grouped['cause_category'] != 'COVID-19']
    
    if group_old_ages:
        print("Grouping ages 105+ together...")
        df_grouped.loc[df_grouped['age_years'] >= 105, 'age_years'] = 105
    
    print("\nCalculating cause fractions...")
    cause_by_age = df_grouped.groupby(['age_years', 'cause_category']).size()
    total_by_age = df_grouped.groupby('age_years').size()
    
    cause_fractions_wide = cause_by_age.unstack(fill_value=0).div(total_by_age, axis=0)
    
    # Sort columns by overall importance
    column_order = cause_fractions_wide.mean().sort_values(ascending=False).index
    cause_fractions_wide = cause_fractions_wide[column_order]
    
    print(f"\nSaving to {output_path}...")
    cause_fractions_wide.to_csv(output_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Output shape: {cause_fractions_wide.shape}")
    print(f"Ages: {cause_fractions_wide.index.min()} to {cause_fractions_wide.index.max()}")
    print(f"Causes tracked: {list(cause_fractions_wide.columns)}")
    
    import os
    file_size = os.path.getsize(output_path)
    print(f"\nFile size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"Compression ratio: {len(df) * 100 / file_size:.1f}x")
    
    return cause_fractions_wide


if __name__ == '__main__':
    input_file = '../raw_data/MMCD_2022.parquet'
    
    # Process total (both sexes)
    print("="*60)
    print("PROCESSING TOTAL (BOTH SEXES)")
    print("="*60)
    process_mmcd(input_file, '../data/CDC/cause_fractions_total.csv', sex=None)

    # Process male
    print("\n" + "="*60)
    print("PROCESSING MALE")
    print("="*60)
    process_mmcd(input_file, '../data/CDC/cause_fractions_male.csv', sex='M')

    # Process female
    print("\n" + "="*60)
    print("PROCESSING FEMALE")
    print("="*60)
    process_mmcd(input_file, '../data/CDC/cause_fractions_female.csv', sex='F')
    
    print("\n✓ All processing complete!")

