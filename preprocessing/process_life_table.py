"""
Preprocess life table data to extract mortality rates.

This extracts just the mortality rates we need from the CDC life tables.
"""

import pandas as pd


def process_life_table(input_path, output_path, year_range='2010-2019'):
    """
    Process CDC life table to extract mortality rates.
    
    Parameters
    ----------
    input_path : str
        Path to raw life table file
    output_path : str
        Path to save processed CSV file
    year_range : str
        Year range to extract
    """
    print(f"Loading life table from {input_path}...")
    life_table = pd.read_csv(input_path, skiprows=2, sep=r'\s+')
    print(f"  Loaded {len(life_table):,} rows")
    
    print(f"\nFiltering for year range: {year_range}")
    life_table = life_table[life_table['Age'] != '110+']
    life_table['Age'] = life_table['Age'].astype(int)
    life_table = life_table[life_table['Year'] == year_range]
    
    print("Extracting mortality rates (mx)...")
    mortality_rates = life_table.set_index('Age')['mx'].astype('float')
    
    print(f"\nSaving to {output_path}...")
    mortality_rates.to_csv(output_path, header=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Ages: {mortality_rates.index.min()} to {mortality_rates.index.max()}")
    print(f"Total records: {len(mortality_rates)}")
    
    import os
    file_size = os.path.getsize(output_path)
    print(f"\nFile size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    return mortality_rates


if __name__ == '__main__':
    # Process total (both sexes)
    print("Processing total life table...\n")
    process_life_table(
        '../raw_data/life_table_total.txt',
        '../data/mortality_rates_total.csv'
    )
    
    print("\n" + "="*60 + "\n")
    
    # Process male
    print("Processing male life table...\n")
    process_life_table(
        '../raw_data/life_table_male.txt',
        '../data/mortality_rates_male.csv'
    )
    
    print("\n" + "="*60 + "\n")
    
    # Process female
    print("Processing female life table...\n")
    process_life_table(
        '../raw_data/life_table_female.txt',
        '../data/mortality_rates_female.csv'
    )
    
    print("\n✓ All life tables processed!")

