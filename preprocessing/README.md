# Data Preprocessing

This directory contains scripts to preprocess raw data files into minimal processed files suitable for GitHub.

## Purpose

The raw data files (MMCD, life tables) are large and contain much more information than needed. These scripts extract only the essential data:

- **MMCD_2022.parquet** (~300 MB) → **cause_fractions.csv** (~15 KB)
- **life_table_*.txt** (~500 KB) → **mortality_rates_*.csv** (~3 KB each)

## Usage

### Initial Setup

1. Place raw data files in `raw_data/`:
   - `MMCD_2022.parquet` - CDC Multiple Cause of Death database
   - `life_table_total.txt` - US life table (both sexes)
   - `life_table_male.txt` - Male life table
   - `life_table_female.txt` - Female life table

2. Run preprocessing scripts:

```bash
cd preprocessing

# Process MMCD data
python process_mmcd.py

# Process life tables
python process_life_table.py
```

3. Processed files will be saved to `data/`:
   - `cause_fractions.csv` - Fraction of deaths by cause and age
   - `mortality_rates_total.csv` - Mortality rates (both sexes)
   - `mortality_rates_male.csv` - Male mortality rates
   - `mortality_rates_female.csv` - Female mortality rates

## What Gets Extracted

### From MMCD_2022.parquet
- Extracts only the **fraction of deaths** from each cause category at each age
- Categorizes ICD-10 codes into major categories (Cancer, Cardiovascular, etc.)
- Groups ages 105+ together for stability
- Excludes COVID-19 deaths (optional)

### From life_table_*.txt
- Extracts only the **mortality rate (mx)** column for year range 2010-2019
- Drops all other columns (qx, lx, dx, Lx, Tx, ex)
- Reduces file size by ~99%

## Scripts

- `process_mmcd.py` - Process Multiple Cause of Death data
- `process_life_table.py` - Process CDC life tables
- `README.md` - This file

## Output Format

### cause_fractions.csv
```
age_years,Cancer,Cardiovascular,External,...
0,0.0012,0.0034,0.0156,...
1,0.0023,0.0045,0.0089,...
...
```

### mortality_rates_*.csv
```
Age,mx
0,0.005823
1,0.000418
2,0.000253
...
```

## Notes

- Raw data files are NOT committed to git (in `.gitignore`)
- Processed files ARE committed (small, essential data only)
- Preprocessing is only needed once or when updating data sources

