# Quick Start Guide

## Installation

1. Navigate to the project directory:
```bash
cd longevity-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Dash app

Start the dashboard locally:
```bash
python app.py
```

Then open `http://localhost:8080` in a browser.

## Using the Library

### Basic Example

```python
from src import load_mortality_rates, stop_aging, calculate_survival_curve, calculate_median_lifespan

# Load preprocessed mortality data
mortality_rates = load_mortality_rates('data/CDC/mortality_rates_total.csv')

# Model stopping aging at 25
adjusted_rates = stop_aging(mortality_rates, final_age=25, pad_to_age=200)

# Calculate survival and lifespan
survival = calculate_survival_curve(adjusted_rates)
median = calculate_median_lifespan(survival)

print(f"Median lifespan: {median} years")
```

### Disease Intervention Example

```python
from src import load_cause_fractions, remove_cause_from_lifetable

# Load cause of death data
cause_fractions = load_cause_fractions('data/CDC/cause_fractions_total.csv')

# Model curing cancer
mortality_no_cancer = remove_cause_from_lifetable(
    mortality_rates, cause_fractions, 'Cancer'
)
```

### Aging Intervention Example

```python
from src import slow_aging

# Slow aging by 20% starting at age 40
adjusted_rates = slow_aging(
    mortality_rates,
    slow_factor=0.8,
    start_age=40,
    pad_to_age=200
)
```

## Module Overview

### `src/mortality.py`
- `load_mortality_rates()` - Load preprocessed mortality rates
- `load_life_table()` - Load raw CDC life table data (legacy)

### `src/interventions.py`
- `stop_aging()` - Model stopping biological aging
- `slow_aging()` - Model slowing biological aging

### `src/causes.py`
- `categorize_cause()` - Categorize ICD-10 codes
- `load_cause_fractions()` - Load cause of death data
- `remove_cause_from_lifetable()` - Model curing diseases

### `src/survival.py`
- `calculate_survival_curve()` - Calculate survival probabilities
- `calculate_median_lifespan()` - Calculate median lifespan

## Data Files

### Processed Data (in `data/`, committed to git)
- `data/CDC/mortality_rates_total.csv` - US mortality rates (2010-2019, both sexes)
- `data/CDC/mortality_rates_male.csv` - Male mortality rates
- `data/CDC/mortality_rates_female.csv` - Female mortality rates
- `data/CDC/cause_fractions_total.csv` - Fraction of deaths by cause and age (both sexes)
- `data/CDC/cause_fractions_male.csv` - Male cause fractions
- `data/CDC/cause_fractions_female.csv` - Female cause fractions
- `data/GBD/` - GBD prevalence and cause-hierarchy data used for healthspan

### Raw Data (in `raw_data/`, NOT committed to git)
- `MMCD_2022.parquet` - Multiple cause of death data (~300 MB)
- `life_table_*.txt` - Raw CDC life tables
- `NCHS_life_expectancies.csv` - Historical life expectancy trends

To regenerate processed data from raw sources, see `preprocessing/README.md`

## Output

Results are saved to `results/` directory:
- `mortality_rates.png` - Visualization of mortality rates
- `survival_curves_comparison.png` - Main comparison plot
- `survival_curve_1.png`, `survival_curve_2.png`, etc. - Individual scenarios

## Next Steps

- Modify scenarios in the Dash UI
- Add new intervention models
- Create custom visualizations
- Add additional data sources and validation

