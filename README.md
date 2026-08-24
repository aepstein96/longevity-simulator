# Longevity Simulator

A Python toolkit for modeling mortality rates and simulating the effects of longevity interventions.

## Overview

This project provides tools to explore "what-if" scenarios for human longevity by modeling:
- **Disease interventions**: What if we cured cancer? Or cardiovascular disease?
- **Aging interventions**: What if we could stop or slow biological aging?
- **Survival analysis**: Calculate survival curves and median lifespans under different scenarios

Uses real-world data from:
- CDC/NCHS United States Life Tables (2010-2019)
- CDC Multiple Cause of Death (MMCD) database (2022)

## Features

- **Load and analyze mortality data** from CDC life tables
- **Model aging interventions**:
  - Stop aging at a specific age
  - Slow the rate of biological aging
- **Model disease cures** by removing specific causes of death
- **Calculate survival curves** and median lifespans
- **Categorize causes of death** using ICD-10 codes

## Installation

```bash
pip install -r requirements.txt
```

## Tests

Run the test suite locally with:

```bash
python -m pytest -q
```

The suite covers aging interventions, survival calculations, cause-of-death
removal, and composition through `LongevityScenario`. Pull requests and pushes
to `main` run the same suite automatically in GitHub Actions.

## Quick Start

```python
from src import load_mortality_rates, stop_aging, calculate_survival_curve, calculate_median_lifespan

# Load preprocessed US mortality data
mortality_rates = load_mortality_rates('data/CDC/mortality_rates_total.csv')

# Model stopping aging at 25
adjusted_rates = stop_aging(mortality_rates, final_age=25, pad_to_age=200)

# Calculate survival curve
survival = calculate_survival_curve(adjusted_rates)

# Get median lifespan
median = calculate_median_lifespan(survival)
print(f"Median lifespan if aging stopped at 25: {median} years")
```

## Project Structure

```
longevity-simulator/
├── src/                      # Python modules
│   ├── mortality.py          # Load mortality data
│   ├── interventions.py      # Aging interventions
│   ├── causes.py             # Cause of death analysis
│   └── survival.py           # Survival curve calculations
├── data/                     # Processed data (committed to git)
│   ├── CDC/                  # Mortality rates and cause fractions
│   └── GBD/                  # Prevalence and cause hierarchy data
├── tests/                    # Unit and scenario-composition tests
├── raw_data/                 # Raw data files (not in git, ~300 MB)
│   ├── MMCD_2022.parquet
│   └── life_table_*.txt
├── preprocessing/            # Scripts to process raw data
│   ├── process_mmcd.py
│   ├── process_life_table.py
│   └── README.md
├── notebooks/                # Jupyter notebooks
│   └── demo.ipynb
├── results/                  # Output plots and figures
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Data Sources

- **Life Tables**: [CDC/NCHS United States Life Tables](https://www.cdc.gov/nchs/products/life_tables.htm)
- **Mortality Data**: [CDC Multiple Cause of Death Database](https://www.cdc.gov/nchs/nvss/mortality_public_use_data.htm)

### Data Preprocessing

The project uses **preprocessed data** stored in `data/` (small CSV files, ~25 KB total). This allows the repository to be easily shared on GitHub without large files.

To regenerate the processed data from raw sources:

1. Place raw data files in `raw_data/`
2. Run preprocessing scripts:
   ```bash
   cd preprocessing
   python process_life_table.py
   python process_mmcd.py
   ```

See `preprocessing/README.md` for details.

## Use Cases

1. **Research**: Model the impact of medical breakthroughs on population lifespan
2. **Education**: Visualize how aging and disease affect mortality
3. **Policy**: Estimate the demographic impact of public health interventions
4. **Curiosity**: Explore "what-if" scenarios for human longevity

## Future Development

- Additional intervention models (combination therapies, age-specific treatments)
- Comparative analysis across countries and time periods
- Economic impact modeling

## License

MIT License - see LICENSE file for details

## Author

Alexander Epstein, Cao Lab, Rockefeller University

