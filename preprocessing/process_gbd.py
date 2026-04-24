"""
Consolidate raw IHME GBD Results Tool exports into tidy per-measure CSVs.

The GBD Results Tool produces one CSV per query, each with a stem like
`IHME-GBD_2023_DATA-<hash>-1.csv`. This script reads every such file under
`data/GBD/raw/`, drops redundant constant columns (location, year, metric,
measure, population_group), and writes:

- data/GBD/incidence_by_age_sex_cause.csv   (measure == 'Incidence')
- data/GBD/deaths_by_age_sex_cause.csv      (measure == 'Deaths')

Tidy schema: sex, age, cause, val, upper, lower.
"""

from pathlib import Path

import pandas as pd


AGE_BAND_ORDER = [
    '<1 year', '0-6 days', '7-27 days', '1-5 months', '6-11 months',
    '<5 years', '1-4 years', '2-4 years',
    '5-9 years', '10-14 years', '15-19 years', '20-24 years',
    '25-29 years', '30-34 years', '35-39 years', '40-44 years',
    '45-49 years', '50-54 years', '55-59 years', '60-64 years',
    '65-69 years', '70-74 years', '75-79 years', '80-84 years',
    '85-89 years', '90-94 years', '95+ years',
]


def _age_sort_key(band: str) -> int:
    try:
        return AGE_BAND_ORDER.index(band)
    except ValueError:
        return len(AGE_BAND_ORDER) + hash(band) % 1000


def consolidate(raw_dir: Path, out_dir: Path) -> None:
    raw_files = sorted(raw_dir.glob('IHME-GBD_*.csv'))
    if not raw_files:
        raise SystemExit(f'No IHME-GBD_*.csv files in {raw_dir}')

    print(f'Loading {len(raw_files)} raw file(s):')
    frames = []
    for f in raw_files:
        df = pd.read_csv(f)
        print(f'  {f.name}: {len(df):>6} rows, measure={df["measure"].unique().tolist()}, '
              f'sex={df["sex"].unique().tolist()}')
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    expected_constants = {
        'location': 'United States of America',
        'metric': 'Rate',
        'population_group': 'All Population',
        'year': 2023,
    }
    for col, expected in expected_constants.items():
        vals = combined[col].unique()
        if len(vals) != 1 or vals[0] != expected:
            print(f'  WARNING: column {col} is not constant {expected!r}: {vals}')

    tidy = combined[['measure', 'sex', 'age', 'cause', 'val', 'upper', 'lower']]
    tidy = tidy.drop_duplicates(subset=['measure', 'sex', 'age', 'cause'], keep='first')

    for measure, slug in [('Incidence', 'incidence'), ('Deaths', 'deaths')]:
        sub = tidy[tidy['measure'] == measure].drop(columns='measure').copy()
        if sub.empty:
            print(f'\nNo rows for measure={measure}, skipping.')
            continue
        sub['_age_sort'] = sub['age'].map(_age_sort_key)
        sub = (sub.sort_values(['sex', 'cause', '_age_sort'])
                  .drop(columns='_age_sort')
                  .reset_index(drop=True))
        out = out_dir / f'{slug}_by_age_sex_cause.csv'
        sub.to_csv(out, index=False)
        print(f'\nWrote {out}: {len(sub)} rows, '
              f'{sub["cause"].nunique()} causes, '
              f'{sub["age"].nunique()} age bands, '
              f'sexes={sorted(sub["sex"].unique())}')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    consolidate(
        raw_dir=root / 'data' / 'GBD' / 'raw',
        out_dir=root / 'data' / 'GBD',
    )
