"""Bundle the runtime CSVs in data/ into web/data.json for the static site.

Run from the repo root whenever data/ changes:
    python preprocessing/export_web_data.py
"""
import json

import pandas as pd

import causes

SEXES = {'All': 'total', 'Male': 'male', 'Female': 'female'}
GBD_SEX = {'All': 'Both', 'Male': 'Male', 'Female': 'Female'}

# Location dropdown: (key, GBD location_name, label). The first entry is the default and uses the
# top-level CDC series; the rest come from data/GBD/regions/ (preprocessing/process_gbd_regions.py).
LOCATIONS = [
    ('US-CDC', None, 'United States (CDC data)'),
    ('US-GBD', 'United States of America', 'United States (GBD data)'),
    ('Global', 'Global', 'Global'),
    ('WB-High', 'World Bank High Income', 'High-income countries'),
    ('WB-UpperMid', 'World Bank Upper Middle Income', 'Upper-middle-income countries'),
    ('WB-LowerMid', 'World Bank Lower Middle Income', 'Lower-middle-income countries'),
    ('WB-Low', 'World Bank Low Income', 'Low-income countries'),
    ('CentralAsia', 'Central Asia', 'Central Asia'),
    ('EU', 'European Union', 'European Union'),
    ('LatAm', 'Latin America and Caribbean', 'Latin America & Caribbean'),
    ('NAME', 'North Africa and Middle East', 'North Africa & Middle East'),
    ('SouthAsia', 'South Asia', 'South Asia'),
    ('SEAsia', 'Southeast Asia, East Asia, and Oceania', 'Southeast Asia, East Asia & Oceania'),
    ('SSA', 'Sub-Saharan Africa', 'Sub-Saharan Africa'),
]

# Chronic-condition buckets summed into the healthspan curve. E[# conditions] = sum of
# prevalences (linearity of expectation), so the web app only needs each bucket's series.
HEALTHSPAN_BUCKETS = (
    'Cardiovascular',
    'Cancer',
    'Neurological',
    'Chronic respiratory',
    'Diabetes & kidney disease',
)


def bucket_prevalence(prev, bucket, sex):
    """Prevalence (proportion 0-1) by single-year age for a bucket's primary GBD cause."""
    cause_id = causes.bucket_gbd_info(bucket)['cause_id']
    rows = prev[(prev['cause_id'] == cause_id) & (prev['sex'] == GBD_SEX[sex])]
    if rows.empty:
        raise ValueError(f"No prevalence data for {bucket!r} (cause_id={cause_id}, sex={sex})")
    return (rows.set_index('age')['val'].astype(float) / 100_000.0).sort_index()


def region_series(loc, buckets):
    """{mortality, fractions, prevalence} for one GBD location, shaped like the top-level CDC data."""
    mort = pd.read_csv('data/GBD/regions/mortality_rates.csv')
    fr = pd.read_csv('data/GBD/regions/cause_fractions.csv')
    prev = pd.read_csv('data/GBD/regions/prevalence_smoothed_single_year.csv')
    mort, fr, prev = (d[d['location'] == loc] for d in (mort, fr, prev))
    out = {'mortality': {}, 'fractions': {}, 'prevalence': {}}
    for sex in SEXES:
        m = mort[mort['sex'] == GBD_SEX[sex]].sort_values('age')
        f = fr[fr['sex'] == GBD_SEX[sex]].sort_values('age_years')
        out['mortality'][sex] = m['mx'].round(8).tolist()
        out['fractions'][sex] = {c: f[c].round(6).tolist() for c in buckets}
        out['prevalence'][sex] = {
            b: bucket_prevalence(prev, b, sex).round(6).tolist() for b in HEALTHSPAN_BUCKETS
        }
    return out


def main():
    labels = causes.bucket_labels()
    cols = pd.read_csv('data/CDC/cause_fractions_total.csv', nrows=1).columns
    available = [c for c in cols if c not in ('age_years', 'Unknown')]
    order = causes.bucket_order()
    ordered = [c for c in order if c in available] + [c for c in available if c not in order]
    prev = pd.read_csv('data/GBD/prevalence_smoothed_single_year.csv')

    out = {
        'causes': [{'key': c, 'label': labels.get(c, c)} for c in ordered],
        'healthspanBuckets': list(HEALTHSPAN_BUCKETS),
        'mortality': {}, 'fractions': {}, 'prevalence': {},
    }
    for sex, suffix in SEXES.items():
        # Arrays are indexed by age, starting at 0 (all source files are contiguous from 0).
        out['mortality'][sex] = pd.read_csv(
            f'data/CDC/mortality_rates_{suffix}.csv', index_col='Age')['mx'].round(8).tolist()
        fr = pd.read_csv(f'data/CDC/cause_fractions_{suffix}.csv', index_col='age_years')
        out['fractions'][sex] = {c: fr[c].round(8).tolist() for c in fr.columns}
        out['prevalence'][sex] = {
            b: bucket_prevalence(prev, b, sex).round(8).tolist() for b in HEALTHSPAN_BUCKETS
        }

    out['locations'] = [{'key': k, 'label': label} for k, _, label in LOCATIONS]
    out['regions'] = {k: region_series(loc, ordered) for k, loc, _ in LOCATIONS if loc}

    with open('web/data.json', 'w') as f:
        json.dump(out, f, separators=(',', ':'))


if __name__ == '__main__':
    main()
