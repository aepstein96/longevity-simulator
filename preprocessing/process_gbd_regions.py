"""Build per-location mortality, cause fractions and prevalence from a GBD 2023 Results-tool export.

Input: one export with Measure = Deaths + Prevalence, Metric = Rate, Year = 2023, Sex = Male/Female/Both,
ages <1 / 12-23 months / 2-4 / 5-9 ... 95+, and causes: All causes, the level-1/2 causes behind each
app bucket (see data/CDC/cause_categories.csv), plus COVID-19.

Run from the repo root:
    python preprocessing/process_gbd_regions.py [raw_data/GBD/IHME-GBD_2023_DATA-6d3a2bd3-1.csv]

Writes data/GBD/regions/{mortality_rates,cause_fractions,prevalence_smoothed_single_year}.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causes  # noqa: E402
from smooth_gbd import FINEGRAINED_MIDPOINTS, smooth_long  # noqa: E402

DEFAULT_INPUT = 'raw_data/GBD/IHME-GBD_2023_DATA-6d3a2bd3-1.csv'
OUT_DIR = 'data/GBD/regions'
MAX_MX_AGE = 109      # matches the CDC life table
MAX_FRACTION_AGE = 105  # matches the CDC cause-fraction table
ALL_CAUSES, COVID = 294, 1048
# Bucket -> GBD cause. "Other" is the remainder of all causes.
FRACTION_BUCKETS = {
    'External': 687, 'Cardiovascular': 491, 'Cancer': 410, 'Neurological': 542, 'Infectious': 295,
    'Chronic respiratory': 508, 'Diabetes & kidney disease': 974, 'Digestive': 526,
}
AGE_START = {'<1 year': 0, '12-23 months': 1, '2-4 years': 2}


def age_start(name):
    return AGE_START[name] if name in AGE_START else int(name.split()[0].split('-')[0].rstrip('+'))


def single_year_mx(rates):
    """{group start age: deaths per person-year} -> mx for ages 0..MAX_MX_AGE.

    Ages 0-4 take their group's rate; 5-94 interpolate log(rate) between 5-year group midpoints;
    95+ follows a Kannisto (logistic) curve fit to the 80-94 groups, the same old-age model the
    Human Mortality Database (our US source) uses.
    """
    mx = np.empty(MAX_MX_AGE + 1)
    mx[0], mx[1], mx[2:5] = rates[0], rates[1], rates[2]
    starts = np.array([a for a in sorted(rates) if 5 <= a <= 90])
    vals = np.array([rates[a] for a in starts])
    ages = np.arange(5, 95)
    mx[5:95] = np.exp(np.interp(ages + 0.5, starts + 2.5, np.log(vals)))
    old = starts >= 80
    b, a = np.polyfit(starts[old] + 2.5, np.log(vals[old] / (1 - vals[old])), 1)
    tail = np.arange(95, MAX_MX_AGE + 1) + 0.5
    mx[95:] = 1 / (1 + np.exp(-(a + b * tail)))
    return mx


def bucket_fractions(deaths):
    """deaths: rows of one location+sex. -> DataFrame (age 0..MAX_FRACTION_AGE) x bucket."""
    w = deaths.pivot_table(index='age', columns='cause_id', values='val', aggfunc='sum')
    total = w[ALL_CAUSES] - w[COVID]  # COVID-19 excluded, as in process_mmcd.py
    out = pd.DataFrame({b: w[c] for b, c in FRACTION_BUCKETS.items()})
    out['Infectious'] -= w[COVID]
    out['Other'] = (total - out.sum(axis=1)).clip(lower=0)
    out = out.div(out.sum(axis=1), axis=0)
    # Step out to single years: each age takes its group's fractions.
    ages = np.arange(MAX_FRACTION_AGE + 1)
    idx = out.index.to_numpy()
    return out.iloc[np.searchsorted(idx, ages, side='right') - 1].set_index(pd.Index(ages, name='age_years'))


def main(path=DEFAULT_INPUT):
    g = pd.read_csv(path)
    assert set(g.metric_name) == {'Rate'}, 'expected Metric = Rate'
    g = g[g.age_name.isin(list(AGE_START)) | g.age_name.str.match(r'^\d+(-\d+|\+) years$')]
    g = g.assign(age=g.age_name.map(age_start), val=g.val / 1e5)  # per 100k -> per person
    deaths = g[g.measure_name == 'Deaths']
    prev = g[g.measure_name == 'Prevalence']

    mort, fracs = [], []
    for (loc, sex), d in deaths.groupby(['location_name', 'sex_name']):
        allc = d[d.cause_id == ALL_CAUSES].set_index('age')['val']
        mort.append(pd.DataFrame({'location': loc, 'sex': sex, 'age': range(MAX_MX_AGE + 1),
                                  'mx': single_year_mx(allc.to_dict()).round(8)}))
        fr = bucket_fractions(d).round(6).reset_index()
        fracs.append(fr.assign(location=loc, sex=sex))

    hs_ids = {causes.bucket_gbd_info(b)['cause_id'] for b in ('Cardiovascular', 'Cancer', 'Neurological',
                                                              'Chronic respiratory', 'Diabetes & kidney disease')}
    p = prev[prev.cause_id.isin(hs_ids)].rename(columns={'sex_name': 'sex'})
    smoothed = []
    for loc, d in p.groupby('location_name'):
        d = d.assign(age=d.age_name, val=d.val * 1e5, upper=d.upper, lower=d.lower,
                     parent_id=0, level=2)  # smooth_long works in per-100k, carries these columns
        smoothed.append(smooth_long(d, np.arange(0, 101), FINEGRAINED_MIDPOINTS).assign(location=loc))

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.concat(mort).to_csv(f'{OUT_DIR}/mortality_rates.csv', index=False)
    cols = ['location', 'sex', 'age_years', *FRACTION_BUCKETS, 'Other']
    pd.concat(fracs)[cols].to_csv(f'{OUT_DIR}/cause_fractions.csv', index=False)
    pd.concat(smoothed)[['location', 'cause_id', 'sex', 'age', 'val']].round({'val': 3}).to_csv(
        f'{OUT_DIR}/prevalence_smoothed_single_year.csv', index=False)
    print(f'Wrote {OUT_DIR}/ for {deaths.location_name.nunique()} locations')


if __name__ == '__main__':
    main(*sys.argv[1:])
