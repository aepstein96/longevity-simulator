"""
PCHIP-interpolate GBD incidence and death rates from 5-year age bands to
single-year ages. Drops aggregate bands like '<5 years' that overlap with
finer bands. Output files mirror the input schema but with integer `age`
column replacing `age` (band string).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


# Representative age for each GBD age band. `None` means the band is an
# aggregate/overlapping one and should be dropped from the fit.
AGE_BAND_TO_MIDPOINT = {
    '<1 year':      0.5,
    '0-6 days':     None,   # neonatal aggregate, overlaps <1 year
    '7-27 days':    None,
    '1-5 months':   None,
    '6-11 months':  None,
    '<5 years':     None,   # overlaps with <1 year and 2-4 years
    '1-4 years':    None,   # overlaps with 2-4 years
    '2-4 years':    3,
    '5-9 years':    7,
    '10-14 years':  12,
    '15-19 years':  17,
    '20-24 years':  22,
    '25-29 years':  27,
    '30-34 years':  32,
    '35-39 years':  37,
    '40-44 years':  42,
    '45-49 years':  47,
    '50-54 years':  52,
    '55-59 years':  57,
    '60-64 years':  62,
    '65-69 years':  67,
    '70-74 years':  72,
    '75-79 years':  77,
    '80-84 years':  82,
    '85-89 years':  87,
    '90-94 years':  92,
    '95+ years':    97,
}


def smooth_long(df: pd.DataFrame, single_year_ages: np.ndarray) -> pd.DataFrame:
    """
    Interpolate a long-format GBD table onto integer ages via PCHIP per
    (sex, cause) series. Non-negative clipping; edge bands extrapolate as
    constant. Returns a long-format DataFrame with columns:
    sex, cause, age, val, upper, lower.
    """
    df = df.copy()
    df['age_mid'] = df['age'].map(AGE_BAND_TO_MIDPOINT)
    df = df.dropna(subset=['age_mid'])

    out_rows = []
    for (sex, cause), grp in df.groupby(['sex', 'cause']):
        grp = (grp.sort_values('age_mid')
                  .drop_duplicates('age_mid', keep='first'))
        if len(grp) < 2:
            continue
        xs = grp['age_mid'].to_numpy(dtype=float)

        results = {'age': single_year_ages}
        for col in ('val', 'upper', 'lower'):
            ys = grp[col].to_numpy(dtype=float)
            # PCHIP fit on band midpoints
            fit = PchipInterpolator(xs, ys, extrapolate=False)
            vals = fit(single_year_ages)
            # Edges: fill with the terminal band value
            vals[single_year_ages < xs[0]]  = ys[0]
            vals[single_year_ages > xs[-1]] = ys[-1]
            # Incidence/death rates can't be negative
            vals = np.maximum(vals, 0.0)
            results[col] = vals

        for age, v, u, l in zip(single_year_ages, results['val'],
                                 results['upper'], results['lower']):
            out_rows.append({'sex': sex, 'cause': cause,
                             'age': int(age), 'val': v,
                             'upper': u, 'lower': l})
    return pd.DataFrame(out_rows)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    gbd_dir = root / 'data' / 'GBD'
    single_year_ages = np.arange(0, 101)

    pairs = [
        ('incidence', 'incidence_by_age_sex_cause.csv',
                      'incidence_smoothed_single_year.csv'),
        ('deaths',    'deaths_by_age_sex_cause.csv',
                      'deaths_smoothed_single_year.csv'),
    ]
    for label, src_name, dst_name in pairs:
        src = gbd_dir / src_name
        if not src.exists():
            print(f'SKIP {label}: {src} missing')
            continue
        df = pd.read_csv(src)
        smooth = smooth_long(df, single_year_ages)
        dst = gbd_dir / dst_name
        smooth.to_csv(dst, index=False)
        print(f'Wrote {dst}: {len(smooth):>5} rows, '
              f'{smooth["cause"].nunique()} causes, '
              f'{smooth["sex"].nunique()} sexes, '
              f'ages {smooth["age"].min()}-{smooth["age"].max()}')


if __name__ == '__main__':
    main()
