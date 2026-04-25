"""Healthspan curve: P(no chronic conditions at age a).

Uses GBD prevalence data (one rate per cause × sex × age, in cases per
100,000) and assumes independence across conditions:

    P(healthy at age a) = ∏_k (1 − prevalence_k(a) / 100,000)

The bucket → GBD primary cause_id mapping comes from
``data/CDC/cause_categories.csv`` (the same CSV that drives bucket labels
and ICD-10 routing). For each app bucket included in the calculation, we
use the bucket's primary GBD cause_id to look up its prevalence series in
``data/GBD/prevalence_smoothed_single_year.csv``.

Independence is a simplification — in reality conditions co-occur (e.g.
diabetes ⊂ CKD ⊂ CV) — so this curve is **pessimistic** about how fast
"healthy" drops with age.
"""

import functools
import os

import numpy as np
import pandas as pd

from src import causes


_GBD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'GBD',
)

# Buckets to include in the default healthspan curve (chronic conditions
# whose primary GBD node is a meaningful chronic-disease aggregate).
DEFAULT_HEALTHSPAN_BUCKETS = (
    'Cardiovascular',
    'Cancer',
    'Neurological',
    'Chronic respiratory',
    'Diabetes & kidney disease',
)


@functools.lru_cache(maxsize=1)
def _load_prevalence():
    return pd.read_csv(
        os.path.join(_GBD_DIR, 'prevalence_smoothed_single_year.csv'))


def _gbd_sex(sex):
    """Translate the app's sex code into the GBD sex_name."""
    if sex in ('All', 'Both'):
        return 'Both'
    if sex == 'Male':
        return 'Male'
    if sex == 'Female':
        return 'Female'
    raise ValueError(f"Unknown sex: {sex!r}")


def get_bucket_prevalence(bucket, sex='All'):
    """Return prevalence (proportion 0-1) by single-year age for a bucket.

    Looks up the bucket's primary GBD cause_id from cause_categories.csv,
    pulls that prevalence series from the smoothed GBD data for the chosen
    sex, and converts the per-100,000 rate to a proportion.
    """
    info = causes.bucket_gbd_info(bucket)
    if info is None:
        raise ValueError(f"Unknown bucket: {bucket!r}")
    cause_id = info['cause_id']
    prev = _load_prevalence()
    series = prev[(prev['cause_id'] == cause_id) &
                  (prev['sex'] == _gbd_sex(sex))]
    if series.empty:
        raise ValueError(
            f"No prevalence data for cause_id={cause_id} sex={sex}. "
            f"Bucket {bucket!r} maps to GBD {info['outline']} "
            f"({info['name']})."
        )
    s = series.set_index('age')['val'].astype(float) / 100_000.0
    s.name = bucket
    return s.sort_index()


def compute_expected_condition_count(sex='All',
                                      buckets=DEFAULT_HEALTHSPAN_BUCKETS,
                                      pad_to=120):
    """Expected number of chronic conditions at each age.

    By linearity of expectation, E[count] = sum of per-condition prevalences,
    so this metric is unaffected by the independence assumption that biases
    the P(no chronic) curve. Returns a Series indexed by age.
    """
    series_per_bucket = {b: get_bucket_prevalence(b, sex=sex) for b in buckets}
    all_ages = sorted(set().union(*[s.index for s in series_per_bucket.values()]))
    df = pd.DataFrame(
        {b: s.reindex(all_ages).ffill().fillna(0)
         for b, s in series_per_bucket.items()},
        index=all_ages,
    )
    expected = df.sum(axis=1)
    expected.name = 'E[# chronic conditions]'

    max_age = int(expected.index.max())
    if pad_to > max_age:
        ext = pd.Series(expected.iloc[-1],
                        index=range(max_age + 1, pad_to + 1))
        expected = pd.concat([expected, ext])
    expected.index = expected.index.astype(int)
    return expected.sort_index()


def apply_aging_remap(series, aging_rate, start_age):
    """Rescale an age-indexed series along the chronological-age axis to
    reflect a slowed/accelerated/frozen aging intervention.

    For chronological age a >= start_age, biological age is
        b(a) = start_age + (a - start_age) * aging_rate
    and the rescaled value at a equals the baseline value at b(a)
    (linearly interpolated; clamped to the data's range at endpoints).
    For a < start_age, values are unchanged. ``aging_rate == 1.0`` is a
    no-op. ``aging_rate == 0.0`` freezes the series at its start_age value.

    Mirrors how ``interventions.slow_aging`` / ``stop_aging`` rescale
    mortality — applied here to any prevalence / healthspan curve.
    """
    if series is None or len(series) == 0 or aging_rate == 1.0:
        return None if series is None else series.copy()
    ages = series.index.to_numpy(dtype=float)
    vals = series.to_numpy(dtype=float)
    bio_age = np.where(ages < start_age,
                       ages,
                       start_age + (ages - start_age) * aging_rate)
    rescaled = np.interp(bio_age, ages, vals)
    return pd.Series(rescaled, index=series.index, name=series.name)


def compute_healthspan(sex='All', buckets=DEFAULT_HEALTHSPAN_BUCKETS,
                        pad_to=120):
    """Compute P(no chronic condition) by single-year age.

    Parameters
    ----------
    sex : 'All', 'Male', 'Female'
        'All' uses GBD's Both-sex aggregate.
    buckets : iterable of str
        App buckets to include. Each must have a primary GBD cause_id
        with prevalence data available.
    pad_to : int
        Hold the final value flat past the prevalence data's max age.
    """
    series_per_bucket = {b: get_bucket_prevalence(b, sex=sex) for b in buckets}

    # Align all series on the union of ages
    all_ages = sorted(set().union(*[s.index for s in series_per_bucket.values()]))
    df = pd.DataFrame(
        {b: s.reindex(all_ages).ffill().fillna(0)
         for b, s in series_per_bucket.items()},
        index=all_ages,
    )

    # P(healthy) = product over buckets of (1 - prevalence_k)
    not_have = (1.0 - df).clip(lower=0.0)
    healthspan = not_have.prod(axis=1)
    healthspan.name = 'P(no chronic condition)'

    max_age = int(healthspan.index.max())
    if pad_to > max_age:
        ext = pd.Series(healthspan.iloc[-1],
                        index=range(max_age + 1, pad_to + 1))
        healthspan = pd.concat([healthspan, ext])
    healthspan.index = healthspan.index.astype(int)
    return healthspan.sort_index()
