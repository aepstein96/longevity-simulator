# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Longevity Simulator — a Python library (in `src/`) plus a Dash web UI (`app.py`) for modeling the effect of disease cures and aging-slowdown interventions on survival curves and median lifespan. Data comes from CDC/NCHS life tables and the CDC Multiple Cause of Death (MMCD) database.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run the Dash app locally (listens on 0.0.0.0:8080)
python app.py
# ...or via gunicorn like production does:
gunicorn -b 0.0.0.0:8080 app:server

# Regenerate processed CSVs in data/ from raw_data/ (only needed when raw data changes)
cd preprocessing && python process_life_table.py && python process_mmcd.py
```

Deployment: a `Dockerfile` builds the app; `fly.toml` deploys it to Fly.io (`fly deploy`), and `Procfile` supports Heroku-style hosts. All three target the same `app:server` gunicorn entrypoint on port 8080.

There is no test suite, lint config, or CI in this repo.

## Architecture

The code is split into a **stateless functional library** under `src/` and a **scenario orchestrator** (`src/scenarios.py`) that the Dash app uses. Understanding the data flow matters more than any individual module:

1. **Load** — `mortality.load_mortality_rates()` returns a `pd.Series` of annual mortality rates `mx` indexed by integer age, from `data/mortality_rates_{total,male,female}.csv`. `causes.load_cause_fractions()` returns a `pd.DataFrame` indexed by age with one column per ICD-10-derived category (`Cancer`, `Cardiovascular`, `External`, …) holding the fraction of deaths at that age attributable to the cause.
2. **Remove causes** — `causes.remove_cause_from_lifetable(mx, fractions, 'Cancer')` multiplies each age's mortality by `(1 - fraction_from_cause)`. Ages beyond the fractions table reuse the last available age's fraction. Apply once per cause to stack cures.
3. **Apply aging intervention** — `interventions.stop_aging(mx, final_age)` flatlines mortality past `final_age`. `interventions.slow_aging(mx, slow_factor, start_age)` remaps each post-`start_age` year onto a fractional biological age (e.g. `slow_factor=0.5` means you age half as fast). Both accept `pad_to_age` to extend the series with the last value — needed because survival curves otherwise terminate at age 100.
4. **Survive** — `survival.calculate_survival_curve(mx)` is just `cumprod(1 - mx)`. `calculate_median_lifespan()` returns the first age where survival drops below 0.5.
5. **Fit (optional)** — `fitting.fit_gompertz()` fits `M(t) = a·exp(b·t)` (or with a Makeham constant) on a user-specified age window, always in log-space via `scipy.optimize.curve_fit`.

**`LongevityScenario` in `src/scenarios.py` is the single integration point** between the UI and the library. It owns a `(sex, aging_rate, slow_aging_age, removed_causes)` tuple, loads the right CSVs in `_load_data()`, and exposes two methods:
- `get_data(pad_to)` → dict with `baseline_mortality`, `baseline_survival`, `intervention_mortality`, `intervention_survival` (Series indexed by age, padded to `pad_to`).
- `fit_curve(target, remove_accidents, use_makeham, fit_region)` → Gompertz fit for either the baseline or the intervention curve.

The important convention: `aging_rate` in the scenario is a multiplier where `1.0` is normal, `0.0` triggers the `stop_aging` branch, and any other value feeds into `slow_aging` as `slow_factor`. The Dash UI exposes this as a **percent** (`aging_rate_percent / 100`).

**`app.py` is a single-file Dash app** with one big `update_dashboard` callback fanning out to seven outputs (two Plotly figures and five KPI/equation strings). When "Remove accidents for fit" is checked it also has to rebuild the intervention mortality series with `External` added to `removed_causes` for the scatter plot — that's why the callback instantiates a `LongevityScenario` twice in some branches. Keep this symmetry in mind when editing the callback: the fit and the plotted mortality series must show the same cause-removal state.

## Data layout

- `data/` — small preprocessed CSVs (~25 KB total), **committed** to git. Loaded at runtime.
- `raw_data/` — large source files (~300 MB, MMCD parquet + raw life tables), **gitignored**. Only needed to regenerate `data/`.
- `preprocessing/` — one-off scripts that read `raw_data/` and write `data/`. Not imported by the app.
- `notebooks/` is gitignored; don't assume notebooks exist.

Sex handling: the UI sends `'All' | 'Male' | 'Female'`, `LongevityScenario.sex_map` translates to the file suffix `total | male | female`.

## Conventions

- All ages are integer years; mortality/survival/cause-fraction series use age as the index.
- Cause categories are the 13 strings produced by `causes.categorize_cause()` from ICD-10 prefixes — any new cause-removal feature must go through that categorization (not raw ICD codes).
- `pad_to_age=0` in the intervention helpers means "don't pad"; the scenario uses this when fitting (no padding) and `pad_to=120` when plotting.
- COVID-19 deaths are excluded from cause fractions by default in the preprocessing step.
