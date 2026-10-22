# Longevity Simulator

A browser-based tool showing how curing diseases or slowing aging could change lifespan and healthspan for a U.S. resident. It's a static site with no backend: everything runs in the browser.

Data sources:
- CDC/NCHS United States Life Tables (2010–2019): mortality rates by age
- CDC Multiple Cause of Death database (2024): cause-of-death fractions by age
- IHME Global Burden of Disease (2023): chronic-condition prevalence by age

## Run locally

```bash
cd web && python3 -m http.server 5180
# open http://127.0.0.1:5180
```

Any static file server works. Opening `index.html` directly from disk won't, because the page `fetch`es `data.json`.

## Deploy

Deploy `web/` as a static site (on Vercel, set the project's Root Directory to `web`, with no framework and no build command).

## Project structure

```
web/                  # The app (deployed as-is)
  index.html          # Layout and copy
  app.js              # UI wiring, charts (Plotly.js), CSV export
  model.js            # The simulation: cause removal, aging slowdown, survival, healthspan, Gompertz fit
  model.check.mjs     # Regression check for model.js: node web/model.check.mjs
  data.json           # Bundled runtime data, generated from data/
data/                 # Processed CSVs (committed)
preprocessing/        # Python scripts: raw_data/ -> data/ -> web/data.json
raw_data/             # Large raw source files (gitignored)
```

## Updating the data

```bash
pip install -r requirements.txt
cd preprocessing && python process_life_table.py && python process_mmcd.py && cd ..
python preprocessing/export_web_data.py    # rebuilds web/data.json
node web/model.check.mjs                   # reference values may need updating if the data changed
```

See `preprocessing/README.md` for details on the raw data.

## License

MIT License, see LICENSE.

## Author

Alexander Epstein, Cao Lab, Rockefeller University
