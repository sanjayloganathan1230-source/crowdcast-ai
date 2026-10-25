# CrowdCast AI — Crowd Flow Predictor

> **DEMO DATA ONLY.** CrowdCast AI is an educational portfolio project. Every observation in it is **synthetic**,
> produced by a seeded generator. It has **no live sensor feed**, and it is **not** an emergency or public-safety
> system. Do not use its output for real decisions.

A web application that learns from historical crowd counts and forecasts crowd volume and occupancy level for four demo
locations (railway station, bus terminal, shopping mall, college campus) up to 7 days ahead.

![Dashboard on desktop](docs/screenshots/desktop_full.png)

<p>
  <img src="docs/screenshots/desktop_evening_forecast.png" alt="Cards and map at the evening rush" width="62%">
  <img src="docs/screenshots/mobile_full.png" alt="Mobile layout" width="22%">
</p>

## Contents
[Features](#features) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Results](#results) ·
[API](#api) · [Testing](#testing) · [Project layout](#project-layout) · [Limitations](#limitations) ·
[Future work](#future-improvements) · [Troubleshooting](#troubleshooting) · [Credits](#credits)

## Features
- **Forecasts** for the next hour and for any chosen hour up to 168 h after the last observation, with an occupancy class
  (Low / Moderate / High / Very High) from **configurable capacity thresholds** (defaults 40 / 70 / 90 %).
- **Dashboard** (React, Recharts, Leaflet): next-hour and later-hour cards, map, 6–72 h forecast chart, hourly trend,
  predicted vs observed on a held-out period, peak hours, location comparison, model evaluation.
- **Filters and settings:** location, history window, weekday/weekend, "assume a scheduled event", thresholds.
- **CSV export** of observations (server) and forecasts (client), always labelled DEMO DATA.
- **Loading, error and retry states** on every panel; inputs validated in the UI and again in the API.
- **Accessible and responsive:** keyboard operable (including map markers), chart text summaries and data tables,
  single-column phone layout. axe-core finds no WCAG 2.1 A/AA violations.
- **Honest by construction:** DEMO DATA banner and per-panel tags; accuracy figures are read from the file written at
  training time; predictions are deterministic (no randomness at request time).

## Quick start
Tested with Python 3.12 and Node 22.12+ on Linux (see [Limitations](#limitations)). The commands avoid shell-specific
syntax apart from the virtual-environment activation line; use `python` or `python3` depending on your system.

```bash
# 1. Backend: install, then generate DEMO DATA -> SQLite -> trained model (about 10 s; safe to re-run)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt                    # exact tested versions (or requirements.txt for tested ranges)
python -m scripts.setup_demo

# 2. Frontend: install and build once
cd frontend && npm ci && npm run build && cd ..

# 3. Run everything on ONE origin: the API also serves the built dashboard
uvicorn backend.app.main:app --port 8000
#    dashboard: http://localhost:8000      API docs (Swagger): http://localhost:8000/docs
```

**Development mode** (hot reload, two terminals): `uvicorn backend.app.main:app --reload` and
`cd frontend && npm run dev` → http://localhost:5173 (Vite proxies `/api` to port 8000).

**Separate origins:** build the frontend with `VITE_API_BASE_URL=https://your-api` and add the page's origin to
`CROWDCAST_ALLOWED_ORIGINS` on the API (comma-separated).

> The trained model file records the scikit-learn version it was built with and is refused by a different
> `major.minor` version. After changing library versions, run `python -m scripts.setup_demo --force`.

### Configuration
Copy `.env.example` to `.env` in the project root. The API reads it at start-up; real environment variables win over it;
Vite reads the `VITE_*` entries from the same file. **No API keys or secrets are needed anywhere.**

| Variable | Default | Meaning |
|---|---|---|
| `CROWDCAST_SEED` | `42` | Seed for data generation and the model |
| `CROWDCAST_LOW_MAX` / `_MODERATE_MAX` / `_HIGH_MAX` | `40` / `70` / `90` | Upper bound of each occupancy class, % of capacity |
| `CROWDCAST_DB_PATH`, `_MODEL_PATH`, `_METRICS_PATH` | `data/…`, `models/…` | Where artifacts are stored |
| `CROWDCAST_ALLOWED_ORIGINS` | `http://localhost:5173` | CORS origins (not needed on a single origin) |
| `CROWDCAST_FRONTEND_DIST` | `frontend/dist` | Built dashboard served by the API when present |
| `VITE_API_BASE_URL` | *(empty = same origin)* | API address baked into a separately hosted frontend |

## How it works

```
scripts/setup_demo ─► seeded generator ─► data/raw/*.csv ─► SQLite ─► feature pipeline ─► RandomForest
                                                                                              │
          React dashboard  ◄── JSON ──  FastAPI  ◄── models/crowd_model.joblib + metrics.json ┘
```

**Data.** 182 days of hourly counts (2026-03-02 to 2026-08-30, 17,472 rows) for four locations with capacities of
5,000 / 3,000 / 8,000 / 6,000. Each location has its own weekday and weekend daily profile, day-to-day variation,
measurement noise and randomly scheduled 3-hour events. Columns: `timestamp`, `location_id`, `location`,
`crowd_count`, `capacity`, `weekday`, `hour`, `event_indicator`, `data_label`. The generator is a pure function of its
seed. Coordinates on the map are illustrative only.

**Features (15).** Hour of day (sine/cosine), weekday, weekend flag, event indicator, capacity, one-hot location, the
count at the same hour 1, 2, 3 and 4 weeks earlier, and the mean of those four.

**Why the 7-day limit.** Every history feature is at least a week old, so it is known for any target up to 168 h after
the last observation. A forecast made one hour ahead and one made six days ahead use the same inputs, so accuracy does
not depend on the horizon inside that range. The trade-off is that the model cannot use "what happened an hour ago",
which would help the very next hour.

**Target.** The forest predicts the deviation from the 4-week same-hour mean as a share of capacity, and the API adds it
back and clips to `[0, capacity]`. An earlier version predicted raw counts and reacted to events by only a third to a
half of their true effect; the change was chosen on a validation window *inside* the training period
(`scripts/compare_target_formulations.py`), not on the test set.

**No data leakage.**
- The split is chronological: the last 28 days are the test period, strictly later than all training rows. Rows are never shuffled.
- Features only look backwards, by timestamp lookup.
- Tests show that overwriting test-period data does not change any training feature, and that forecasts made from history
  cut at the forecast origin equal the batch evaluation to 1e-9. Deliberately leaking the future makes those tests fail.

**Model and baselines.** `RandomForestRegressor` (100 trees, `min_samples_leaf=8`, `max_features=0.6`; fixed, not tuned).
Compared with two simple baselines on MAE and RMSE: *same hour last week* and *mean of the same hour over 4 weeks*.
Training also saves the exact feature list, target formulation and library version, and the API refuses a model that
does not match the code.

## Results
Synthetic data, held-out last 28 days (2,688 rows), seed 42. Your own run prints these and the dashboard's
*Model evaluation* panel shows the contents of `models/metrics.json`; nothing is hardcoded.

| Model | MAE (people) | RMSE (people) | MAE as % of capacity |
|---|---|---|---|
| Baseline: same hour last week | 213.6 | 360.9 | 3.97 % |
| Baseline: mean of the same hour over 4 weeks | 168.9 | 283.8 | 3.14 % |
| **RandomForest** | **145.4** | **220.8** | **2.69 %** |

Versus the last-week baseline: MAE −31.9 %, RMSE −38.8 %. Repeating generation, training and evaluation for seeds 1–5
(`python -m scripts.evaluate_seeds`, saved in `docs/seed_robustness.json`), the forest beat both baselines in **5 of 5**
seeds, with MAE improvements of 30.5–36.2 % (mean 34.1 %).

**Events** are the hardest hours. For seed 42, on the 30 event hours in the test set the forest's MAE is 211 people (bias
−46) against 1,224 for the last-week baseline, but event hours are rare, so that estimate rests on few examples.

**Read these numbers carefully:** the data comes from smooth, hand-designed daily patterns, so a good score shows that
the pipeline works, not that real crowds are this predictable. See [docs/VERIFICATION.md](docs/VERIFICATION.md).

## API
Interactive documentation is at `/docs` when the server runs. All data responses include `data_label`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness, row count, model status |
| GET | `/api/config` | Data range, 168 h horizon, default thresholds |
| GET | `/api/locations` | Four demo locations (illustrative coordinates) |
| GET | `/api/observations` | History; filters `location_id`, `start`, `end`; `limit` (≤ 5000) and `offset` |
| GET | `/api/observations/export` | The same data as CSV |
| GET | `/api/analytics/hourly-profile` | Hourly averages and peak hours (`day_type`: all, weekday, weekend) |
| GET | `/api/analytics/location-comparison` | Mean and peak occupancy per location |
| GET | `/api/model/metrics` | The metrics file written by training |
| POST | `/api/predictions` | Forecast for up to 168 hourly timestamps |
| GET | `/api/predictions/next-hour` | First hour after the last observation |
| GET | `/api/predictions/backtest` | Predicted vs observed on the held-out test period only |

```bash
curl -X POST localhost:8000/api/predictions -H "Content-Type: application/json" \
  -d '{"location_id":"railway_station","timestamps":["2026-08-31T08:00:00","2026-08-31T18:00:00"]}'
```
```json
{
  "location_id": "railway_station", "location_name": "Demo Railway Station", "origin": "2026-08-30T23:00:00",
  "event_indicator": 0, "thresholds_used": {"low_max": 40, "moderate_max": 70, "high_max": 90},
  "predictions": [
    {"timestamp": "2026-08-31T08:00:00", "predicted_count": 3692, "capacity": 5000, "occupancy_pct": 73.8, "level": "High"},
    {"timestamp": "2026-08-31T18:00:00", "predicted_count": 3937, "capacity": 5000, "occupancy_pct": 78.7, "level": "High"}
  ],
  "data_label": "DEMO DATA - synthetic, not live sensor data",
  "note": "DEMO DATA: forecast from a model trained on synthetic observations. ..."
}
```

**Rules enforced** (HTTP 422 with an explanatory message otherwise): known `location_id`; timestamps timezone-naive, on the
hour, unique, after the last observation and at most 168 h after it; `event_indicator` 0 or 1 (applies to every requested
hour); `capacity_override` 1–1,000,000 (changes only the percentage and class, never the predicted count);
thresholds with `0 < low < moderate < high ≤ 100`. A missing model returns 503 with the command that fixes it.

"Future" means after the last observation in the demo dataset (2026-08-30 23:00), which is a fixed snapshot so that
results are reproducible, not the real current time.

## Testing
```bash
python -m pytest                   # 141 backend tests: data, features, training, leakage, API, config, static serving
cd frontend && npm test            # 45 dashboard tests (Vitest + Testing Library)
python tests/e2e/ui_smoke.py       # optional real-browser check, 37 checks (see the file header for prerequisites)
```
The browser check drives headless Chromium through the real API and model: rendering, interactions, keyboard use,
CSV downloads, an axe-core accessibility audit and layouts at 320 / 390 / 768 / 1280 px. The full evidence, the defects
it found along the way and what was **not** verified are in [docs/VERIFICATION.md](docs/VERIFICATION.md).

## Project layout
```
backend/app/
  api/routes.py           HTTP routes            ml/data_generation.py  seeded synthetic data
  core/config.py          settings, .env         ml/preprocessing.py    validation of observation tables
  core/occupancy.py       Low…Very High          ml/features.py         single feature pipeline (train + serve)
  db/database.py          SQLite layer           ml/model.py            target formulation (train + serve)
  schemas/api.py          request/response       ml/training.py         split, baselines, evaluation
  main.py                 app factory, static    ml/model_store.py      save/load with consistency checks
                                                 ml/predict.py          validated inference
frontend/src/             React app (components/, hooks.js, api.js, utils.js, test/)
scripts/                  setup_demo, generate_data, seed_db, train_model, evaluate_seeds, compare_target_formulations
tests/                    pytest suites and e2e/ui_smoke.py
docs/                     ARCHITECTURE.md, VERIFICATION.md, seed_robustness.json, screenshots/
data/  models/            generated artifacts (git-ignored)
```
More detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Limitations
- **Synthetic data.** Real crowds depend on holidays, weather, disruptions and behaviour that this data does not contain.
  Reported accuracy says nothing about real-world performance.
- **Fixed snapshot.** The dataset ends on 2026-08-30; "next hour" and "future" are relative to that, and nothing ingests new observations.
- **7-day horizon**, and no use of very recent observations, so next-hour forecasts are no more accurate than forecasts three days out.
- **Event handling is coarse.** One event flag applies to every requested hour; there is no event schedule or size, and event hours are rare in the data.
- **One chronological split** with fixed, untuned hyperparameters. No cross-validation, no tuning, and no uncertainty intervals (point forecasts only).
- **Four locations only.** Locations are one-hot encoded, so adding one means changing the code and retraining. Capacity is a fixed property of each location.
- **Counts are capped at capacity**, and timestamps are timezone-naive local demo time.
- **Not production-hardened.** No authentication or rate limiting, a single SQLite file, and a model held in memory in one process. Load was tested only with a burst of 240 requests.
- **Verified in a narrow environment:** Linux, Python 3.12, Node 22, Chromium. Not tried on Windows or macOS, in Firefox or Safari, or with a screen reader.
- **Map tiles** come from OpenStreetMap's public servers and need internet access; their [usage policy](https://operations.osmfoundation.org/policies/tiles/) is not suited to heavy traffic.
- **Bundle size:** the JavaScript bundle is about 795 kB (234 kB gzipped); it is not code-split.
- **No licence is included yet.** Add one before publishing the repository.

## Future improvements
- Use recent observations for short horizons (a separate next-few-hours model) and compare it with the current design.
- Walk-forward cross-validation and hyperparameter search, then prediction intervals (quantile forests or conformal prediction).
- Calendar and weather features; an input for scheduled events with start, end and expected size.
- An ingestion path for real counts (with a clear schema, validation and a retraining job) and drift monitoring.
- Add locations without code changes (location embeddings or per-location models).
- Docker image and CI running the three test suites; pin and update dependencies automatically.
- Authentication, rate limiting and a production database for any non-demo use.
- Route-level code splitting, self-hosted map tiles, and cross-browser and screen-reader testing.

## Troubleshooting
| Symptom | Cause and fix |
|---|---|
| Forecast panels say "No trained model…" (HTTP 503) | Run `python -m scripts.setup_demo` |
| 503 "…scikit-learn X but running Y" or "target mode" | The saved model is from other library versions or an older design: `python -m scripts.setup_demo --force` |
| UI shows "Cannot reach the API" | Backend not running, or (separate origins) the page's origin is missing from `CROWDCAST_ALLOWED_ORIGINS` |
| `/` returns 404 on the API port | The frontend is not built: `cd frontend && npm ci && npm run build` |
| `npm test` fails to start | Vitest needs Node ≥ 22.12 |
| Map shows grey tiles | OpenStreetMap tiles need internet access; markers and data still work |
| Port already in use | Pass another port: `uvicorn backend.app.main:app --port 8001` (and set the proxy target `CROWDCAST_API_TARGET` for `npm run dev`) |

## Credits
Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors. Built with scikit-learn, pandas, NumPy,
FastAPI, SQLite, React, Vite, Recharts and Leaflet. Accessibility audited with axe-core; tests use pytest, Vitest,
Testing Library and Playwright.
