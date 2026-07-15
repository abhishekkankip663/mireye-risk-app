# Mireye Field Risk Report

An end-to-end web app that combines **Mireye Earth** (cited, current-state
terrain/hazard data) with **Global Forest Watch** (historical tree-cover-loss
time series) into a single erosion + deforestation risk assessment for any
U.S. coordinate.

This exists to demonstrate a real-world use case built on top of Mireye's
API: point-in-time hazard data alone can't tell you a hillside is
destabilizing — you need to know it *also* lost its vegetation cover
recently. This app makes that connection automatically.

## What it does

1. You enter a coordinate (or click a preset location).
2. The backend calls Mireye's `natural_hazard` preset (slope, landslide
   susceptibility, flood zone, seismic data — all cited to USGS/FEMA/NOAA/
   USACE/NRCS).
3. It separately calls Global Forest Watch for tree-cover-loss-by-year at
   that point.
4. It combines both into a risk score and renders a report: a visual
   "core sample" of risk factors, a cited data table, and a tree-cover-loss
   chart.

## Setup

### 1. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt --break-system-packages
```

### 2. Mireye authentication

You should already be logged in from earlier (`uvx mireye-mcp login`).
This app reads the same credential file it created
(`~/.config/mireye-mcp/credentials.json`).

If it can't find the token automatically, set it directly:

```bash
export MIREYE_BEARER_TOKEN="your_token_here"
```

### 3. Global Forest Watch authentication (free)

1. Sign up at https://www.globalforestwatch.org/ — use the **"Sign up!"**
   link, not Google/Facebook login, so you get a password you can use for
   API auth.
2. Follow GFW's one-time setup to exchange your login for an `api_key`
   (get an access token, then `POST /auth/apikey` — see comments in
   `backend/main.py`'s `get_gfw_api_key()` for the exact flow), then set:

```bash
export GFW_API_KEY="your_api_key_here"
```

If you skip this step, the app still runs — it just omits the
deforestation-history chart rather than showing an error (missing data is
hidden from the report rather than called out as unavailable).

## Run it

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

## Most At-Risk Areas (county rankings)

Besides looking up a single point, the app has a second view that ranks the
worst counties in the US (filterable by state/region/entire US). This is
backed by a precomputed cache (`backend/rankings_cache.json`), not a live
calculation — computing every county on every request would be far too slow
and would hammer several free public APIs' rate limits. To (re)generate it:

```bash
cd backend
python3 precompute_rankings.py
```

This scores all 3,144 US counties (`backend/counties_full.csv`, from the
official US Census Gazetteer file) using 10 concurrent workers and keeps the
top 100 by risk score. Takes roughly 30-40 minutes and needs the same Mireye
token + `GFW_API_KEY` as the main app. See the script's own docstring for
more detail (why it's a separate script, why concurrency instead of
sampling, etc.).

In production this runs on a weekly schedule via
`.github/workflows/refresh-rankings.yml` (GitHub Actions), which commits the
refreshed cache back to `main` — that commit then triggers an automatic
redeploy on Render, so the live rankings stay current without needing any
particular machine to be running the job.

## Deployment

The app deploys to [Render](https://render.com) as a free web service via
the `render.yaml` Blueprint at the repo root:

1. Push this repo to GitHub (already done if you're reading this from there).
2. In the Render dashboard: New → Blueprint → connect this repo → Apply.
3. When prompted, paste in real values for `MIREYE_BEARER_TOKEN` and
   `GFW_API_KEY` (Render keeps these out of the YAML/git via `sync: false`).
4. Add the same two values as **repository secrets** in GitHub (Settings →
   Secrets and variables → Actions) so `refresh-rankings.yml` can run.

Render auto-deploys on every push to `main`, including the automated weekly
commits from the rankings-refresh workflow. The free tier spins down after
15 minutes of inactivity — the first request after a quiet period takes
30-60 seconds to wake back up.

## Notes on reliability

- The **Mireye integration is verified** — its request/response shape was
  confirmed against a live call before this was built.
- Several fields Mireye sometimes returns as null are backfilled from live
  public fallback sources instead of being shown as missing: USDA Soil Data
  Access (soil fields), USGS 3DEP (slope), FEMA's National Flood Hazard
  Layer (flood zone), USFWS's National Wetlands Inventory (wetlands), and
  MRLC's NLCD (tree canopy) — see the corresponding sections in
  `backend/main.py` for confidence levels and caveats per source.
- A field that's still missing after all fallbacks is simply omitted from
  the report rather than shown as an error.

## Project structure

```
mireye-risk-app/
├── render.yaml                            Render Blueprint (hosting config)
├── .github/workflows/
│   ├── refresh-rankings.yml               Weekly: recompute county rankings, commit if changed
│   └── tests.yml                          CI: run the pytest suite on push/PR
├── backend/
│   ├── main.py                            FastAPI app: /api/risk, /api/rankings, serves frontend
│   ├── precompute_rankings.py             Batch script behind the rankings cache
│   ├── counties_full.csv                  All 3,144 US counties (Census Gazetteer)
│   ├── counties_seed.csv                  Older 51-county starter set (kept for reference, unused by default)
│   ├── rankings_cache.json                Precomputed top-100 risk rankings (regenerated periodically)
│   ├── test_main.py / test_precompute_rankings.py   pytest suites
│   └── requirements.txt
├── frontend/
│   └── index.html                         Single-page UI (vanilla JS, no build step)
└── README.md
```
