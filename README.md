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
2. Set:

```bash
export GFW_USERNAME="you@example.com"
export GFW_PASSWORD="your_password"
```

The backend exchanges these for a short-lived API token automatically.
If you skip this step, the app still runs — it just shows Mireye's
terrain/hazard data without the deforestation history piece, and tells you
why in the UI.

## Run it

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

## Notes on reliability

- The **Mireye integration is verified** — its request/response shape was
  confirmed against a live call before this was built.
- The **Global Forest Watch integration is built from their public API
  docs** but wasn't live-tested end-to-end while building this (no network
  access to their API from the environment this was built in). If the GFW
  call errors on your first run, the UI will show you the exact error
  rather than failing silently — paste it back for a quick fix if needed.

## Project structure

```
mireye-risk-app/
├── backend/
│   ├── main.py           FastAPI app: calls Mireye + GFW, scores risk, serves frontend
│   └── requirements.txt
├── frontend/
│   └── index.html        Single-page UI (vanilla JS, no build step)
└── README.md
```
