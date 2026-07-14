#!/usr/bin/env python3
"""
Mireye Field Risk Report — backend
===================================
A small FastAPI app that combines:
  - Mireye Earth (current-state terrain/hazard data, cited to federal sources)
  - Global Forest Watch (historical tree-cover-loss time series)

into one erosion + deforestation risk assessment for a US coordinate, and
serves the frontend that displays it.

SETUP
-----
1. pip install -r requirements.txt
2. Mireye: you should already be logged in via `uvx mireye-mcp login`.
   If this script can't find your token automatically, set it directly:
       export MIREYE_BEARER_TOKEN="your_token_here"
3. Global Forest Watch (free): sign up at globalforestwatch.org (use the
   "Sign up!" link, not Google/Facebook, so you get a password), then:
       export GFW_USERNAME="you@example.com"
       export GFW_PASSWORD="your_password"

RUN
---
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import json
import os
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

MIREYE_BASE_URL = "https://api.mireye.com"
MIREYE_CREDENTIALS_FILE = Path.home() / ".config" / "mireye-mcp" / "credentials.json"
GFW_BASE_URL = "https://data-api.globalforestwatch.org"

app = FastAPI(title="Mireye Field Risk Report")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Mireye
# ---------------------------------------------------------------------------

def get_mireye_token() -> str:
    token = os.environ.get("MIREYE_BEARER_TOKEN")
    if token:
        return token

    if MIREYE_CREDENTIALS_FILE.exists():
        creds = json.loads(MIREYE_CREDENTIALS_FILE.read_text())
        token = creds.get("access_token") or creds.get("token") or creds.get("api_key")
        if token:
            return token

    raise RuntimeError(
        "No Mireye token found. Run `uvx mireye-mcp login`, or set "
        "MIREYE_BEARER_TOKEN yourself (check "
        "~/.config/mireye-mcp/credentials.json for the right field name "
        "if this keeps failing)."
    )


def fetch_mireye_hazard_data(lat: float, lng: float) -> dict:
    token = get_mireye_token()
    resp = requests.post(
        f"{MIREYE_BASE_URL}/v1/fetch",
        headers={"Authorization": f"Bearer {token}"},
        json={"lat": lat, "lng": lng, "preset": "natural_hazard"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Global Forest Watch
# ---------------------------------------------------------------------------

def get_gfw_token() -> str:
    token = os.environ.get("GFW_API_TOKEN")
    if token:
        return token

    username = os.environ.get("GFW_USERNAME")
    password = os.environ.get("GFW_PASSWORD")
    if not username or not password:
        raise RuntimeError("GFW_USERNAME / GFW_PASSWORD not set — skipping GFW data.")

    resp = requests.post(
        f"{GFW_BASE_URL}/auth/token",
        data={"username": username, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def make_point_buffer_geojson(lat: float, lng: float, buffer_deg: float = 0.001) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [lng - buffer_deg, lat - buffer_deg],
            [lng + buffer_deg, lat - buffer_deg],
            [lng + buffer_deg, lat + buffer_deg],
            [lng - buffer_deg, lat + buffer_deg],
            [lng - buffer_deg, lat - buffer_deg],
        ]],
    }


def fetch_gfw_tree_cover_loss(lat: float, lng: float) -> list:
    token = get_gfw_token()
    geometry = make_point_buffer_geojson(lat, lng)
    sql = (
        "SELECT umd_tree_cover_loss__year, SUM(area__ha) AS area_ha "
        "FROM data WHERE umd_tree_cover_density_2000__percent > 30 "
        "GROUP BY umd_tree_cover_loss__year"
    )
    resp = requests.post(
        f"{GFW_BASE_URL}/dataset/umd_tree_cover_loss/latest/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"geometry": geometry, "sql": sql},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def score_erosion_risk(hazard_fields: dict, loss_years: list) -> dict:
    slope = hazard_fields.get("slope_degrees", {}).get("value")
    landslide_idx = hazard_fields.get("landslide_susceptibility_index", {}).get("value")
    in_floodplain = hazard_fields.get("within_floodplain_polygon", {}).get("value")

    total_recent_loss_ha = sum(
        row.get("area_ha", 0)
        for row in loss_years
        if row.get("umd_tree_cover_loss__year", 0) >= 2018
    )

    score = 0
    factors = []

    if slope is not None:
        if slope > 15:
            score += 2
            factors.append({"label": "Steep slope", "detail": f"{slope:.1f}°", "severity": "high"})
        elif slope > 5:
            score += 1
            factors.append({"label": "Moderate slope", "detail": f"{slope:.1f}°", "severity": "moderate"})
        else:
            factors.append({"label": "Gentle slope", "detail": f"{slope:.1f}°", "severity": "low"})

    if landslide_idx is not None:
        if landslide_idx > 50:
            score += 2
            factors.append({"label": "Landslide susceptibility", "detail": str(landslide_idx), "severity": "high"})
        else:
            factors.append({"label": "Landslide susceptibility", "detail": str(landslide_idx), "severity": "low"})

    if in_floodplain is not None:
        if in_floodplain:
            score += 1
            factors.append({"label": "Floodplain", "detail": "Within FEMA-mapped floodplain", "severity": "moderate"})
        else:
            factors.append({"label": "Floodplain", "detail": "Outside mapped floodplain", "severity": "low"})

    if total_recent_loss_ha > 0:
        score += 2
        factors.append({
            "label": "Recent deforestation",
            "detail": f"{total_recent_loss_ha:.2f} ha lost since 2018",
            "severity": "high",
        })
    elif loss_years:
        factors.append({"label": "Recent deforestation", "detail": "None detected since 2018", "severity": "low"})

    level = "low"
    if score >= 5:
        level = "high"
    elif score >= 3:
        level = "moderate"

    return {"score": score, "level": level, "factors": factors}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/api/risk")
def get_risk(lat: float = Query(...), lng: float = Query(...)):
    try:
        mireye_data = fetch_mireye_hazard_data(lat, lng)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mireye request failed: {e}")

    hazard_fields = mireye_data.get("fields", {})

    gfw_error = None
    try:
        loss_years = fetch_gfw_tree_cover_loss(lat, lng)
    except Exception as e:
        gfw_error = str(e)
        loss_years = []

    risk = score_erosion_risk(hazard_fields, loss_years)

    return {
        "location": {"lat": lat, "lng": lng},
        "risk": risk,
        "mireye_fields": hazard_fields,
        "tree_cover_loss_by_year": loss_years,
        "gfw_error": gfw_error,
    }


# Serve the frontend (index.html, css, js) from ../frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
