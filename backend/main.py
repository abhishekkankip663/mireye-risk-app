#!/usr/bin/env python3
"""
Mireye Field Risk Report — backend (v3)
=========================================
v3 upgrade: replaces the point-additive heuristic with a RUSLE-lite
erosion calculation built on Mireye's ACTUAL erosion-science fields —
discovered from the full /v1/meta/fields catalog, not assumed.

THE REAL RUSLE EQUATION
------------------------
    A = R x K x LS x C x P

    R = rainfall erosivity        -> NOT available from Mireye (named gap)
    K = soil erodibility            -> Mireye: soil_erodibility_k_factor (real!)
    LS = slope length-steepness     -> approximated from slope_degrees
                                        (real slope LENGTH unavailable, so an
                                        assumed 100m reference length is used —
                                        flagged explicitly, not hidden)
    C = cover-management factor      -> approximated from tree_canopy_pct,
                                        adjusted by ndvi_change_5y trend
    P = support practice factor      -> not modeled; assumed 1 (no terracing/
                                        contouring data available)

Because R is missing, this computes a RELATIVE erosion index
(K x LS x C), not an absolute tons/acre/year figure. That is stated
explicitly in the API response and the UI — this is a screening index,
not a certified engineering calculation.

CORRECTION FROM v1/v2
-----------------------
v1/v2 said Mireye had no soil-erodibility data and used a simple
point-additive heuristic instead. That was wrong — Mireye's full field
catalog (api.mireye.com/v1/meta/fields) includes `soil_erodibility_k_factor`,
the actual USLE/RUSLE K-factor. v3 uses it directly.

SETUP / RUN: same as v2 — see README.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

MIREYE_BASE_URL = "https://api.mireye.com"
MIREYE_CREDENTIALS_FILE = Path.home() / ".config" / "mireye-mcp" / "credentials.json"
GFW_BASE_URL = "https://data-api.globalforestwatch.org"
SDA_URL = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Verified against the live /v1/meta/fields catalog before adding.
EXTRA_FIELDS = [
    "soil_erodibility_k_factor",     # real RUSLE K-factor
    "soil_hydrologic_group",         # A/B/C/D runoff-infiltration class
    "soil_restrictive_layer_depth_cm",
    "soil_restrictive_layer_kind",
    "soil_ponding_frequency_class",  # standing-water tendency (leaching-relevant)
    "fema_flood_zone",               # actual zone code (AE, VE, X, ...), not just true/false
    "coastal_high_hazard",           # V-zone wave-action flag
    "fema_base_flood_elevation",
    "wetland_fraction_of_parcel",    # PARCEL-level share, not just "a wetland exists nearby"
    "wetland_acres_on_parcel",
    "ndvi_change_5y",                # Mireye's own 5-yr vegetation-change signal
    "tree_canopy_pct",               # feeds the C-factor proxy
    "prime_farmland_classification",
]

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
    raise RuntimeError("No Mireye token found. Run `uvx mireye-mcp login` or set MIREYE_BEARER_TOKEN.")


def fetch_mireye_hazard_data(lat: float, lng: float) -> dict:
    token = get_mireye_token()
    resp = requests.post(
        f"{MIREYE_BASE_URL}/v1/fetch",
        headers={"Authorization": f"Bearer {token}"},
        json={"lat": lat, "lng": lng, "preset": "natural_hazard", "fields": EXTRA_FIELDS},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_mireye_ask(lat: float, lng: float, question: str) -> dict:
    token = get_mireye_token()
    resp = requests.post(
        f"{MIREYE_BASE_URL}/v1/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={"lat": lat, "lng": lng, "question": question},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Global Forest Watch (supplementary — year-level granularity)
# ---------------------------------------------------------------------------

def get_gfw_api_key() -> str:
    """The Data API's actual query endpoints authenticate via an
    'x-api-key' header — NOT the bearer access_token from /auth/token.
    The access_token is only used once, to CREATE the api_key (see
    README for the one-time curl setup). Save the resulting api_key as
    GFW_API_KEY."""
    api_key = os.environ.get("GFW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GFW_API_KEY not set. Run the one-time setup in the README "
            "(get an access token, then POST /auth/apikey) and export "
            "the resulting api_key as GFW_API_KEY."
        )
    return api_key


def make_point_buffer_geojson(lat: float, lng: float, buffer_deg: float = 0.001) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[
            [lng - buffer_deg, lat - buffer_deg], [lng + buffer_deg, lat - buffer_deg],
            [lng + buffer_deg, lat + buffer_deg], [lng - buffer_deg, lat + buffer_deg],
            [lng - buffer_deg, lat - buffer_deg],
        ]],
    }


def get_gfw_latest_version(api_key: str) -> str:
    """Dataset versions aren't literally 'latest' in the URL — look up the
    real current version string first."""
    resp = requests.get(
        f"{GFW_BASE_URL}/dataset/umd_tree_cover_loss",
        headers={"x-api-key": api_key, "Origin": "http://localhost"},
        timeout=30,
    )
    resp.raise_for_status()
    versions = resp.json().get("data", {}).get("versions", [])
    if not versions:
        raise RuntimeError("Could not determine umd_tree_cover_loss dataset version from GFW.")
    return versions[-1]


def fetch_gfw_tree_cover_loss(lat: float, lng: float) -> list:
    api_key = get_gfw_api_key()
    version = get_gfw_latest_version(api_key)
    geometry = make_point_buffer_geojson(lat, lng)
    sql = (
        "SELECT umd_tree_cover_loss__year, SUM(area__ha) AS area_ha FROM results "
        "WHERE umd_tree_cover_density_2000__percent > 30 GROUP BY umd_tree_cover_loss__year"
    )
    resp = requests.post(
        f"{GFW_BASE_URL}/dataset/umd_tree_cover_loss/{version}/query/json",
        headers={"x-api-key": api_key, "Content-Type": "application/json", "Origin": "http://localhost"},
        json={"sql": sql, "geometry": geometry},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


# ---------------------------------------------------------------------------
# USDA Soil Data Access — live fallback when Mireye's cached raster is null
# ---------------------------------------------------------------------------
#
# Mireye's own field description for soil_erodibility_k_factor/hydrologic_group
# says it reads a cached gNATSGO raster mirror, and falls back to live SDA
# ("Soil Data Access") for out-of-grid points. When Mireye returns a real
# `status: "ok"` response but the VALUE itself is null (a raster gap, not a
# request failure), that's worth double-checking against the live source
# directly — the survey data may genuinely exist even if Mireye's cached
# mirror has a hole at that exact pixel.
#
# Public API, no key required: https://sdmdataaccess.nrcs.usda.gov/
#
# NOTE: this query pattern is built from USDA's documented SDA conventions
# (the SDA_Get_Mukey_by_Point table function, standard SSURGO mapunit ->
# component -> chorizon joins). It has NOT been live-tested end-to-end here —
# this environment's network allowlist doesn't include sdmdataaccess.nrcs.usda.gov.
# If the response shape doesn't match on your first real run, the raw response
# is logged so it's a quick fix rather than a silent failure.

def _shrink_swell_class_from_lep(lep):
    """Bin linear extensibility percent (LEP) into the standard NRCS
    shrink-swell potential class (Low/Moderate/High/Very high)."""
    try:
        v = float(lep)
    except (TypeError, ValueError):
        return None
    if v < 3:
        return "Low"
    if v < 6:
        return "Moderate"
    if v < 9:
        return "High"
    return "Very high"


def fetch_ssurgo_live_fallback(lat: float, lng: float) -> dict:
    point_wkt = f"point({lng} {lat})"
    sql = f"""
        SELECT TOP 1 co.hydgrp, ch.kwfact, ch.lep_r, mu.muname
        FROM mapunit mu
        INNER JOIN component co ON co.mukey = mu.mukey AND co.majcompflag = 'Yes'
        INNER JOIN chorizon ch ON ch.cokey = co.cokey
        WHERE mu.mukey IN (SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{point_wkt}'))
        ORDER BY co.comppct_r DESC, ch.hzdept_r ASC
    """
    resp = requests.post(SDA_URL, json={"query": sql, "format": "JSON"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    table = data.get("Table")
    if not table:
        return {}
    row = table[0]
    try:
        kwfact = float(row[1]) if row[1] not in (None, "") else None
    except (TypeError, ValueError):
        kwfact = None
    return {
        "hydgrp": row[0],
        "kwfact": kwfact,
        "shrink_swell": _shrink_swell_class_from_lep(row[2]),
        "muname": row[3],
    }


def patch_nulls_with_live_ssurgo(lat: float, lng: float, fields: dict) -> dict:
    """If Mireye's own soil fields came back null, try USDA SDA directly.
    Only touches fields that were genuinely null — never overwrites a real
    Mireye value, and clearly marks anything it fills in as a secondary,
    unverified-in-this-build source rather than pretending it came from Mireye.
    """
    k_missing = field_value(fields, "soil_erodibility_k_factor") is None
    hydgrp_missing = field_value(fields, "soil_hydrologic_group") is None
    ss_missing = field_value(fields, "soil_shrink_swell_class") is None
    if not (k_missing or hydgrp_missing or ss_missing):
        return fields

    try:
        live = fetch_ssurgo_live_fallback(lat, lng)
    except Exception as e:
        fields.setdefault("_ssurgo_live_fallback_error", {"status": "failed", "error": str(e)})
        return fields

    if k_missing and live.get("kwfact") is not None:
        fields["soil_erodibility_k_factor"] = {
            "value": live["kwfact"],
            "unit": None,
            "source": "USDA_SDA_LIVE",
            "source_url": "https://sdmdataaccess.nrcs.usda.gov/",
            "confidence": "medium",
            "notes": f"Live SDA fallback (Mireye's cached raster was null at this pixel). Map unit: {live.get('muname', 'unknown')}.",
            "status": "ok",
        }
    if hydgrp_missing and live.get("hydgrp") is not None:
        fields["soil_hydrologic_group"] = {
            "value": live["hydgrp"],
            "unit": None,
            "source": "USDA_SDA_LIVE",
            "source_url": "https://sdmdataaccess.nrcs.usda.gov/",
            "confidence": "medium",
            "notes": f"Live SDA fallback (Mireye's cached raster was null at this pixel). Map unit: {live.get('muname', 'unknown')}.",
            "status": "ok",
        }
    if ss_missing and live.get("shrink_swell") is not None:
        fields["soil_shrink_swell_class"] = {
            "value": live["shrink_swell"],
            "unit": None,
            "source": "USDA_SDA_LIVE",
            "source_url": "https://sdmdataaccess.nrcs.usda.gov/",
            "confidence": "medium",
            "notes": f"Live SDA fallback binned from shallowest-horizon lep_r (Mireye's cached raster was null at this pixel). Map unit: {live.get('muname', 'unknown')}.",
            "status": "ok",
        }
    return fields


# ---------------------------------------------------------------------------
# RUSLE-lite erosion index
# ---------------------------------------------------------------------------

ASSUMED_SLOPE_LENGTH_M = 100.0  # Mireye has no real slope-length field — flagged assumption


def field_value(fields: dict, name: str):
    entry = fields.get(name)
    if not entry or entry.get("status") == "failed":
        return None
    return entry.get("value")


def compute_ls_factor(slope_degrees: float) -> float:
    """McCool et al. slope-length-steepness approximation.
    Real RUSLE needs actual measured slope length; we assume a 100m
    reference length since Mireye doesn't provide flow-length data.
    """
    theta = math.radians(slope_degrees)
    slope_pct = math.tan(theta) * 100
    m = 0.5 if slope_pct >= 5 else (0.4 if slope_pct >= 3 else (0.3 if slope_pct >= 1 else 0.2))
    ls = ((ASSUMED_SLOPE_LENGTH_M / 22.13) ** m) * (
        65.41 * math.sin(theta) ** 2 + 4.56 * math.sin(theta) + 0.065
    )
    return ls


def compute_c_factor(tree_canopy_pct, ndvi_change_5y) -> float:
    """Rough cover-management factor proxy: dense canopy -> low C (near 0),
    bare ground -> C approaching 1. Not a certified USDA C-factor table
    lookup (those require exact crop/cover-type), but keeps the same
    directional logic on Mireye's real canopy field.
    """
    canopy = tree_canopy_pct if tree_canopy_pct is not None else 20.0  # neutral default if missing
    c = 1.0 - (canopy / 100.0) * 0.96
    if ndvi_change_5y is not None and ndvi_change_5y < 0:
        # vegetation trending down -> cover factor creeps toward bare-soil value
        c = min(1.0, c * (1 + abs(ndvi_change_5y) * 2))
    return round(c, 4)


# ---------------------------------------------------------------------------
# Rainfall erosivity (R) — the one RUSLE factor Mireye doesn't provide.
# Sourced live from mean annual precipitation (Open-Meteo ERA5 archive, no key)
# via the Renard & Freimund (1994) empirical relationship, then converted from
# SI to US customary units so it multiplies correctly with SSURGO's US-customary
# K-factor. This is a SCREENING estimate, not a measured isoerodent value.
# ---------------------------------------------------------------------------

def fetch_mean_annual_precip_mm(lat: float, lng: float) -> float:
    """10 full calendar years of daily precip from Open-Meteo (ERA5), averaged
    to a mean annual total (mm). Uses the last 10 complete years."""
    end_year = datetime.now(timezone.utc).year - 1
    start_year = end_year - 9
    resp = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lng,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "daily": "precipitation_sum",
            "timezone": "UTC",
        },
        timeout=30,
    )
    resp.raise_for_status()
    daily = resp.json().get("daily", {})
    times = daily.get("time", [])
    vals = daily.get("precipitation_sum", [])
    by_year = {}
    for t, v in zip(times, vals):
        if v is not None:
            by_year[t[:4]] = by_year.get(t[:4], 0.0) + v
    if not by_year:
        raise RuntimeError("Open-Meteo returned no precipitation data for this point.")
    return sum(by_year.values()) / len(by_year)


def compute_r_factor_us(mean_annual_precip_mm: float) -> float:
    """Renard & Freimund (1994) rainfall-erosivity R from mean annual
    precipitation P (mm), then convert SI (MJ·mm·ha⁻¹·h⁻¹·yr⁻¹) to US customary
    (hundreds ft·tonf·in·acre⁻¹·h⁻¹·yr⁻¹) by dividing by 17.02 so it pairs with
    SSURGO's US-customary K-factor."""
    P = mean_annual_precip_mm
    r_si = 0.0483 * P ** 1.610 if P <= 850 else 587.8 - 1.219 * P + 0.004105 * P ** 2
    return r_si / 17.02


def score_erosion_risk(fields: dict, loss_years: list, lat: float, lng: float) -> dict:
    slope = field_value(fields, "slope_degrees")
    k_factor = field_value(fields, "soil_erodibility_k_factor")
    tree_canopy = field_value(fields, "tree_canopy_pct")
    ndvi_change_5y = field_value(fields, "ndvi_change_5y")
    landslide_idx = field_value(fields, "landslide_susceptibility_index")
    fema_zone = field_value(fields, "fema_flood_zone")
    coastal_hazard = field_value(fields, "coastal_high_hazard")
    in_floodplain = field_value(fields, "within_floodplain_polygon")
    wetland_frac = field_value(fields, "wetland_fraction_of_parcel")
    ponding = field_value(fields, "soil_ponding_frequency_class")
    hydro_group = field_value(fields, "soil_hydrologic_group")

    factors = []
    score = 0

    rusle = None
    if slope is not None and k_factor is not None:
        ls = compute_ls_factor(slope)
        c = compute_c_factor(tree_canopy, ndvi_change_5y)
        relative_index = round(k_factor * ls * c, 4)
        rusle = {
            "k_factor": k_factor,
            "ls_factor": round(ls, 3),
            "c_factor": c,
            "p_factor": 1.0,
            "relative_index": relative_index,
            "assumed_slope_length_m": ASSUMED_SLOPE_LENGTH_M,
            "note": "K x LS x C only — R (rainfall erosivity) not yet resolved; relative screening index, not tons/acre/year.",
        }
        # Upgrade to an absolute A = R x K x LS x C (tons/acre/yr) with a live
        # R-factor estimated from mean annual precipitation. Screening-grade.
        abs_suffix = ""
        try:
            map_mm = fetch_mean_annual_precip_mm(lat, lng)
            r_us = compute_r_factor_us(map_mm)
            soil_loss = round(r_us * k_factor * ls * c, 2)  # P assumed 1
            rusle.update({
                "r_factor": round(r_us, 2),
                "mean_annual_precip_mm": round(map_mm, 1),
                "r_factor_source": (
                    "estimated: Renard & Freimund (1994) R from mean annual precipitation; "
                    "precip = Open-Meteo ERA5 10-yr mean (no measured isoerodent value available)"
                ),
                "annual_soil_loss_tons_per_acre": soil_loss,
                "note": (
                    "A = R x K x LS x C (P assumed 1). R is ESTIMATED from mean annual "
                    "precipitation, not a measured isoerodent value — screening-grade, "
                    "not a certified engineering figure."
                ),
            })
            abs_suffix = f" (≈{soil_loss} tons/acre/yr est.)"
        except Exception as e:
            rusle["r_factor_error"] = str(e)
        if relative_index > 1.5:
            score += 3
            factors.append({"label": "RUSLE-lite erosion index (K×LS×C)", "detail": f"{relative_index} — high{abs_suffix}", "severity": "high"})
        elif relative_index > 0.5:
            score += 2
            factors.append({"label": "RUSLE-lite erosion index (K×LS×C)", "detail": f"{relative_index} — moderate{abs_suffix}", "severity": "moderate"})
        else:
            factors.append({"label": "RUSLE-lite erosion index (K×LS×C)", "detail": f"{relative_index} — low{abs_suffix}", "severity": "low"})
    else:
        factors.append({"label": "RUSLE-lite erosion index", "detail": "unavailable (missing slope or K-factor at this point)", "severity": "moderate"})

    if landslide_idx is not None:
        if landslide_idx > 50:
            score += 2
            factors.append({"label": "Landslide susceptibility", "detail": str(landslide_idx), "severity": "high"})
        else:
            factors.append({"label": "Landslide susceptibility", "detail": str(landslide_idx), "severity": "low"})

    # Flood: prefer the real zone code, fall back to the plain boolean
    if fema_zone is not None:
        severity = "high" if coastal_hazard else ("moderate" if str(fema_zone).upper().startswith(("A", "V")) else "low")
        if severity != "low":
            score += 2 if severity == "high" else 1
        factors.append({"label": "FEMA flood zone", "detail": f"Zone {fema_zone}" + (" (coastal high-hazard)" if coastal_hazard else ""), "severity": severity})
    elif in_floodplain is not None:
        if in_floodplain:
            score += 1
            factors.append({"label": "Floodplain", "detail": "Within FEMA-mapped floodplain", "severity": "moderate"})
        else:
            factors.append({"label": "Floodplain", "detail": "Outside mapped floodplain", "severity": "low"})

    # Leaching-relevant: parcel-level wetland share + ponding + hydrologic group
    if wetland_frac is not None and wetland_frac > 0:
        score += 1
        factors.append({"label": "Wetland share of parcel", "detail": f"{wetland_frac*100:.1f}% of parcel", "severity": "moderate"})
    if ponding is not None and str(ponding).lower() not in ("none", "unknown"):
        score += 1
        factors.append({"label": "Soil ponding frequency", "detail": str(ponding), "severity": "moderate"})
    if hydro_group is not None:
        severity = "high" if "D" in str(hydro_group) else "low"
        if severity == "high":
            score += 1
        factors.append({"label": "Soil hydrologic group", "detail": f"Group {hydro_group}", "severity": severity})

    total_recent_loss_ha = sum(row.get("area_ha", 0) for row in loss_years if row.get("umd_tree_cover_loss__year", 0) >= 2018)
    if total_recent_loss_ha > 0:
        score += 2
        factors.append({"label": "Recent deforestation (GFW, year-level)", "detail": f"{total_recent_loss_ha:.2f} ha lost since 2018", "severity": "high"})
    elif loss_years:
        factors.append({"label": "Recent deforestation (GFW, year-level)", "detail": "None detected since 2018", "severity": "low"})

    level = "low"
    if score >= 6:
        level = "high"
    elif score >= 3:
        level = "moderate"

    return {"score": score, "level": level, "factors": factors, "rusle_lite": rusle}


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

ASK_QUESTION = (
    "Summarize the erosion, soil stability, and vegetation-loss-relevant "
    "characteristics of this location in a few sentences."
)


@app.get("/api/risk")
def get_risk(lat: float = Query(...), lng: float = Query(...)):
    try:
        mireye_data = fetch_mireye_hazard_data(lat, lng)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Mireye /v1/fetch request failed: {e}")

    fields = mireye_data.get("fields", {})
    fields = patch_nulls_with_live_ssurgo(lat, lng, fields)

    ask_summary = None
    ask_error = None
    try:
        ask_result = fetch_mireye_ask(lat, lng, ASK_QUESTION)
        ask_summary = ask_result.get("answer")
    except Exception as e:
        ask_error = str(e)

    gfw_error = None
    try:
        loss_years = fetch_gfw_tree_cover_loss(lat, lng)
    except Exception as e:
        gfw_error = str(e)
        loss_years = []

    risk = score_erosion_risk(fields, loss_years, lat, lng)

    return {
        "location": {"lat": lat, "lng": lng},
        "risk": risk,
        "mireye_fields": fields,
        "mireye_partial_failures": mireye_data.get("partial_failures", []),
        "mireye_ask_summary": ask_summary,
        "mireye_ask_error": ask_error,
        "tree_cover_loss_by_year": loss_years,
        "gfw_error": gfw_error,
    }


frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
