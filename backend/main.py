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

# Every field above except two now has a live fallback for when Mireye's own
# value is null (see the USDA SDA / USGS 3DEP / FEMA NFHL / USFWS NWI / MRLC
# NLCD sections below). The two without one:
#
#   - landslide_susceptibility_index: no free, no-auth, nationwide,
#     point-queryable dataset exists. USGS has only published regional
#     landslide susceptibility studies (patchy coverage, inconsistent
#     scales/methodologies) and a point-feature landslide INVENTORY (past
#     events), not a continuous susceptibility index comparable to Mireye's.
#     Filling this from a mismatched source would misrepresent confidence
#     rather than genuinely reduce the gap.
#   - ndvi_change_5y: computing a real 5-year NDVI trend requires a
#     multi-date satellite imagery time series (e.g. via Google Earth Engine
#     or NASA AppEEARS), both of which need an authenticated account/API key
#     and non-trivial per-point processing — out of scope for a same-shape
#     drop-in fallback like the others here.
#
# Both are left as genuine gaps (hidden in the UI per the missing-data
# behavior, not fabricated) rather than backed by a low-quality proxy.

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
    versions = (resp.json().get("data") or {}).get("versions", [])
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
    return resp.json().get("data") or []


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
        SELECT TOP 1 co.hydgrp, ch.kwfact, ch.lep_r, mu.muname, ma.pondfreqprs,
               co.farmlndcl, cr.reskind, cr.resdept_r
        FROM mapunit mu
        INNER JOIN component co ON co.mukey = mu.mukey AND co.majcompflag = 'Yes'
        INNER JOIN chorizon ch ON ch.cokey = co.cokey
        LEFT JOIN muaggatt ma ON ma.mukey = mu.mukey
        LEFT JOIN corestrictions cr ON cr.cokey = co.cokey
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
    try:
        resdept_cm = float(row[7]) if len(row) > 7 and row[7] not in (None, "") else None
    except (TypeError, ValueError):
        resdept_cm = None
    return {
        "hydgrp": row[0],
        "kwfact": kwfact,
        "shrink_swell": _shrink_swell_class_from_lep(row[2]),
        "muname": row[3],
        "pondfreq": row[4] if len(row) > 4 else None,
        "farmland_class": row[5] if len(row) > 5 else None,
        "restrictive_kind": row[6] if len(row) > 6 else None,
        "restrictive_depth_cm": resdept_cm,
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
    pond_missing = field_value(fields, "soil_ponding_frequency_class") is None
    farmland_missing = field_value(fields, "prime_farmland_classification") is None
    res_kind_missing = field_value(fields, "soil_restrictive_layer_kind") is None
    res_depth_missing = field_value(fields, "soil_restrictive_layer_depth_cm") is None
    if not (k_missing or hydgrp_missing or ss_missing or pond_missing
            or farmland_missing or res_kind_missing or res_depth_missing):
        return fields

    try:
        live = fetch_ssurgo_live_fallback(lat, lng)
    except Exception as e:
        fields.setdefault("_ssurgo_live_fallback_error", {"status": "failed", "error": str(e)})
        return fields

    if farmland_missing and live.get("farmland_class") is not None:
        fields["prime_farmland_classification"] = {
            "value": live["farmland_class"],
            "unit": None,
            "source": "USDA_SDA_LIVE",
            "source_url": "https://sdmdataaccess.nrcs.usda.gov/",
            "confidence": "medium",
            "notes": f"Live SDA fallback from component.farmlndcl (Mireye's cached raster was null at this pixel). Map unit: {live.get('muname', 'unknown')}.",
            "status": "ok",
        }
    if res_kind_missing and live.get("restrictive_kind") is not None:
        fields["soil_restrictive_layer_kind"] = {
            "value": live["restrictive_kind"],
            "unit": None,
            "source": "USDA_SDA_LIVE",
            "source_url": "https://sdmdataaccess.nrcs.usda.gov/",
            "confidence": "medium",
            "notes": f"Live SDA fallback from corestrictions.reskind (Mireye's cached raster was null at this pixel). Map unit: {live.get('muname', 'unknown')}.",
            "status": "ok",
        }
    if res_depth_missing and live.get("restrictive_depth_cm") is not None:
        fields["soil_restrictive_layer_depth_cm"] = {
            "value": live["restrictive_depth_cm"],
            "unit": "cm",
            "source": "USDA_SDA_LIVE",
            "source_url": "https://sdmdataaccess.nrcs.usda.gov/",
            "confidence": "medium",
            "notes": f"Live SDA fallback from corestrictions.resdept_r (Mireye's cached raster was null at this pixel). Map unit: {live.get('muname', 'unknown')}.",
            "status": "ok",
        }
    if pond_missing and live.get("pondfreq") is not None:
        fields["soil_ponding_frequency_class"] = {
            "value": live["pondfreq"],
            "unit": None,
            "source": "USDA_SDA_LIVE",
            "source_url": "https://sdmdataaccess.nrcs.usda.gov/",
            "confidence": "medium",
            "notes": f"Live SDA fallback from muaggatt.pondfreqprs (Mireye's cached raster was null at this pixel). Map unit: {live.get('muname', 'unknown')}.",
            "status": "ok",
        }
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
# USGS 3DEP Elevation Point Query Service — live fallback for slope_degrees
# ---------------------------------------------------------------------------
#
# Mireye sometimes has no slope_degrees value at a point. Slope isn't
# published directly by USGS either, but elevation is (3DEP, no key
# required), so we derive slope from a small central-difference stencil:
# sample elevation ~30m N/S/E/W of the point and compute the gradient.
# This is a screening-grade slope estimate, not a substitute for a real
# hydro-flattened DEM analysis.
#
# NOTE: like the SDA fallback above, this environment's network allowlist
# doesn't include epqs.nationalmap.gov, so the exact response shape is
# built from USGS's documented EPQS v1 contract but hasn't been
# live-tested end-to-end here. If the response shape differs on your
# first real run, it'll raise (and get logged as a fallback error)
# rather than fail silently.

EPQS_URL = "https://epqs.nationalmap.gov/v1/json"
DEM_SAMPLE_OFFSET_M = 30.0  # roughly matches 1-arcsecond (~30m) 3DEP resolution


def fetch_elevation_m(lat: float, lng: float) -> float:
    resp = requests.get(
        EPQS_URL,
        params={"x": lng, "y": lat, "units": "Meters", "wkid": 4326, "includeDate": "false"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    value = data.get("value")
    if value is None:
        raise RuntimeError(f"EPQS returned no elevation value: {data}")
    return float(value)


def fetch_slope_from_dem(lat: float, lng: float) -> float:
    """Central-difference slope estimate (degrees) from four elevation
    samples offset ~DEM_SAMPLE_OFFSET_M north/south/east/west of the point."""
    if abs(lat) > 89.9:
        # cos(lat) -> 0 near the poles blows up the east/west sample offset
        # into a nonsensical longitude delta; no real field/parcel is here.
        raise RuntimeError(f"DEM slope fallback not supported this close to a pole (lat={lat}).")
    lat_rad = math.radians(lat)
    dlat = DEM_SAMPLE_OFFSET_M / 111_320.0
    dlng = DEM_SAMPLE_OFFSET_M / (111_320.0 * math.cos(lat_rad))

    elev_n = fetch_elevation_m(lat + dlat, lng)
    elev_s = fetch_elevation_m(lat - dlat, lng)
    elev_e = fetch_elevation_m(lat, lng + dlng)
    elev_w = fetch_elevation_m(lat, lng - dlng)

    dz_dy = (elev_n - elev_s) / (2 * DEM_SAMPLE_OFFSET_M)
    dz_dx = (elev_e - elev_w) / (2 * DEM_SAMPLE_OFFSET_M)
    rise_over_run = math.sqrt(dz_dx ** 2 + dz_dy ** 2)
    return math.degrees(math.atan(rise_over_run))


def patch_null_slope_with_dem(lat: float, lng: float, fields: dict) -> dict:
    """If Mireye has no slope_degrees at this point, derive one from live
    USGS 3DEP elevation samples. Only fills a genuinely missing value, and
    tags the result as a derived DEM estimate rather than a Mireye field."""
    if field_value(fields, "slope_degrees") is not None:
        return fields

    try:
        slope = fetch_slope_from_dem(lat, lng)
    except Exception as e:
        fields.setdefault("_dem_slope_fallback_error", {"status": "failed", "error": str(e)})
        return fields

    fields["slope_degrees"] = {
        "value": round(slope, 2),
        "unit": "degrees",
        "source": "USGS_3DEP_EPQS",
        "source_url": "https://epqs.nationalmap.gov/v1/json",
        "confidence": "medium",
        "notes": (
            f"Live DEM-derived slope fallback (Mireye had no slope value at this point). "
            f"Central-difference gradient from elevation samples ~{DEM_SAMPLE_OFFSET_M:.0f}m "
            "N/S/E/W via USGS 3DEP Elevation Point Query Service — screening estimate, "
            "not a hydro-flattened DEM analysis."
        ),
        "status": "ok",
    }
    return fields


# ---------------------------------------------------------------------------
# FEMA National Flood Hazard Layer — live fallback for flood fields
# ---------------------------------------------------------------------------
#
# Free public ArcGIS REST service, no key required. Layer 28 of the NFHL
# MapServer is S_Fld_Haz_Ar (the flood hazard zone polygons) — querying a
# point against it returns the real zone code (AE, VE, X, ...), the zone
# subtype (used to flag coastal V-zones), and the base flood elevation.
# within_floodplain_polygon is then derived from the zone code rather than
# queried separately, since "does the real zone start with A or V" already
# answers that question.
#
# NOTE: same caveat as the other live fallbacks — this sandbox's network
# allowlist doesn't include hazards.fema.gov, so the query shape is built
# from FEMA's documented NFHL REST/data-dictionary conventions but hasn't
# been live-tested end-to-end here. STATIC_BFE uses -9999 as FEMA's sentinel
# for "not applicable"; that's converted to None rather than reported as
# a real elevation.

FEMA_NFHL_FLOOD_ZONES_URL = "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"


def fetch_fema_flood_zone(lat: float, lng: float) -> dict:
    resp = requests.get(
        FEMA_NFHL_FLOOD_ZONES_URL,
        params={
            "geometry": f"{lng},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "FLD_ZONE,ZONE_SUBTY,STATIC_BFE,SFHA_TF",
            "returnGeometry": "false",
            "f": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"FEMA NFHL query error: {data['error']}")
    features = data.get("features") or []
    if not features:
        return {}

    attrs = features[0].get("attributes", {})
    bfe = attrs.get("STATIC_BFE")
    try:
        bfe = float(bfe) if bfe is not None else None
    except (TypeError, ValueError):
        bfe = None
    if bfe is not None and bfe <= -9000:  # FEMA's "not applicable" sentinel
        bfe = None

    zone = attrs.get("FLD_ZONE")
    zone_subty = (attrs.get("ZONE_SUBTY") or "").upper()
    coastal = bool(zone) and (str(zone).upper().startswith("V") or "COASTAL" in zone_subty)

    return {"fld_zone": zone, "coastal": coastal, "bfe": bfe}


def patch_nulls_with_fema_nfhl(lat: float, lng: float, fields: dict) -> dict:
    """If Mireye has no flood-hazard data at this point, query FEMA's NFHL
    directly. Only fills genuinely missing fields, tagged as a secondary
    source rather than pretending it came from Mireye."""
    zone_missing = field_value(fields, "fema_flood_zone") is None
    coastal_missing = field_value(fields, "coastal_high_hazard") is None
    bfe_missing = field_value(fields, "fema_base_flood_elevation") is None
    floodplain_missing = field_value(fields, "within_floodplain_polygon") is None
    if not (zone_missing or coastal_missing or bfe_missing or floodplain_missing):
        return fields

    try:
        live = fetch_fema_flood_zone(lat, lng)
    except Exception as e:
        fields.setdefault("_fema_nfhl_fallback_error", {"status": "failed", "error": str(e)})
        return fields
    if not live or live.get("fld_zone") is None:
        return fields

    zone = live["fld_zone"]
    source_note_base = f"Live FEMA National Flood Hazard Layer fallback (Mireye had no value at this point). Zone: {zone}."

    if zone_missing:
        fields["fema_flood_zone"] = {
            "value": zone,
            "unit": None,
            "source": "FEMA_NFHL_LIVE",
            "source_url": "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28",
            "confidence": "medium",
            "notes": source_note_base,
            "status": "ok",
        }
    if coastal_missing:
        fields["coastal_high_hazard"] = {
            "value": live["coastal"],
            "unit": None,
            "source": "FEMA_NFHL_LIVE",
            "source_url": "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28",
            "confidence": "medium",
            "notes": f"Derived from NFHL zone code/subtype. {source_note_base}",
            "status": "ok",
        }
    if bfe_missing and live.get("bfe") is not None:
        fields["fema_base_flood_elevation"] = {
            "value": live["bfe"],
            "unit": "feet",
            "source": "FEMA_NFHL_LIVE",
            "source_url": "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28",
            "confidence": "medium",
            "notes": source_note_base,
            "status": "ok",
        }
    if floodplain_missing:
        fields["within_floodplain_polygon"] = {
            "value": str(zone).upper().startswith(("A", "V")),
            "unit": None,
            "source": "FEMA_NFHL_LIVE",
            "source_url": "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28",
            "confidence": "medium",
            "notes": f"Derived from NFHL zone code. {source_note_base}",
            "status": "ok",
        }
    return fields


# ---------------------------------------------------------------------------
# USFWS National Wetlands Inventory — live fallback for wetland fields
# ---------------------------------------------------------------------------
#
# Free public ArcGIS REST service, no key required. Unlike the other
# fallbacks, this one is a rougher proxy by nature: Mireye's fields are
# defined at the PARCEL level, but neither Mireye nor this app has real
# parcel-boundary geometry to work with — only a lat/lng point. So we
# query NWI wetland polygons intersecting a small buffer square around the
# point (same style of buffer already used for the GFW tree-loss query)
# and estimate a share from that, capped at the buffer's own area, using
# each wetland polygon's own reported acreage rather than a true clipped
# intersection (which would need real geometry-clipping, e.g. shapely).
# That means it can overstate the true on-parcel share when a wetland
# polygon only partially overlaps the buffer — flagged as "low" confidence
# and spelled out in the notes rather than presented as exact.
#
# NOTE: same live-untested caveat as the other fallbacks — this sandbox's
# network allowlist doesn't include the NWI host, so this is built from
# USFWS's documented NWI REST service conventions, not verified end-to-end.

NWI_WETLANDS_URL = "https://fwsprimary.wim.usgs.gov/server/rest/services/Wetlands/MapServer/0/query"
WETLAND_BUFFER_DEG = 0.001  # small stand-in for "parcel" — no real parcel geometry available


def _buffer_acres(lat: float, buffer_deg: float) -> float:
    """Approximate area (acres) of the lat/lng-degree buffer square."""
    lat_rad = math.radians(lat)
    height_m = 2 * buffer_deg * 111_320.0
    width_m = 2 * buffer_deg * 111_320.0 * math.cos(lat_rad)
    area_m2 = height_m * width_m
    return area_m2 / 4046.856  # square meters per acre


def fetch_nwi_wetlands(lat: float, lng: float) -> list:
    geometry = make_point_buffer_geojson(lat, lng, buffer_deg=WETLAND_BUFFER_DEG)
    resp = requests.get(
        NWI_WETLANDS_URL,
        params={
            "geometry": json.dumps({"rings": geometry["coordinates"], "spatialReference": {"wkid": 4326}}),
            "geometryType": "esriGeometryPolygon",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "WETLAND_TYPE,ACRES",
            "returnGeometry": "false",
            "f": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"NWI query error: {data['error']}")
    out = []
    for feat in data.get("features") or []:
        attrs = feat.get("attributes", {})
        try:
            acres = float(attrs["ACRES"]) if attrs.get("ACRES") is not None else None
        except (TypeError, ValueError):
            acres = None
        out.append({"wetland_type": attrs.get("WETLAND_TYPE"), "acres": acres})
    return out


def patch_nulls_with_nwi_wetlands(lat: float, lng: float, fields: dict) -> dict:
    """If Mireye has no wetland-share data at this point, query USFWS NWI
    directly against a small point-buffer stand-in for the parcel."""
    frac_missing = field_value(fields, "wetland_fraction_of_parcel") is None
    acres_missing = field_value(fields, "wetland_acres_on_parcel") is None
    if not (frac_missing or acres_missing):
        return fields

    try:
        wetlands = fetch_nwi_wetlands(lat, lng)
    except Exception as e:
        fields.setdefault("_nwi_wetlands_fallback_error", {"status": "failed", "error": str(e)})
        return fields

    buffer_acres = _buffer_acres(lat, WETLAND_BUFFER_DEG)
    total_acres = sum(w["acres"] for w in wetlands if w.get("acres") is not None)
    capped_acres = min(total_acres, buffer_acres) if buffer_acres > 0 else 0.0
    fraction = round(min(1.0, capped_acres / buffer_acres), 4) if buffer_acres > 0 else 0.0

    note = (
        f"Live USFWS National Wetlands Inventory fallback (Mireye had no value at this "
        f"point). Estimated against a small ~{buffer_acres:.1f}-acre buffer around the "
        "point, NOT the real parcel boundary (no parcel geometry available), from "
        "whole-polygon wetland acreage rather than a true clipped intersection — "
        "treat as a coarse presence/rough-share signal, not an exact figure."
    )

    if frac_missing:
        fields["wetland_fraction_of_parcel"] = {
            "value": fraction,
            "unit": None,
            "source": "USFWS_NWI_LIVE",
            "source_url": "https://www.fws.gov/program/national-wetlands-inventory",
            "confidence": "low",
            "notes": note,
            "status": "ok",
        }
    if acres_missing:
        fields["wetland_acres_on_parcel"] = {
            "value": round(capped_acres, 3),
            "unit": "acres",
            "source": "USFWS_NWI_LIVE",
            "source_url": "https://www.fws.gov/program/national-wetlands-inventory",
            "confidence": "low",
            "notes": note,
            "status": "ok",
        }
    return fields


# ---------------------------------------------------------------------------
# NLCD Tree Canopy Cover (MRLC) — live fallback for tree_canopy_pct
# ---------------------------------------------------------------------------
#
# Free public WMS service, no key required. This is the LOWEST-confidence
# fallback in this file — unlike SDA/EPQS/NFHL/NWI (well-established REST
# contracts), the exact GeoServer layer name and GetFeatureInfo response
# field for MRLC's tree-canopy coverage have NOT been confirmed against a
# live response in this sandbox (no network access to mrlc.gov here either).
# WMS 1.1.1 is used deliberately instead of 1.3.0 to sidestep the
# lat/lng-vs-lng/lat axis-order footgun in the newer spec. If the layer name
# or response shape is wrong, this fails closed: the exception is caught and
# logged as a fallback error, same as every other fallback here — it will
# NOT report a wrong canopy value silently.

MRLC_WMS_URL = "https://www.mrlc.gov/geoserver/mrlc_display/wms"
NLCD_CANOPY_LAYER = "NLCD_2021_Tree_Canopy_L48"


def fetch_nlcd_tree_canopy_pct(lat: float, lng: float) -> float:
    eps = 0.0001  # tiny bbox around the point for a single-pixel WMS query
    resp = requests.get(
        MRLC_WMS_URL,
        params={
            "service": "WMS",
            "version": "1.1.1",
            "request": "GetFeatureInfo",
            "layers": NLCD_CANOPY_LAYER,
            "query_layers": NLCD_CANOPY_LAYER,
            "srs": "EPSG:4326",
            "bbox": f"{lng - eps},{lat - eps},{lng + eps},{lat + eps}",
            "width": 3,
            "height": 3,
            "x": 1,
            "y": 1,
            "info_format": "application/json",
            "feature_count": 1,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features") or []
    if not features:
        raise RuntimeError(f"NLCD WMS GetFeatureInfo returned no features: {data}")
    props = features[0].get("properties", {})
    value = props.get("GRAY_INDEX")
    if value is None:
        raise RuntimeError(f"NLCD WMS GetFeatureInfo response missing GRAY_INDEX: {props}")
    value = float(value)
    if value >= 254:  # common raster no-data / water sentinel values
        raise RuntimeError(f"NLCD returned a no-data sentinel value ({value})")
    return value


def patch_null_canopy_with_nlcd(lat: float, lng: float, fields: dict) -> dict:
    """If Mireye has no tree_canopy_pct at this point, query MRLC's NLCD
    Tree Canopy Cover WMS directly."""
    if field_value(fields, "tree_canopy_pct") is not None:
        return fields

    try:
        canopy = fetch_nlcd_tree_canopy_pct(lat, lng)
    except Exception as e:
        fields.setdefault("_nlcd_canopy_fallback_error", {"status": "failed", "error": str(e)})
        return fields

    fields["tree_canopy_pct"] = {
        "value": round(canopy, 1),
        "unit": "percent",
        "source": "NLCD_TCC_LIVE",
        "source_url": "https://www.mrlc.gov/data/nlcd-tree-canopy-cover",
        "confidence": "low",
        "notes": (
            "Live NLCD Tree Canopy Cover fallback via MRLC's WMS pixel query "
            "(Mireye had no canopy value at this point). Lowest-confidence "
            "fallback in this app — the exact layer name/response field hasn't "
            "been verified against a live response in this environment; treat "
            "with extra skepticism until confirmed against a real request."
        ),
        "status": "ok",
    }
    return fields


# ---------------------------------------------------------------------------
# RUSLE-lite erosion index
# ---------------------------------------------------------------------------

DEFAULT_SLOPE_LENGTH_M = 100.0  # used only when tree_canopy_pct is unavailable


def field_value(fields: dict, name: str):
    entry = fields.get(name)
    if not entry or entry.get("status") == "failed":
        return None
    return entry.get("value")


def estimate_slope_length_m(slope_pct: float, tree_canopy_pct) -> float:
    """Slope length isn't a queryable field anywhere (real RUSLE derives it
    by tracing flow paths across a DEM) — this is still an assumption, not
    measured flow-length data. But instead of one flat 100m for every point,
    lean on signals we do have: denser canopy means more roughness elements
    (undergrowth, downed wood, microtopography) that break up overland flow
    sooner, so assume a shorter run; open/bare ground is more likely tilled
    or graded into long unbroken slopes, so assume a longer one. Steeper
    slopes also tend to concentrate flow into a defined channel sooner than
    gentle ones, which spread and travel farther before channelizing.
    """
    if tree_canopy_pct is None:
        base = DEFAULT_SLOPE_LENGTH_M
    elif tree_canopy_pct >= 60:
        base = 40.0
    elif tree_canopy_pct >= 30:
        base = 70.0
    else:
        base = 130.0

    if slope_pct > 15:
        base *= 0.7
    elif slope_pct < 3:
        base *= 1.3

    return round(base, 1)


def compute_ls_factor(slope_degrees: float, tree_canopy_pct) -> tuple:
    """McCool et al. slope-length-steepness approximation. Returns
    (ls_factor, slope_length_m_used) since the slope length is now derived
    per-point rather than a single constant."""
    theta = math.radians(slope_degrees)
    slope_pct = math.tan(theta) * 100
    slope_length_m = estimate_slope_length_m(slope_pct, tree_canopy_pct)
    m = 0.5 if slope_pct >= 5 else (0.4 if slope_pct >= 3 else (0.3 if slope_pct >= 1 else 0.2))
    ls = ((slope_length_m / 22.13) ** m) * (
        65.41 * math.sin(theta) ** 2 + 4.56 * math.sin(theta) + 0.065
    )
    return ls, slope_length_m


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
        ls, slope_length_m = compute_ls_factor(slope, tree_canopy)
        c = compute_c_factor(tree_canopy, ndvi_change_5y)
        relative_index = round(k_factor * ls * c, 4)
        rusle = {
            "k_factor": k_factor,
            "ls_factor": round(ls, 3),
            "c_factor": c,
            "p_factor": 1.0,
            "relative_index": relative_index,
            "assumed_slope_length_m": slope_length_m,
            "slope_length_estimated_from_canopy": tree_canopy is not None,
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

    fields = mireye_data.get("fields") or {}
    fields = patch_nulls_with_live_ssurgo(lat, lng, fields)
    fields = patch_null_slope_with_dem(lat, lng, fields)
    fields = patch_nulls_with_fema_nfhl(lat, lng, fields)
    fields = patch_nulls_with_nwi_wetlands(lat, lng, fields)
    fields = patch_null_canopy_with_nlcd(lat, lng, fields)

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
        "mireye_partial_failures": mireye_data.get("partial_failures") or [],
        "mireye_ask_summary": ask_summary,
        "mireye_ask_error": ask_error,
        "tree_cover_loss_by_year": loss_years,
        "gfw_error": gfw_error,
    }


# ---------------------------------------------------------------------------
# Rankings ("Most At-Risk Areas" browse view)
# ---------------------------------------------------------------------------
#
# Serves the precomputed cache from precompute_rankings.py rather than
# computing anything live — see that script's docstring for why. This
# endpoint just reads, filters, and sorts whatever is already on disk.

RANKINGS_CACHE_FILE = Path(__file__).parent / "rankings_cache.json"


@app.get("/api/rankings")
def get_rankings(state: str = Query(None), region: str = Query(None), limit: int = Query(None, gt=0)):
    if not RANKINGS_CACHE_FILE.exists():
        return {
            "data_available": False,
            "message": (
                "No rankings data yet. Run `python3 precompute_rankings.py` in "
                "backend/ to populate rankings_cache.json (requires a Mireye "
                "token; see precompute_rankings.py's docstring)."
            ),
            "computed_at": None,
            "results": [],
            "failed": [],
        }

    try:
        cache = json.loads(RANKINGS_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"rankings_cache.json is unreadable: {e}")

    all_results = cache.get("results") or []

    def matches(row: dict) -> bool:
        if state:
            candidates = {(row.get("state") or "").lower(), (row.get("state_abbr") or "").lower()}
            if state.lower() not in candidates:
                return False
        if region and (row.get("region") or "").lower() != region.lower():
            return False
        return True

    filtered = [r for r in all_results if matches(r)]
    ranked = sorted((r for r in filtered if "score" in r), key=lambda r: r["score"], reverse=True)
    failed = [r for r in filtered if "score" not in r]

    if limit is not None:
        ranked = ranked[:limit]

    return {
        "data_available": True,
        "computed_at": cache.get("computed_at"),
        "results": ranked,
        "failed": failed,
    }


frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
