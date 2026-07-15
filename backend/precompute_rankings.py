#!/usr/bin/env python3
"""
Precompute risk rankings for the "Most At-Risk Areas" browse view.

WHY THIS EXISTS AS A SEPARATE SCRIPT, NOT A LIVE ENDPOINT
-----------------------------------------------------------
/api/risk computes one point at a time via several live external calls
(Mireye /v1/fetch + /v1/ask, GFW, and up to five fallback sources: USDA
SDA, USGS 3DEP, FEMA NFHL, USFWS NWI, MRLC NLCD). Running that pipeline
live on every filter click would be slow and would hammer several free
public APIs' rate limits for what should be a fast, filterable browse UI.

So this script runs the same pipeline once (per run) over a reference
list of counties and caches the results to rankings_cache.json, which
/api/rankings then just reads and filters/sorts instantly. Run this
periodically (e.g. a weekly cron job) to keep the cache fresh — it is
NOT invoked automatically by the web app.

ONLY THE TOP N SURVIVE IN THE OUTPUT, BUT EVERY COUNTY STILL GETS CHECKED
----------------------------------------------------------------------------
The final cache keeps only the worst --top counties (default 100), but
there's no way to know in advance which ones those are — some pass over
every county is unavoidable to find them correctly. An earlier version of
this script tried to dodge that by screening all counties with a cheap
1-call pass and only fully computing a smaller candidate pool. That
turned out not to help: Mireye's own /v1/fetch call is the dominant cost
(~5s) regardless of how many fallback calls follow it, so the "cheap"
screening pass took about as long as just computing everything properly
would have — while also being less accurate (a county whose risk-relevant
fields are null in Mireye's raw data, recoverable only via a fallback,
would score artificially low in a screening-only pass).

The actual fix for "3,144 slow sequential network calls" is concurrency,
not trimming which calls happen — see WORKERS below. This script now runs
the full accurate pipeline (all fallbacks + GFW) on every county, just
with many requests in flight at once, and keeps the top N at the end.

CONCURRENCY (WORKERS)
-----------------------
--workers (default 10) sets how many counties are processed at once via a
thread pool. Each county's own calls are still sequential (Mireye, then
its fallbacks, then GFW) — only the across-county work is parallelized.
Mireye's actual rate limits aren't known here, so this is somewhat
exploratory: a request that gets rate-limited under load just fails
gracefully for that one county (recorded as an error, same as any other
failure) rather than crashing the run. Lower --workers if you see a lot
of failures that look rate-limit-related.

COUNTY DATA
------------
counties_full.csv has all 3,144 US counties/county-equivalents (50 states
+ DC; territories excluded), parsed directly from the official US Census
Bureau 2023 Gazetteer county file (INTPTLAT/INTPTLONG centroids) — not
hand-compiled. counties_seed.csv (51 rows, one biggest-city county per
state) also still exists but isn't used by default: it systematically
under-represents real hazard because major metro areas tend to sit on
the flattest, least flood/landslide-prone land in a state.

USAGE
------
    cd backend
    python3 precompute_rankings.py [--counties-file counties_full.csv] [--out rankings_cache.json] \\
        [--top 100] [--workers 10]

Requires the same environment as running the server itself: a Mireye
token (MIREYE_BEARER_TOKEN or ~/.config/mireye-mcp/credentials.json) and
GFW_API_KEY if you want tree-cover-loss data included. A county whose
Mireye fetch fails is recorded with an error rather than aborting the
whole run.
"""

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import main as app  # reuse the exact same fetch/fallback/scoring logic as /api/risk

DEFAULT_COUNTIES_FILE = Path(__file__).parent / "counties_full.csv"
DEFAULT_OUT_FILE = Path(__file__).parent / "rankings_cache.json"
DEFAULT_TOP_N = 100
DEFAULT_WORKERS = 10  # concurrent counties; each county's own calls are still sequential

_print_lock = threading.Lock()


def load_counties(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _base_result(county_row: dict) -> dict:
    return {
        "county": county_row["county"],
        "state": county_row["state"],
        "state_abbr": county_row["state_abbr"],
        "region": county_row["region"],
        "lat": float(county_row["lat"]),
        "lng": float(county_row["lng"]),
    }


def compute_one(county_row: dict) -> dict:
    """The full pipeline — all live fallbacks + GFW — same as /api/risk
    computes for a single point."""
    result = _base_result(county_row)
    lat, lng = result["lat"], result["lng"]

    try:
        mireye_data = app.fetch_mireye_hazard_data(lat, lng)
    except Exception as e:
        result["error"] = f"Mireye /v1/fetch failed: {e}"
        return result

    fields = mireye_data.get("fields") or {}
    fields = app.patch_nulls_with_live_ssurgo(lat, lng, fields)
    fields = app.patch_null_slope_with_dem(lat, lng, fields)
    fields = app.patch_nulls_with_fema_nfhl(lat, lng, fields)
    fields = app.patch_nulls_with_nwi_wetlands(lat, lng, fields)
    fields = app.patch_null_canopy_with_nlcd(lat, lng, fields)

    try:
        loss_years = app.fetch_gfw_tree_cover_loss(lat, lng)
    except Exception:
        loss_years = []  # non-fatal, same as /api/risk's own handling

    risk = app.score_erosion_risk(fields, loss_years, lat, lng)
    result["score"] = risk["score"]
    result["level"] = risk["level"]
    result["top_factors"] = [f["label"] for f in risk["factors"] if f["severity"] in ("high", "moderate")][:3]
    return result


def run_concurrent(counties: list, workers: int) -> list:
    results = []
    total = len(counties)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_county = {executor.submit(compute_one, c): c for c in counties}
        for future in as_completed(future_to_county):
            county_row = future_to_county[future]
            done += 1
            try:
                result = future.result()
            except Exception as e:
                # a bug in our own code shouldn't lose the rest of the batch either
                result = {**_base_result(county_row), "error": f"unexpected error: {e}"}
            results.append(result)
            with _print_lock:
                print(f"[{done}/{total}] {county_row['county']}, {county_row['state_abbr']}...", file=sys.stderr)
    return results


def main(counties_file: Path, out_file: Path, top_n: int, workers: int) -> None:
    counties = load_counties(counties_file)
    print(f"Computing risk for {len(counties)} counties with {workers} concurrent workers...", file=sys.stderr)

    results = run_concurrent(counties, workers)

    ranked = sorted((r for r in results if "score" in r), key=lambda r: r["score"], reverse=True)
    failed = [r for r in results if "score" not in r]
    top_results = ranked[:top_n]

    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "method": f"full pipeline on all {len(counties)} counties, kept the top {len(top_results)} by score",
        "counties_computed": len(counties),
        "county_count": len(top_results),
        "failed_count": len(failed),
        "results": top_results,
    }
    out_file.write_text(json.dumps(payload, indent=2))
    print(
        f"Wrote top {len(top_results)} of {len(counties)} computed counties "
        f"({len(failed)} failed) to {out_file}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counties-file", type=Path, default=DEFAULT_COUNTIES_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                         help="how many counties to keep in the final output")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help="how many counties to process concurrently")
    args = parser.parse_args()
    main(args.counties_file, args.out, args.top, args.workers)
