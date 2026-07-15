"""
Test suite for precompute_rankings.py — the batch script that computes
/api/rankings' cache file. Everything that hits main.py's live fetch
functions is mocked; these tests verify the script's own logic (CSV
parsing, concurrency, ranking/truncation, failure handling).
"""

import csv
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

import main
import precompute_rankings as pr


@pytest.fixture
def counties_csv(tmp_path):
    rows = [
        {"county": f"County{i}", "state": "Teststate", "state_abbr": "TS", "region": "West",
         "lat": str(40.0 + i), "lng": str(-100.0 - i)}
        for i in range(10)
    ]
    path = tmp_path / "counties.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["county", "state", "state_abbr", "region", "lat", "lng"])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def patch_fallbacks():
    """Patches every live fallback + GFW as a pass-through/no-op, so
    compute_one only exercises the Mireye fetch + scoring it's under test for."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(main, "patch_nulls_with_live_ssurgo", side_effect=lambda lat, lng, f: f))
        stack.enter_context(patch.object(main, "patch_null_slope_with_dem", side_effect=lambda lat, lng, f: f))
        stack.enter_context(patch.object(main, "patch_nulls_with_fema_nfhl", side_effect=lambda lat, lng, f: f))
        stack.enter_context(patch.object(main, "patch_nulls_with_nwi_wetlands", side_effect=lambda lat, lng, f: f))
        stack.enter_context(patch.object(main, "patch_null_canopy_with_nlcd", side_effect=lambda lat, lng, f: f))
        stack.enter_context(patch.object(main, "fetch_gfw_tree_cover_loss", return_value=[]))
        yield


class TestLoadCounties:
    def test_parses_all_rows(self, counties_csv):
        rows = pr.load_counties(counties_csv)
        assert len(rows) == 10
        assert rows[0]["county"] == "County0"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pr.load_counties(tmp_path / "nope.csv")


class TestBaseResult:
    def test_coerces_lat_lng_to_float(self):
        row = {"county": "X", "state": "Y", "state_abbr": "Z", "region": "West", "lat": "40.5", "lng": "-100.5"}
        result = pr._base_result(row)
        assert result["lat"] == 40.5
        assert result["lng"] == -100.5
        assert isinstance(result["lat"], float)


class TestComputeOne:
    def test_success_path_returns_score(self, patch_fallbacks):
        row = {"county": "X", "state": "Y", "state_abbr": "Z", "region": "West", "lat": "40.0", "lng": "-100.0"}
        with patch.object(main, "fetch_mireye_hazard_data", return_value={
            "fields": {"slope_degrees": {"status": "ok", "value": 10},
                       "soil_erodibility_k_factor": {"status": "ok", "value": 0.3}}
        }), patch.object(main, "fetch_mean_annual_precip_mm", return_value=900.0):
            result = pr.compute_one(row)
        assert "score" in result
        assert "error" not in result
        assert result["county"] == "X"

    def test_mireye_failure_recorded_as_error(self):
        row = {"county": "X", "state": "Y", "state_abbr": "Z", "region": "West", "lat": "40.0", "lng": "-100.0"}
        with patch.object(main, "fetch_mireye_hazard_data", side_effect=RuntimeError("token missing")):
            result = pr.compute_one(row)
        assert "error" in result
        assert "score" not in result
        assert "token missing" in result["error"]

    def test_top_factors_limited_to_three_high_or_moderate(self, patch_fallbacks):
        row = {"county": "X", "state": "Y", "state_abbr": "Z", "region": "West", "lat": "40.0", "lng": "-100.0"}
        fake_risk = {
            "score": 5, "level": "moderate",
            "factors": [
                {"label": "A", "detail": "", "severity": "high"},
                {"label": "B", "detail": "", "severity": "moderate"},
                {"label": "C", "detail": "", "severity": "low"},
                {"label": "D", "detail": "", "severity": "high"},
                {"label": "E", "detail": "", "severity": "moderate"},
            ],
            "rusle_lite": None,
        }
        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": {}}), \
             patch.object(main, "score_erosion_risk", return_value=fake_risk):
            result = pr.compute_one(row)
        assert result["top_factors"] == ["A", "B", "D"]


class TestRunConcurrent:
    def test_all_counties_processed(self, counties_csv, patch_fallbacks):
        counties = pr.load_counties(counties_csv)
        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": {}}), \
             patch.object(main, "score_erosion_risk", return_value={"score": 1, "level": "low", "factors": [], "rusle_lite": None}):
            results = pr.run_concurrent(counties, workers=5)
        assert len(results) == 10

    def test_partial_failures_dont_lose_other_results(self, counties_csv, patch_fallbacks):
        counties = pr.load_counties(counties_csv)

        def flaky(lat, lng):
            if int(lat) % 2 == 0:
                raise RuntimeError("simulated failure")
            return {"fields": {}}

        with patch.object(main, "fetch_mireye_hazard_data", side_effect=flaky), \
             patch.object(main, "score_erosion_risk", return_value={"score": 1, "level": "low", "factors": [], "rusle_lite": None}):
            results = pr.run_concurrent(counties, workers=5)
        assert len(results) == 10
        assert sum(1 for r in results if "error" in r) == 5

    def test_unexpected_exception_in_compute_one_is_caught(self, counties_csv):
        counties = pr.load_counties(counties_csv)
        with patch.object(pr, "compute_one", side_effect=ValueError("boom")):
            results = pr.run_concurrent(counties, workers=5)
        assert len(results) == 10
        assert all("error" in r for r in results)


class TestMain:
    def test_keeps_only_top_n_sorted_by_score(self, counties_csv, patch_fallbacks, tmp_path):
        out_file = tmp_path / "out.json"

        def fake_score(fields, loss_years, lat, lng):
            idx = int(lat - 40.0)
            return {"score": idx, "level": "low", "factors": [], "rusle_lite": None}

        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": {}}), \
             patch.object(main, "score_erosion_risk", side_effect=fake_score):
            pr.main(counties_csv, out_file, top_n=3, workers=5)

        payload = json.loads(out_file.read_text())
        assert payload["county_count"] == 3
        assert [r["county"] for r in payload["results"]] == ["County9", "County8", "County7"]
        assert payload["failed_count"] == 0
        assert "computed_at" in payload
        assert payload["counties_computed"] == 10

    def test_failed_counties_counted_but_excluded_from_results(self, counties_csv, tmp_path):
        out_file = tmp_path / "out.json"

        with patch.object(main, "fetch_mireye_hazard_data", side_effect=RuntimeError("down")):
            pr.main(counties_csv, out_file, top_n=5, workers=5)

        payload = json.loads(out_file.read_text())
        assert payload["county_count"] == 0
        assert payload["failed_count"] == 10
        assert payload["results"] == []

    def test_top_n_larger_than_available_is_harmless(self, counties_csv, patch_fallbacks, tmp_path):
        out_file = tmp_path / "out.json"
        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": {}}), \
             patch.object(main, "score_erosion_risk", return_value={"score": 1, "level": "low", "factors": [], "rusle_lite": None}):
            pr.main(counties_csv, out_file, top_n=9999, workers=5)

        payload = json.loads(out_file.read_text())
        assert payload["county_count"] == 10
