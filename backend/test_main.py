"""
Test suite for backend/main.py.

Every external HTTP call (Mireye, GFW, USDA SDA, USGS 3DEP, FEMA NFHL,
USFWS NWI, MRLC NLCD, Open-Meteo) is mocked — this sandbox's network
allowlist can't reach any of these real services, so these tests verify
the app's own logic (parsing, gating, merging, error-handling, math)
rather than the third parties' actual responses.
"""

import math
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

import main


def mock_response(json_data=None, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_data if json_data is not None else {}
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = main.requests.exceptions.HTTPError("boom")
    return resp


# ---------------------------------------------------------------------------
# Pure math / helper functions
# ---------------------------------------------------------------------------

class TestShrinkSwellClass:
    def test_low(self):
        assert main._shrink_swell_class_from_lep(2) == "Low"

    def test_moderate(self):
        assert main._shrink_swell_class_from_lep(4) == "Moderate"

    def test_high(self):
        assert main._shrink_swell_class_from_lep(7) == "High"

    def test_very_high(self):
        assert main._shrink_swell_class_from_lep(15) == "Very high"

    def test_boundary_values_are_inclusive_of_lower_bound(self):
        assert main._shrink_swell_class_from_lep(3) == "Moderate"
        assert main._shrink_swell_class_from_lep(6) == "High"
        assert main._shrink_swell_class_from_lep(9) == "Very high"

    def test_none_returns_none(self):
        assert main._shrink_swell_class_from_lep(None) is None

    def test_non_numeric_returns_none(self):
        assert main._shrink_swell_class_from_lep("not-a-number") is None


class TestFieldValue:
    def test_missing_key_returns_none(self):
        assert main.field_value({}, "slope_degrees") is None

    def test_failed_status_returns_none(self):
        fields = {"slope_degrees": {"status": "failed", "value": 12}}
        assert main.field_value(fields, "slope_degrees") is None

    def test_ok_value_returned(self):
        fields = {"slope_degrees": {"status": "ok", "value": 12.5}}
        assert main.field_value(fields, "slope_degrees") == 12.5

    def test_falsy_but_valid_zero_value_is_not_swallowed(self):
        # 0 is a legitimate real value (e.g. flat ground, no canopy) — must
        # not be conflated with "missing".
        fields = {"tree_canopy_pct": {"status": "ok", "value": 0}}
        assert main.field_value(fields, "tree_canopy_pct") == 0

    def test_falsy_but_valid_false_value_is_not_swallowed(self):
        fields = {"coastal_high_hazard": {"status": "ok", "value": False}}
        assert main.field_value(fields, "coastal_high_hazard") is False

    def test_empty_dict_entry_returns_none(self):
        assert main.field_value({"x": {}}, "x") is None


class TestEstimateSlopeLength:
    def test_missing_canopy_uses_default(self):
        assert main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=None) == main.DEFAULT_SLOPE_LENGTH_M

    def test_dense_canopy_shortens_length(self):
        assert main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=80) == 40.0

    def test_moderate_canopy(self):
        assert main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=45) == 70.0

    def test_open_ground_lengthens(self):
        assert main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=5) == 130.0

    def test_steep_slope_shortens_further(self):
        base = main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=80)
        steep = main.estimate_slope_length_m(slope_pct=20, tree_canopy_pct=80)
        assert steep < base

    def test_gentle_slope_lengthens_further(self):
        base = main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=80)
        gentle = main.estimate_slope_length_m(slope_pct=1, tree_canopy_pct=80)
        assert gentle > base

    def test_canopy_boundary_60_is_dense(self):
        assert main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=60) == 40.0

    def test_canopy_boundary_30_is_moderate(self):
        assert main.estimate_slope_length_m(slope_pct=5, tree_canopy_pct=30) == 70.0


class TestComputeLsFactor:
    def test_returns_tuple_of_ls_and_length(self):
        # slope_degrees=5 -> slope_pct ~8.7%, inside the 3-15% band where
        # neither the steep nor gentle multiplier applies, isolating the
        # canopy-only base length.
        ls, length = main.compute_ls_factor(slope_degrees=5, tree_canopy_pct=50)
        assert isinstance(ls, float)
        assert length == 70.0

    def test_steeper_slope_increases_ls(self):
        ls_flat, _ = main.compute_ls_factor(slope_degrees=2, tree_canopy_pct=50)
        ls_steep, _ = main.compute_ls_factor(slope_degrees=20, tree_canopy_pct=50)
        assert ls_steep > ls_flat

    def test_zero_slope_does_not_crash(self):
        ls, length = main.compute_ls_factor(slope_degrees=0, tree_canopy_pct=50)
        assert ls >= 0


class TestComputeCFactor:
    def test_missing_canopy_uses_neutral_default(self):
        c = main.compute_c_factor(tree_canopy_pct=None, ndvi_change_5y=None)
        assert 0 < c < 1

    def test_full_canopy_near_zero(self):
        c = main.compute_c_factor(tree_canopy_pct=100, ndvi_change_5y=None)
        assert c < 0.1

    def test_bare_ground_near_one(self):
        c = main.compute_c_factor(tree_canopy_pct=0, ndvi_change_5y=None)
        assert c == 1.0

    def test_declining_ndvi_pushes_toward_bare_soil(self):
        c_stable = main.compute_c_factor(tree_canopy_pct=50, ndvi_change_5y=0)
        c_declining = main.compute_c_factor(tree_canopy_pct=50, ndvi_change_5y=-0.5)
        assert c_declining > c_stable

    def test_c_factor_is_capped_at_one(self):
        c = main.compute_c_factor(tree_canopy_pct=0, ndvi_change_5y=-5.0)
        assert c <= 1.0


class TestComputeRFactor:
    def test_zero_precip_gives_zero_r(self):
        assert main.compute_r_factor_us(0) == 0

    def test_typical_precip_is_positive(self):
        assert main.compute_r_factor_us(1000) > 0

    def test_low_and_high_precip_branches_both_run(self):
        # formula switches branches at P <= 850
        assert main.compute_r_factor_us(849) > 0
        assert main.compute_r_factor_us(851) > 0


class TestBufferAcres:
    def test_positive_at_normal_latitude(self):
        assert main._buffer_acres(40.0, 0.001) > 0

    def test_does_not_crash_near_pole(self):
        # cos(90deg) isn't exactly 0 in floating point, but should still
        # produce a sane (near-zero) non-negative area, not a crash.
        result = main._buffer_acres(89.999, 0.001)
        assert result >= 0


# ---------------------------------------------------------------------------
# SSURGO live fallback
# ---------------------------------------------------------------------------

SSURGO_ROW = ["B", "0.32", "4.5", "Test Loam", "Occasional", "Prime farmland if drained", "Fragipan", "45"]


class TestSsurgoFallback:
    @patch("main.requests.post")
    def test_parses_all_columns_by_position(self, mock_post):
        mock_post.return_value = mock_response({"Table": [SSURGO_ROW]})
        result = main.fetch_ssurgo_live_fallback(40.0, -100.0)
        assert result == {
            "hydgrp": "B",
            "kwfact": 0.32,
            "shrink_swell": "Moderate",
            "muname": "Test Loam",
            "pondfreq": "Occasional",
            "farmland_class": "Prime farmland if drained",
            "restrictive_kind": "Fragipan",
            "restrictive_depth_cm": 45.0,
        }

    @patch("main.requests.post")
    def test_empty_table_returns_empty_dict(self, mock_post):
        mock_post.return_value = mock_response({"Table": []})
        assert main.fetch_ssurgo_live_fallback(40.0, -100.0) == {}

    @patch("main.requests.post")
    def test_missing_table_key_returns_empty_dict(self, mock_post):
        mock_post.return_value = mock_response({})
        assert main.fetch_ssurgo_live_fallback(40.0, -100.0) == {}

    @patch("main.requests.post")
    def test_non_numeric_kwfact_becomes_none(self, mock_post):
        row = list(SSURGO_ROW)
        row[1] = ""
        mock_post.return_value = mock_response({"Table": [row]})
        result = main.fetch_ssurgo_live_fallback(40.0, -100.0)
        assert result["kwfact"] is None

    @patch("main.requests.post")
    def test_short_row_does_not_crash(self, mock_post):
        # simulates a response shape that only returns the original 5 columns
        mock_post.return_value = mock_response({"Table": [SSURGO_ROW[:5]]})
        result = main.fetch_ssurgo_live_fallback(40.0, -100.0)
        assert result["farmland_class"] is None
        assert result["restrictive_kind"] is None
        assert result["restrictive_depth_cm"] is None

    def test_only_fills_missing_fields_never_overwrites(self):
        fields = {
            "soil_erodibility_k_factor": {"status": "ok", "value": 0.99, "source": "MIREYE"},
        }
        with patch.object(main, "fetch_ssurgo_live_fallback", return_value={
            "hydgrp": "D", "kwfact": 0.1, "shrink_swell": "Low", "muname": "X",
            "pondfreq": "None", "farmland_class": "Not prime farmland",
            "restrictive_kind": None, "restrictive_depth_cm": None,
        }):
            result = main.patch_nulls_with_live_ssurgo(40.0, -100.0, fields)
        assert result["soil_erodibility_k_factor"]["value"] == 0.99
        assert result["soil_erodibility_k_factor"]["source"] == "MIREYE"
        assert result["soil_hydrologic_group"]["value"] == "D"
        assert result["soil_hydrologic_group"]["source"] == "USDA_SDA_LIVE"

    def test_fetch_error_is_logged_not_raised(self):
        fields = {}
        with patch.object(main, "fetch_ssurgo_live_fallback", side_effect=RuntimeError("network down")):
            result = main.patch_nulls_with_live_ssurgo(40.0, -100.0, fields)
        assert result["_ssurgo_live_fallback_error"]["status"] == "failed"

    def test_no_fetch_when_nothing_missing(self):
        fields = {
            "soil_erodibility_k_factor": {"status": "ok", "value": 1},
            "soil_hydrologic_group": {"status": "ok", "value": "A"},
            "soil_shrink_swell_class": {"status": "ok", "value": "Low"},
            "soil_ponding_frequency_class": {"status": "ok", "value": "None"},
            "prime_farmland_classification": {"status": "ok", "value": "X"},
            "soil_restrictive_layer_kind": {"status": "ok", "value": "X"},
            "soil_restrictive_layer_depth_cm": {"status": "ok", "value": 10},
        }
        with patch.object(main, "fetch_ssurgo_live_fallback") as mock_fetch:
            main.patch_nulls_with_live_ssurgo(40.0, -100.0, fields)
            mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# DEM slope fallback
# ---------------------------------------------------------------------------

class TestDemSlopeFallback:
    @patch("main.fetch_elevation_m")
    def test_flat_ground_gives_near_zero_slope(self, mock_elev):
        mock_elev.return_value = 100.0  # same elevation everywhere -> flat
        slope = main.fetch_slope_from_dem(40.0, -100.0)
        assert slope == pytest.approx(0.0, abs=1e-6)

    @patch("main.fetch_elevation_m")
    def test_uniform_gradient_gives_expected_slope(self, mock_elev):
        # elevation rises 30m to the east, flat north-south
        def side_effect(lat, lng):
            return 100.0 + (lng - (-100.0)) * 1000  # crude gradient proxy
        mock_elev.side_effect = side_effect
        slope = main.fetch_slope_from_dem(40.0, -100.0)
        assert slope > 0

    def test_near_pole_raises_instead_of_dividing_by_zero(self):
        with pytest.raises(RuntimeError):
            main.fetch_slope_from_dem(89.95, -100.0)

    def test_skips_when_slope_already_present(self):
        fields = {"slope_degrees": {"status": "ok", "value": 5.0}}
        with patch.object(main, "fetch_slope_from_dem") as mock_fetch:
            result = main.patch_null_slope_with_dem(40.0, -100.0, fields)
            mock_fetch.assert_not_called()
        assert result["slope_degrees"]["value"] == 5.0

    def test_fills_when_missing(self):
        fields = {}
        with patch.object(main, "fetch_slope_from_dem", return_value=7.5):
            result = main.patch_null_slope_with_dem(40.0, -100.0, fields)
        assert result["slope_degrees"]["value"] == 7.5
        assert result["slope_degrees"]["source"] == "USGS_3DEP_EPQS"

    def test_error_logged_not_raised(self):
        fields = {}
        with patch.object(main, "fetch_slope_from_dem", side_effect=RuntimeError("no network")):
            result = main.patch_null_slope_with_dem(40.0, -100.0, fields)
        assert result["_dem_slope_fallback_error"]["status"] == "failed"


# ---------------------------------------------------------------------------
# FEMA NFHL fallback
# ---------------------------------------------------------------------------

class TestFemaFallback:
    @patch("main.requests.get")
    def test_parses_zone_and_bfe(self, mock_get):
        mock_get.return_value = mock_response({
            "features": [{"attributes": {
                "FLD_ZONE": "AE", "ZONE_SUBTY": "1 PCT ANNUAL CHANCE FLOOD HAZARD",
                "STATIC_BFE": 812.3, "SFHA_TF": "T",
            }}]
        })
        result = main.fetch_fema_flood_zone(40.0, -100.0)
        assert result == {"fld_zone": "AE", "coastal": False, "bfe": 812.3}

    @patch("main.requests.get")
    def test_v_zone_flagged_coastal(self, mock_get):
        mock_get.return_value = mock_response({
            "features": [{"attributes": {"FLD_ZONE": "VE", "ZONE_SUBTY": None, "STATIC_BFE": None}}]
        })
        result = main.fetch_fema_flood_zone(40.0, -100.0)
        assert result["coastal"] is True

    @patch("main.requests.get")
    def test_bfe_sentinel_converted_to_none(self, mock_get):
        mock_get.return_value = mock_response({
            "features": [{"attributes": {"FLD_ZONE": "X", "ZONE_SUBTY": "", "STATIC_BFE": -9999}}]
        })
        result = main.fetch_fema_flood_zone(40.0, -100.0)
        assert result["bfe"] is None

    @patch("main.requests.get")
    def test_no_features_returns_empty(self, mock_get):
        mock_get.return_value = mock_response({"features": []})
        assert main.fetch_fema_flood_zone(40.0, -100.0) == {}

    @patch("main.requests.get")
    def test_arcgis_error_payload_raises(self, mock_get):
        mock_get.return_value = mock_response({"error": {"code": 400, "message": "bad geometry"}})
        with pytest.raises(RuntimeError):
            main.fetch_fema_flood_zone(40.0, -100.0)

    def test_derives_within_floodplain_from_zone(self):
        fields = {}
        with patch.object(main, "fetch_fema_flood_zone", return_value={"fld_zone": "A", "coastal": False, "bfe": None}):
            result = main.patch_nulls_with_fema_nfhl(40.0, -100.0, fields)
        assert result["within_floodplain_polygon"]["value"] is True

    def test_zone_x_is_not_floodplain(self):
        fields = {}
        with patch.object(main, "fetch_fema_flood_zone", return_value={"fld_zone": "X", "coastal": False, "bfe": None}):
            result = main.patch_nulls_with_fema_nfhl(40.0, -100.0, fields)
        assert result["within_floodplain_polygon"]["value"] is False

    def test_never_overwrites_real_mireye_value(self):
        fields = {"fema_flood_zone": {"status": "ok", "value": "X", "source": "MIREYE"}}
        with patch.object(main, "fetch_fema_flood_zone", return_value={"fld_zone": "AE", "coastal": False, "bfe": 800}):
            result = main.patch_nulls_with_fema_nfhl(40.0, -100.0, fields)
        assert result["fema_flood_zone"]["value"] == "X"
        assert result["fema_flood_zone"]["source"] == "MIREYE"

    def test_no_zone_in_response_fills_nothing(self):
        fields = {}
        with patch.object(main, "fetch_fema_flood_zone", return_value={}):
            result = main.patch_nulls_with_fema_nfhl(40.0, -100.0, fields)
        assert "fema_flood_zone" not in result
        assert "within_floodplain_polygon" not in result


# ---------------------------------------------------------------------------
# NWI wetlands fallback
# ---------------------------------------------------------------------------

class TestNwiFallback:
    @patch("main.requests.get")
    def test_parses_acres(self, mock_get):
        mock_get.return_value = mock_response({
            "features": [
                {"attributes": {"WETLAND_TYPE": "Freshwater Emergent Wetland", "ACRES": "1.5"}},
                {"attributes": {"WETLAND_TYPE": "Freshwater Pond", "ACRES": "0.5"}},
            ]
        })
        result = main.fetch_nwi_wetlands(40.0, -100.0)
        assert len(result) == 2
        assert result[0]["acres"] == 1.5

    @patch("main.requests.get")
    def test_arcgis_error_raises(self, mock_get):
        mock_get.return_value = mock_response({"error": {"message": "bad request"}})
        with pytest.raises(RuntimeError):
            main.fetch_nwi_wetlands(40.0, -100.0)

    @patch("main.requests.get")
    def test_non_numeric_acres_becomes_none(self, mock_get):
        mock_get.return_value = mock_response({
            "features": [{"attributes": {"WETLAND_TYPE": "X", "ACRES": "n/a"}}]
        })
        result = main.fetch_nwi_wetlands(40.0, -100.0)
        assert result[0]["acres"] is None

    def test_no_wetlands_found_fills_zero_not_error(self):
        fields = {}
        with patch.object(main, "fetch_nwi_wetlands", return_value=[]):
            result = main.patch_nulls_with_nwi_wetlands(40.0, -100.0, fields)
        assert result["wetland_fraction_of_parcel"]["value"] == 0.0
        assert result["wetland_acres_on_parcel"]["value"] == 0.0

    def test_fraction_capped_at_one(self):
        fields = {}
        with patch.object(main, "fetch_nwi_wetlands", return_value=[{"wetland_type": "X", "acres": 999999}]):
            result = main.patch_nulls_with_nwi_wetlands(40.0, -100.0, fields)
        assert result["wetland_fraction_of_parcel"]["value"] <= 1.0

    def test_confidence_is_low(self):
        fields = {}
        with patch.object(main, "fetch_nwi_wetlands", return_value=[]):
            result = main.patch_nulls_with_nwi_wetlands(40.0, -100.0, fields)
        assert result["wetland_fraction_of_parcel"]["confidence"] == "low"

    def test_error_logged_not_raised(self):
        fields = {}
        with patch.object(main, "fetch_nwi_wetlands", side_effect=RuntimeError("down")):
            result = main.patch_nulls_with_nwi_wetlands(40.0, -100.0, fields)
        assert result["_nwi_wetlands_fallback_error"]["status"] == "failed"


# ---------------------------------------------------------------------------
# NLCD canopy fallback
# ---------------------------------------------------------------------------

class TestNlcdFallback:
    @patch("main.requests.get")
    def test_parses_gray_index(self, mock_get):
        mock_get.return_value = mock_response({"features": [{"properties": {"GRAY_INDEX": 42}}]})
        assert main.fetch_nlcd_tree_canopy_pct(40.0, -100.0) == 42.0

    @patch("main.requests.get")
    def test_nodata_sentinel_raises(self, mock_get):
        mock_get.return_value = mock_response({"features": [{"properties": {"GRAY_INDEX": 255}}]})
        with pytest.raises(RuntimeError):
            main.fetch_nlcd_tree_canopy_pct(40.0, -100.0)

    @patch("main.requests.get")
    def test_no_features_raises(self, mock_get):
        mock_get.return_value = mock_response({"features": []})
        with pytest.raises(RuntimeError):
            main.fetch_nlcd_tree_canopy_pct(40.0, -100.0)

    def test_skips_when_present(self):
        fields = {"tree_canopy_pct": {"status": "ok", "value": 55}}
        with patch.object(main, "fetch_nlcd_tree_canopy_pct") as mock_fetch:
            main.patch_null_canopy_with_nlcd(40.0, -100.0, fields)
            mock_fetch.assert_not_called()

    def test_error_logged_not_raised(self):
        fields = {}
        with patch.object(main, "fetch_nlcd_tree_canopy_pct", side_effect=RuntimeError("layer not found")):
            result = main.patch_null_canopy_with_nlcd(40.0, -100.0, fields)
        assert result["_nlcd_canopy_fallback_error"]["status"] == "failed"


# ---------------------------------------------------------------------------
# score_erosion_risk
# ---------------------------------------------------------------------------

class TestScoreErosionRisk:
    def test_no_data_at_all_gives_low_risk_no_rusle(self):
        risk = main.score_erosion_risk({}, [], 40.0, -100.0)
        assert risk["level"] == "low"
        assert risk["rusle_lite"] is None

    def test_rusle_requires_both_slope_and_k_factor(self):
        fields = {"slope_degrees": {"status": "ok", "value": 10}}
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        assert risk["rusle_lite"] is None

    @patch("main.fetch_mean_annual_precip_mm", side_effect=RuntimeError("no network"))
    def test_rusle_computed_relative_index_when_r_unavailable(self, _mock):
        fields = {
            "slope_degrees": {"status": "ok", "value": 15},
            "soil_erodibility_k_factor": {"status": "ok", "value": 0.3},
        }
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        assert risk["rusle_lite"] is not None
        assert "r_factor" not in risk["rusle_lite"]
        assert "r_factor_error" in risk["rusle_lite"]

    @patch("main.fetch_mean_annual_precip_mm", return_value=900.0)
    def test_rusle_computed_absolute_when_r_available(self, _mock):
        fields = {
            "slope_degrees": {"status": "ok", "value": 15},
            "soil_erodibility_k_factor": {"status": "ok", "value": 0.3},
        }
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        assert "annual_soil_loss_tons_per_acre" in risk["rusle_lite"]

    def test_high_landslide_index_raises_score(self):
        low = main.score_erosion_risk({"landslide_susceptibility_index": {"status": "ok", "value": 10}}, [], 40.0, -100.0)
        high = main.score_erosion_risk({"landslide_susceptibility_index": {"status": "ok", "value": 90}}, [], 40.0, -100.0)
        assert high["score"] > low["score"]

    def test_fema_zone_preferred_over_boolean_floodplain(self):
        fields = {
            "fema_flood_zone": {"status": "ok", "value": "AE"},
            "within_floodplain_polygon": {"status": "ok", "value": True},
        }
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        flood_factors = [f for f in risk["factors"] if "flood" in f["label"].lower() or "floodplain" in f["label"].lower()]
        assert len(flood_factors) == 1
        assert "Zone AE" in flood_factors[0]["detail"]

    def test_coastal_hazard_forces_high_severity(self):
        fields = {"fema_flood_zone": {"status": "ok", "value": "VE"}, "coastal_high_hazard": {"status": "ok", "value": True}}
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        flood_factor = next(f for f in risk["factors"] if "flood" in f["label"].lower())
        assert flood_factor["severity"] == "high"

    def test_wetland_zero_fraction_adds_no_factor(self):
        fields = {"wetland_fraction_of_parcel": {"status": "ok", "value": 0.0}}
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        assert not any("wetland" in f["label"].lower() for f in risk["factors"])

    def test_wetland_positive_fraction_adds_factor(self):
        fields = {"wetland_fraction_of_parcel": {"status": "ok", "value": 0.3}}
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        assert any("wetland" in f["label"].lower() for f in risk["factors"])

    def test_hydro_group_d_is_high_severity(self):
        fields = {"soil_hydrologic_group": {"status": "ok", "value": "D"}}
        risk = main.score_erosion_risk(fields, [], 40.0, -100.0)
        factor = next(f for f in risk["factors"] if "hydrologic" in f["label"].lower())
        assert factor["severity"] == "high"

    def test_recent_deforestation_detected(self):
        loss_years = [{"umd_tree_cover_loss__year": 2020, "area_ha": 5.0}]
        risk = main.score_erosion_risk({}, loss_years, 40.0, -100.0)
        assert any("deforestation" in f["label"].lower() and "high" == f["severity"] for f in risk["factors"])

    def test_old_deforestation_before_2018_not_counted_as_recent(self):
        loss_years = [{"umd_tree_cover_loss__year": 2010, "area_ha": 5.0}]
        risk = main.score_erosion_risk({}, loss_years, 40.0, -100.0)
        assert not any("high" == f["severity"] and "deforestation" in f["label"].lower() for f in risk["factors"])

    def test_score_thresholds_produce_expected_levels(self):
        # 0 factors -> low
        assert main.score_erosion_risk({}, [], 40.0, -100.0)["level"] == "low"
        # landslide (2) + hydro group D (1) = 3 -> moderate
        fields = {
            "landslide_susceptibility_index": {"status": "ok", "value": 90},
            "soil_hydrologic_group": {"status": "ok", "value": "D"},
        }
        assert main.score_erosion_risk(fields, [], 40.0, -100.0)["level"] == "moderate"

    def test_missing_loss_years_list_handled(self):
        # regression: score_erosion_risk must not crash if loss_years is
        # falsy/empty rather than a populated list
        risk = main.score_erosion_risk({}, [], 40.0, -100.0)
        assert risk is not None


# ---------------------------------------------------------------------------
# get_mireye_token
# ---------------------------------------------------------------------------

class TestMireyeToken:
    def test_env_var_used_first(self, monkeypatch):
        monkeypatch.setenv("MIREYE_BEARER_TOKEN", "env-token")
        assert main.get_mireye_token() == "env-token"

    def test_raises_when_nothing_available(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MIREYE_BEARER_TOKEN", raising=False)
        monkeypatch.setattr(main, "MIREYE_CREDENTIALS_FILE", tmp_path / "nope.json")
        with pytest.raises(RuntimeError):
            main.get_mireye_token()

    def test_reads_credentials_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MIREYE_BEARER_TOKEN", raising=False)
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text('{"access_token": "file-token"}')
        monkeypatch.setattr(main, "MIREYE_CREDENTIALS_FILE", creds_file)
        assert main.get_mireye_token() == "file-token"


# ---------------------------------------------------------------------------
# Full /api/risk endpoint (integration, everything mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(main.app)


class TestApiRiskEndpoint:
    def test_mireye_failure_returns_502(self, client):
        with patch.object(main, "fetch_mireye_hazard_data", side_effect=RuntimeError("token missing")):
            resp = client.get("/api/risk", params={"lat": 40.0, "lng": -100.0})
        assert resp.status_code == 502

    def test_mireye_fields_explicitly_null_does_not_crash(self, client):
        # regression test: Mireye returning {"fields": null} (valid JSON,
        # legitimate "no data" shape) must not 500 the whole endpoint.
        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": None, "partial_failures": None}), \
             patch.object(main, "fetch_mireye_ask", return_value={"answer": None}), \
             patch.object(main, "fetch_gfw_tree_cover_loss", return_value=[]), \
             patch.object(main, "patch_nulls_with_live_ssurgo", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_slope_with_dem", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_fema_nfhl", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_nwi_wetlands", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_canopy_with_nlcd", side_effect=lambda lat, lng, f: f):
            resp = client.get("/api/risk", params={"lat": 40.0, "lng": -100.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["mireye_fields"] == {}
        assert body["mireye_partial_failures"] == []

    def test_gfw_data_explicitly_null_does_not_crash(self, client):
        # regression test: GFW returning {"data": null} must not crash
        # score_erosion_risk's iteration over loss_years.
        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": {}, "partial_failures": []}), \
             patch.object(main, "fetch_mireye_ask", return_value={"answer": "summary"}), \
             patch("main.requests.get") as mock_get, \
             patch.object(main, "patch_nulls_with_live_ssurgo", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_slope_with_dem", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_fema_nfhl", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_nwi_wetlands", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_canopy_with_nlcd", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "get_gfw_api_key", return_value="fake-key"), \
             patch.object(main, "get_gfw_latest_version", return_value="v1"):
            mock_get.return_value = mock_response({"data": None}, status_ok=True)
            # fetch_gfw_tree_cover_loss uses requests.post, not requests.get;
            # patch that directly instead for this test.
            with patch("main.requests.post") as mock_post:
                mock_post.return_value = mock_response({"data": None})
                resp = client.get("/api/risk", params={"lat": 40.0, "lng": -100.0})
        assert resp.status_code == 200
        assert resp.json()["tree_cover_loss_by_year"] == []

    def test_full_success_path(self, client):
        mireye_payload = {
            "fields": {
                "slope_degrees": {"status": "ok", "value": 12, "unit": "degrees", "source": "MIREYE"},
                "soil_erodibility_k_factor": {"status": "ok", "value": 0.3, "source": "MIREYE"},
                "tree_canopy_pct": {"status": "ok", "value": 40, "unit": "percent", "source": "MIREYE"},
            },
            "partial_failures": [],
        }
        with patch.object(main, "fetch_mireye_hazard_data", return_value=mireye_payload), \
             patch.object(main, "fetch_mireye_ask", return_value={"answer": "All quiet here."}), \
             patch.object(main, "fetch_gfw_tree_cover_loss", return_value=[{"umd_tree_cover_loss__year": 2019, "area_ha": 0.0}]), \
             patch.object(main, "patch_nulls_with_live_ssurgo", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_slope_with_dem", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_fema_nfhl", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_nwi_wetlands", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_canopy_with_nlcd", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "fetch_mean_annual_precip_mm", return_value=900.0):
            resp = client.get("/api/risk", params={"lat": 40.0, "lng": -100.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["mireye_ask_summary"] == "All quiet here."
        assert body["risk"]["rusle_lite"] is not None
        assert body["location"] == {"lat": 40.0, "lng": -100.0}

    def test_ask_failure_is_non_fatal(self, client):
        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": {}, "partial_failures": []}), \
             patch.object(main, "fetch_mireye_ask", side_effect=RuntimeError("ask endpoint down")), \
             patch.object(main, "fetch_gfw_tree_cover_loss", return_value=[]), \
             patch.object(main, "patch_nulls_with_live_ssurgo", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_slope_with_dem", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_fema_nfhl", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_nwi_wetlands", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_canopy_with_nlcd", side_effect=lambda lat, lng, f: f):
            resp = client.get("/api/risk", params={"lat": 40.0, "lng": -100.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["mireye_ask_summary"] is None
        assert "ask endpoint down" in body["mireye_ask_error"]

    def test_gfw_failure_is_non_fatal(self, client):
        with patch.object(main, "fetch_mireye_hazard_data", return_value={"fields": {}, "partial_failures": []}), \
             patch.object(main, "fetch_mireye_ask", return_value={"answer": None}), \
             patch.object(main, "fetch_gfw_tree_cover_loss", side_effect=RuntimeError("GFW_API_KEY not set")), \
             patch.object(main, "patch_nulls_with_live_ssurgo", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_slope_with_dem", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_fema_nfhl", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_nulls_with_nwi_wetlands", side_effect=lambda lat, lng, f: f), \
             patch.object(main, "patch_null_canopy_with_nlcd", side_effect=lambda lat, lng, f: f):
            resp = client.get("/api/risk", params={"lat": 40.0, "lng": -100.0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["gfw_error"] is not None
        assert body["tree_cover_loss_by_year"] == []

    def test_missing_query_params_returns_422(self, client):
        resp = client.get("/api/risk")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/rankings ("Most At-Risk Areas" browse view)
# ---------------------------------------------------------------------------

SAMPLE_RANKINGS_CACHE = {
    "computed_at": "2026-07-14T00:00:00+00:00",
    "county_count": 3,
    "failed_count": 1,
    "results": [
        {"county": "Harris County (Houston)", "state": "Texas", "state_abbr": "TX", "region": "South", "lat": 29.76, "lng": -95.37, "score": 7, "level": "high"},
        {"county": "Los Angeles County", "state": "California", "state_abbr": "CA", "region": "West", "lat": 34.05, "lng": -118.24, "score": 4, "level": "moderate"},
        {"county": "Cook County (Chicago)", "state": "Illinois", "state_abbr": "IL", "region": "Midwest", "lat": 41.88, "lng": -87.63, "error": "Mireye /v1/fetch failed: timeout"},
    ],
}


class TestRankingsEndpoint:
    def test_no_cache_file_returns_graceful_empty_state(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", tmp_path / "does_not_exist.json")
        resp = client.get("/api/rankings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_available"] is False
        assert body["results"] == []
        assert "precompute_rankings.py" in body["message"]

    def test_full_cache_sorted_by_score_desc(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_available"] is True
        assert [r["county"] for r in body["results"]] == ["Harris County (Houston)", "Los Angeles County"]
        assert body["results"][0]["score"] >= body["results"][1]["score"]

    def test_failed_entries_excluded_from_ranked_results(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings")
        body = resp.json()
        assert len(body["failed"]) == 1
        assert body["failed"][0]["county"] == "Cook County (Chicago)"
        assert all("score" in r for r in body["results"])

    def test_filter_by_state_abbr(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"state": "TX"})
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["state_abbr"] == "TX"

    def test_filter_by_state_full_name_case_insensitive(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"state": "california"})
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["state"] == "California"

    def test_filter_by_region(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"region": "West"})
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["region"] == "West"

    def test_filter_matching_nothing_returns_empty_not_error(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"state": "Wyoming"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data_available"] is True
        assert body["results"] == []

    def test_limit_truncates_ranked_results(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"limit": 1})
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["results"][0]["county"] == "Harris County (Houston)"  # highest score

    def test_limit_does_not_truncate_failed_list(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"limit": 1})
        body = resp.json()
        assert len(body["failed"]) == 1

    def test_limit_larger_than_result_count_is_harmless(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"limit": 999})
        body = resp.json()
        assert len(body["results"]) == 2

    def test_limit_zero_or_negative_rejected(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps(SAMPLE_RANKINGS_CACHE))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings", params={"limit": 0})
        assert resp.status_code == 422

    def test_malformed_cache_file_returns_500_not_crash_unhandled(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text("{not valid json")
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings")
        assert resp.status_code == 500

    def test_cache_with_null_results_key_does_not_crash(self, client, monkeypatch, tmp_path):
        cache_file = tmp_path / "rankings_cache.json"
        cache_file.write_text(main.json.dumps({"computed_at": "x", "results": None}))
        monkeypatch.setattr(main, "RANKINGS_CACHE_FILE", cache_file)

        resp = client.get("/api/rankings")
        assert resp.status_code == 200
        assert resp.json()["results"] == []
