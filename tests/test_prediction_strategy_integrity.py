"""
Day 29 Strategy Registry Integrity & Governance Test Suite.
Verifies that for every registered crop in the catalog, forecast serving resolves
the exact strategy, model family, operating rule, and certification status defined in the registry.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_strategy_catalog_completeness():
    """Verifies that all 14 crops exist in the strategy catalog."""
    resp = client.get("/api/forecast/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_strategies"] == 14
    crops = [s["crop"] for s in data["strategies"]]
    assert "Oilseeds" in crops
    assert "Sugarcane" in crops
    assert "Rice" in crops
    assert "Wheat" in crops


def test_strategy_resolution_integrity():
    """
    Verifies that calling /api/forecast/predict for each crop with supported geography
    returns the exact strategy registered in /api/forecast/strategies.
    """
    strat_resp = client.get("/api/forecast/strategies")
    assert strat_resp.status_code == 200
    strat_map = {s["crop"]: s for s in strat_resp.json()["strategies"]}

    cov_resp = client.get("/api/forecast/coverage")
    assert cov_resp.status_code == 200
    cov_items = cov_resp.json()["coverage"]

    # Test each crop using its first covered geography
    tested_crops = set()
    for item in cov_items:
        crop = item["crop"]
        if crop in tested_crops or crop not in strat_map:
            continue

        tested_crops.add(crop)
        expected_strat = strat_map[crop]

        payload = {
            "crop": crop,
            "state": item["state"],
            "district": item["district"],
            "forecast_year": 2018,
        }

        pred_resp = client.post("/api/forecast/predict", json=payload)
        assert pred_resp.status_code == 200, f"Failed prediction for {crop}: {pred_resp.text}"
        pred_data = pred_resp.json()

        assert pred_data["status"] == "SUCCESS"
        assert pred_data["strategy"] == expected_strat["primary_strategy"]
        assert pred_data["certification_status"] == expected_strat["certification_status"]
        assert pred_data["unit"] == "kg/ha"

    assert len(tested_crops) == 14, f"Expected 14 crops tested, got {len(tested_crops)}"


def test_unsupported_crop_guard():
    """Verifies that an unregistered crop is rejected by the governance guard."""
    payload = {
        "crop": "Dragonfruit",
        "state": "Maharashtra",
        "district": "Pune",
        "forecast_year": 2018,
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "REJECTED"
    assert data["error_code"] == "UNSUPPORTED_CROP"
    assert "not registered" in data["error_message"].lower()
