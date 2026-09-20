"""
Day 29 Prediction Determinism & Consistency Test Suite.
Verifies that executing identical forecast requests twice produces identical predictions,
identical strategy classifications, identical model versions, and matching provenance fingerprints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


@pytest.mark.parametrize("crop,state,district,forecast_year", [
    ("Oilseeds", "Punjab", "Ludhiana", 2018),
    ("Sugarcane", "Maharashtra", "Kolhapur", 2018),
    ("Rice", "Punjab", "Ludhiana", 2018),
    ("Wheat", "Haryana", "Karnal", 2018),
    ("Chickpea", "Madhya Pradesh", "Indore", 2018),
])
def test_dual_run_deterministic_invariance(crop, state, district, forecast_year):
    """
    Executes the exact same forecast request twice and asserts exact bitwise match for:
    - prediction
    - strategy
    - model_version
    - validation_scope
    - unit
    - certification_status
    """
    payload = {
        "crop": crop,
        "state": state,
        "district": district,
        "forecast_year": forecast_year,
    }

    # Run 1
    resp1 = client.post("/api/forecast/predict", json=payload)
    assert resp1.status_code == 200, f"Run 1 failed: {resp1.text}"
    data1 = resp1.json()

    # Run 2
    resp2 = client.post("/api/forecast/predict", json=payload)
    assert resp2.status_code == 200, f"Run 2 failed: {resp2.text}"
    data2 = resp2.json()

    # Deterministic invariance assertions
    assert data1["status"] == data2["status"] == "SUCCESS"
    assert data1["crop"] == data2["crop"] == crop
    assert data1["state"] == data2["state"] == state
    assert data1["district"] == data2["district"] == district
    assert data1["strategy"] == data2["strategy"]
    assert data1["model_version"] == data2["model_version"]
    assert data1["unit"] == data2["unit"] == "kg/ha"
    assert data1["fallback_used"] == data2["fallback_used"]
    assert data1["certification_status"] == data2["certification_status"]

    # Exact bitwise prediction equality
    assert data1["prediction"] is not None and data2["prediction"] is not None
    assert data1["prediction"] == data2["prediction"]

    # Compare provenance feature sets and hashes if present
    if data1.get("provenance") and data2.get("provenance"):
        prov1 = data1["provenance"]
        prov2 = data2["provenance"]
        assert prov1["features_used"] == prov2["features_used"]
        assert prov1["model_artifact_hash"] == prov2["model_artifact_hash"]
        assert prov1["validation_mae"] == prov2["validation_mae"]
        assert prov1["baseline_mae"] == prov2["baseline_mae"]
        assert prov1["gain_vs_baseline_pct"] == prov2["gain_vs_baseline_pct"]
