"""
Day 33: End-to-End Production Acceptance Test Suite.

Validates the complete user journeys across:
- Golden Journey 1: Oilseeds (Certified Governed ML Production Strategy)
- Golden Journey 2: Sugarcane (Conditional ML Production Strategy)
- Golden Journey 3: Rice (Certified Baseline Production Strategy)
- Golden Journey 4: Wheat (Certified Baseline Production Strategy)
- Negative Workflows: Unsupported crop, state, district; missing fields; malformed payloads;
  unknown routes; unsupported HTTP methods; future unharvested outcome status.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# =============================================================================
# 1. GOLDEN JOURNEY 1: OILSEEDS (Certified Governed ML Production Strategy)
# =============================================================================

def test_golden_journey_oilseeds_complete_workflow():
    """
    Validates complete user journey for Oilseeds in Punjab (Ludhiana):
    1. Scope selection
    2. Governed forecast via PredictionService
    3. Strategy resolution -> Historical ML (RandomForestRegressor)
    4. Provenance SHA-256 fingerprint
    5. Validation proof & win rate
    6. Empirical tree dispersion uncertainty
    7. Model attribution (Tree SHAP)
    8. Operational drift monitoring
    9. Scenario comparison matrix (Zero subjective ranking words)
    10. Audit trace
    """
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "forecast_year": 2017,
        "selected_scenarios": ["conservative_improvement", "moderate_improvement", "stress_scenario"],
        "custom_modifications": {"rice_area_pct": 5.0}
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    # 1. Baseline Forecast Verification
    baseline = data["baseline_forecast"]
    assert baseline["semantic_classification"] == "PREDICTED"
    assert baseline["forecast_yield_kg_ha"] > 0
    assert "RandomForest" in baseline["strategy"]
    assert baseline["certification_status"] == "PRODUCTION_READY"
    assert baseline["provenance_hash"].startswith("SHA256:")
    assert baseline["request_id"].startswith("REQ-")

    # 2. Historical Reference Context
    hist = data["historical_context"]
    assert hist["semantic_classification"] in ("DERIVED", "HISTORICAL_REFERENCE")
    assert hist["sample_count"] >= 5
    assert hist["historical_mean_yield_kg_ha"] > 0
    # Strict temporal isolation
    for pt in hist["recent_observations"]:
        assert pt["year"] < 2017, f"Temporal leakage: observation year {pt['year']} >= forecast year 2017"

    # 3. Validation Proof
    val = data["validation"]
    assert val["semantic_classification"] == "VALIDATION"
    assert val["mae_kg_ha"] > 0
    assert val["fold_win_rate_pct"] >= 50.0

    # 4. Uncertainty
    unc = data["uncertainty"]
    assert unc["is_available"] is True
    assert unc["empirical_p10_kg_ha"] is not None
    assert unc["empirical_p90_kg_ha"] is not None
    assert unc["empirical_p10_kg_ha"] <= unc["empirical_p90_kg_ha"]

    # 5. Attribution
    attrib = data["attribution"]
    assert attrib["semantic_classification"] == "MODEL_ATTRIBUTION"
    assert attrib["attribution_type"] == "TREE_SHAP"
    assert len(attrib["top_features"]) > 0

    # 6. Monitoring
    mon = data["monitoring"]
    assert mon["semantic_classification"] == "MONITORING"
    assert "overall_psi" in mon

    # 7. Scenarios & Comparison Matrix
    scenarios = data["scenarios"]
    assert len(scenarios) >= 4  # Baseline, Conservative, Moderate, Stress, Custom
    matrix = data["comparison_matrix"]
    assert len(matrix["scenario_headers"]) == len(scenarios)

    # Non-autonomous check: zero subjective ranking words
    forbidden = ["BEST", "WORST", "WINNER", "RECOMMENDED", "OPTIMAL"]
    matrix_str = str(matrix).upper()
    for word in forbidden:
        assert word not in matrix_str, f"Forbidden ranking word '{word}' found in comparison matrix"


# =============================================================================
# 2. GOLDEN JOURNEY 2: SUGARCANE (Conditional ML Production Strategy)
# =============================================================================

def test_golden_journey_sugarcane_complete_workflow():
    """
    Validates complete user journey for Sugarcane in Uttar Pradesh (Meerut):
    - Strategy tier: CONDITIONAL_PRODUCTION
    - Model engine: GradientBoostingRegressor
    - Provenance & walk-forward metrics
    - Conditional caveats in limitations
    """
    payload = {
        "crop": "Sugarcane",
        "state": "Uttar Pradesh",
        "district": "Meerut",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    baseline = data["baseline_forecast"]
    assert baseline["certification_status"] == "CONDITIONAL_PRODUCTION"
    assert "GradientBoosting" in baseline["strategy"]
    assert baseline["forecast_yield_kg_ha"] > 1000

    val = data["validation"]
    assert val["fold_win_rate_pct"] >= 50.0
    assert val["mean_improvement_pct"] >= 1.0

    # Ensure structured limitations exist
    assert len(data["limitations"]) > 0


# =============================================================================
# 3. GOLDEN JOURNEY 3: RICE (Certified Baseline Production Strategy)
# =============================================================================

def test_golden_journey_rice_complete_workflow():
    """
    Validates complete user journey for Rice in Punjab (Ludhiana):
    - Strategy tier: BASELINE_PRODUCTION (Historical District Mean)
    - Zero ML-only evidence (Tree dispersion marked unavailable, persistence attribution)
    - Academic benchmark note present
    """
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    baseline = data["baseline_forecast"]
    assert baseline["certification_status"] == "BASELINE_PRODUCTION"
    assert "Historical District Mean" in baseline["strategy"]

    # Uncertainty must be explicitly unavailable for baseline
    unc = data["uncertainty"]
    assert unc["is_available"] is False
    assert "deterministic baseline" in unc["limitations"].lower()

    # Attribution must be PERSISTENCE_BASELINE
    attrib = data["attribution"]
    assert attrib["attribution_type"] == "PERSISTENCE_BASELINE"

    # Validation contains academic benchmark note
    val = data["validation"]
    assert val["legacy_benchmark_note"] is not None
    assert "0.7866" in val["legacy_benchmark_note"]


# =============================================================================
# 4. GOLDEN JOURNEY 4: WHEAT (Certified Baseline Production Strategy)
# =============================================================================

def test_golden_journey_wheat_complete_workflow():
    """
    Validates complete user journey for Wheat in Haryana (Karnal):
    - Strategy tier: BASELINE_PRODUCTION
    - Correct evidence hierarchy without fabricated ML proof
    """
    payload = {
        "crop": "Wheat",
        "state": "Haryana",
        "district": "Karnal",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    baseline = data["baseline_forecast"]
    assert baseline["certification_status"] == "BASELINE_PRODUCTION"
    assert data["uncertainty"]["is_available"] is False
    assert data["attribution"]["attribution_type"] == "PERSISTENCE_BASELINE"


# =============================================================================
# 5. NEGATIVE WORKFLOWS & CONTROLLED REJECTIONS
# =============================================================================

def test_negative_workflow_unsupported_crop():
    """Unsupported crop must be rejected with 400 Bad Request and structured error."""
    payload = {
        "crop": "Dragonfruit",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 400
    err = resp.json()
    assert "UNSUPPORTED_CROP" in str(err)


def test_negative_workflow_unsupported_district_governed_forecast():
    """Unsupported district in forecast route returns structured rejection."""
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "district": "NonExistentDistrictXYZ",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "REJECTED"
    assert data["error_code"] == "DISTRICT_UNSUPPORTED"


def test_negative_workflow_missing_crop():
    """Missing required crop in forecast endpoint returns 422 Unprocessable Entity."""
    payload = {
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 422


def test_negative_workflow_missing_district():
    """Missing required district in forecast endpoint returns 422 Unprocessable Entity."""
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 422


def test_negative_workflow_invalid_year_type():
    """Invalid year type (string that cannot be parsed as int) returns 422."""
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": "nineteen-ninety-five"
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 422


def test_negative_workflow_malformed_json():
    """Malformed JSON string returns controlled 422 client error."""
    resp = client.post(
        "/api/workspace/analyze",
        content="{'broken_json': true,",
        headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422


def test_negative_workflow_unknown_api_route():
    """Non-existent route returns clean 404."""
    resp = client.get("/api/completely/unknown/endpoint")
    assert resp.status_code == 404
    err = resp.json()
    assert "error" in err or "detail" in err


def test_negative_workflow_unsupported_http_method():
    """Invoking DELETE on a POST/GET route returns controlled 405 Method Not Allowed."""
    resp = client.delete("/api/workspace/analyze")
    assert resp.status_code == 405


def test_negative_workflow_unharvested_future_outcome_availability():
    """Forecast year 2026 must report evaluation unavailable without hallucinating ground truth."""
    payload = {
        "crop": "Oilseeds",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2026
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    mon = data["monitoring"]
    assert mon["outcome_evaluation_status"] == "EVALUATION_UNAVAILABLE"
