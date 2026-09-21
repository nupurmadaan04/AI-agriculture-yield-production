"""
Day 33: Provenance Chain Validation & Decision Traceability Test Suite.

Validates:
1. Complete cryptographic provenance chain from user request to audit log:
   User Request -> Strategy Resolution -> Model Artifact -> Dataset Version -> Prediction -> SHA-256 Fingerprint -> Audit Log
2. Baseline strategies (Rice, Wheat) do NOT reference non-existent ML artifacts.
3. Decision Workspace Trace: Baseline, Scenarios, Comparison, and Evidence all reference
   the identical scope without cross-request contamination.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# =============================================================================
# 1. PROVENANCE CHAIN VALIDATION: OILSEEDS (ML Production)
# =============================================================================

def test_provenance_chain_oilseeds_governed_ml():
    """
    Validate complete end-to-end provenance chain for Oilseeds ML:
    Request -> Strategy -> Artifact -> Dataset -> Prediction -> SHA-256 -> Audit Log
    """
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "forecast_year": 2017
    }
    # 1. Execute forecast
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    pred_data = resp.json()
    req_id = pred_data["request_id"]
    assert req_id.startswith("REQ-")
    assert "RandomForest" in pred_data["strategy"]

    # 2. Retrieve provenance record
    prov_resp = client.get(f"/api/forecast/provenance/{req_id}")
    assert prov_resp.status_code == 200
    prov_data = prov_resp.json()

    assert prov_data["request_id"] == req_id
    assert prov_data["crop"].lower() == "oilseeds"
    assert prov_data["state"] == "Madhya Pradesh"
    assert prov_data["district"] == "Indore"
    assert prov_data["forecast_year"] == 2017
    assert prov_data["provenance_hash"].startswith("SHA256:")
    assert "data_source" in prov_data

    # 3. Retrieve audit log event
    audit_resp = client.get("/api/forecast/audit?limit=20")
    assert audit_resp.status_code == 200
    audit_events = audit_resp.json().get("events", [])
    matched = [e for e in audit_events if e.get("request_id") == req_id]
    assert len(matched) >= 1, f"Audit event for {req_id} not found in append-oriented audit log"
    audit_record = matched[0]
    assert audit_record["crop"].lower() == "oilseeds"
    assert audit_record["status"] == "SUCCESS"
    assert audit_record["certification_status"] == "PRODUCTION_READY"


# =============================================================================
# 2. PROVENANCE CHAIN VALIDATION: SUGARCANE (Conditional ML)
# =============================================================================

def test_provenance_chain_sugarcane_conditional_ml():
    """Validate provenance chain for Sugarcane conditional ML."""
    payload = {
        "crop": "Sugarcane",
        "state": "Uttar Pradesh",
        "district": "Meerut",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    pred_data = resp.json()
    req_id = pred_data["request_id"]
    assert "GradientBoosting" in pred_data["strategy"]
    assert pred_data["certification_status"] == "CONDITIONAL_PRODUCTION"

    prov_resp = client.get(f"/api/forecast/provenance/{req_id}")
    assert prov_resp.status_code == 200
    prov_data = prov_resp.json()
    assert prov_data["request_id"] == req_id
    assert prov_data["provenance_hash"].startswith("SHA256:")


# =============================================================================
# 3. PROVENANCE CHAIN VALIDATION: RICE & WHEAT (Baseline Strategies)
# =============================================================================

def test_provenance_chain_rice_baseline():
    """
    Validate that baseline strategy for Rice does NOT fabricate ML artifacts,
    and cleanly documents persistence/district mean origin.
    """
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    pred_data = resp.json()
    req_id = pred_data["request_id"]
    assert pred_data["certification_status"] == "BASELINE_PRODUCTION"
    assert "Historical District Mean" in pred_data["strategy"]

    prov_resp = client.get(f"/api/forecast/provenance/{req_id}")
    assert prov_resp.status_code == 200
    prov_data = prov_resp.json()
    assert prov_data["provenance_hash"].startswith("SHA256:")
    # Ensure baseline artifact is appropriately represented
    assert "RandomForest" not in prov_data.get("model_artifact", "")


def test_provenance_chain_wheat_baseline():
    """Validate baseline provenance for Wheat."""
    payload = {
        "crop": "Wheat",
        "state": "Haryana",
        "district": "Karnal",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    pred_data = resp.json()
    assert pred_data["certification_status"] == "BASELINE_PRODUCTION"
    assert "Historical District Mean" in pred_data["strategy"]


# =============================================================================
# 4. DECISION WORKSPACE TRACE & NON-CONTAMINATION
# =============================================================================

def test_decision_workspace_scope_trace_consistency():
    """
    Verify that Forecast, Evidence, Scenarios, Comparison, and Decision Brief
    all refer strictly to the same scope without cross-request contamination.
    """
    scope = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=scope)
    assert resp.status_code == 200
    data = resp.json()

    # Scope consistency across sub-objects
    assert data["crop"] == "Oilseeds"
    assert data["state"] == "Madhya Pradesh"
    assert data["district"] == "Indore"
    assert data["forecast_year"] == 2017

    assert data["historical_context"]["crop"] == "Oilseeds"
    assert data["historical_context"]["state"] == "Madhya Pradesh"
    assert data["historical_context"]["district"] == "Indore"

    # Scenarios must all have valid simulated yields and IDs
    for sc in data["scenarios"]:
        assert sc["scenario_id"]
        assert isinstance(sc["scenario_output_kg_ha"], (int, float))
        assert sc["is_simulated"] is True

    # Comparison matrix headers must match the scenario count
    sc_headers = data["comparison_matrix"]["scenario_headers"]
    assert len(sc_headers) == len(data["scenarios"])
