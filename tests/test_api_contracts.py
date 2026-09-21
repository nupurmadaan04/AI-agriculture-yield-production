"""
Day 33: Production API Contract & Schema Compliance Test Suite.

Validates that critical endpoints return the expected HTTP statuses,
required fields, and strict data types according to their canonical schemas:
- Health & Readiness Probes
- Governed Forecast Serving Endpoints
- Decision Workspace Endpoints
- Decision Intelligence Endpoints
- Forecast Monitoring Endpoints
- Observability Telemetry Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# =============================================================================
# 1. SYSTEM HEALTH & READINESS CONTRACTS
# =============================================================================

def test_contract_health_probe():
    """Verify /health contract: status: ok, service, version."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["service"], str)
    assert isinstance(data["version"], str)


def test_contract_ready_probe():
    """Verify /ready contract: status, components dictionary."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "components" in data
    assert isinstance(data["components"], dict)
    assert "dataset" in data["components"]


# =============================================================================
# 2. GOVERNED FORECAST API CONTRACTS
# =============================================================================

def test_contract_forecast_predict():
    """Verify POST /api/forecast/predict contract."""
    payload = {
        "crop": "Oilseeds",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    required_fields = [
        "status", "request_id", "prediction", "unit", "crop",
        "state", "district", "forecast_year", "strategy",
        "certification_status", "fallback_used", "validation_scope"
    ]
    for field in required_fields:
        assert field in data, f"Missing required field '{field}' in forecast response"

    assert data["status"] == "SUCCESS"
    assert isinstance(data["prediction"], float)
    assert data["unit"] == "kg/ha"
    assert isinstance(data["fallback_used"], bool)


def test_contract_forecast_strategies():
    """Verify GET /api/forecast/strategies contract."""
    resp = client.get("/api/forecast/strategies")
    assert resp.status_code == 200
    data = resp.json()
    assert "strategies" in data
    assert "total_strategies" in data
    assert isinstance(data["strategies"], list)
    assert data["total_strategies"] == len(data["strategies"])
    assert data["total_strategies"] >= 14


def test_contract_forecast_coverage():
    """Verify GET /api/forecast/coverage contract."""
    resp = client.get("/api/forecast/coverage")
    assert resp.status_code == 200
    data = resp.json()
    assert "coverage" in data
    assert "total_records" in data
    assert isinstance(data["coverage"], list)
    assert data["total_records"] > 0


def test_contract_forecast_provenance():
    """Verify GET /api/forecast/provenance/{request_id} contract."""
    # First make a forecast with valid scope
    pred_resp = client.post("/api/forecast/predict", json={
        "crop": "Oilseeds", "state": "Madhya Pradesh", "district": "Indore", "forecast_year": 2017
    })
    req_id = pred_resp.json()["request_id"]

    prov_resp = client.get(f"/api/forecast/provenance/{req_id}")
    assert prov_resp.status_code == 200
    data = prov_resp.json()
    assert data["request_id"] == req_id
    assert "provenance_hash" in data
    assert data["provenance_hash"].startswith("SHA256:")
    assert "model_artifact_hash" in data
    assert "data_source" in data


# =============================================================================
# 3. DECISION WORKSPACE CONTRACTS
# =============================================================================

def test_contract_workspace_analyze():
    """Verify POST /api/workspace/analyze master contract."""
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    top_level_keys = [
        "workspace_id", "crop", "state", "district", "forecast_year",
        "baseline_forecast", "historical_context", "validation",
        "uncertainty", "monitoring", "attribution", "scenarios",
        "comparison_matrix", "limitations", "generated_at"
    ]
    for key in top_level_keys:
        assert key in data, f"Missing key '{key}' in workspace response"

    assert isinstance(data["scenarios"], list)
    assert isinstance(data["comparison_matrix"]["scenario_headers"], list)
    assert isinstance(data["comparison_matrix"]["rows"], list)
    assert isinstance(data["limitations"], list)


def test_contract_workspace_templates():
    """Verify GET /api/workspace/templates contract."""
    resp = client.get("/api/workspace/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert "archetypes" in data
    assert "supported_features" in data
    assert "parameter_bounds" in data
    assert isinstance(data["archetypes"], list)
    assert len(data["archetypes"]) >= 3


# =============================================================================
# 4. DECISION INTELLIGENCE CONTRACTS
# =============================================================================

def test_contract_decision_analyze():
    """Verify POST /api/decision/analyze contract."""
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "year": 2017
    }
    resp = client.post("/api/decision/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "decision_id" in data
    assert "context" in data
    assert "brief" in data
    assert "is_scientifically_validated" in data


def test_contract_decision_brief():
    """Verify GET /api/decision/brief contract."""
    resp = client.get("/api/decision/brief?crop=Rice&state=Punjab&district=Ludhiana&year=2017")
    assert resp.status_code == 200
    data = resp.json()
    assert "decision_id" in data
    assert "executive_summary" in data
    assert "historical_context" in data
    assert "validation_evidence" in data
    assert "uncertainty_evidence" in data
    assert "monitoring_evidence" in data
    assert "limitations" in data
    assert "audit_record" in data


# =============================================================================
# 5. OBSERVABILITY & TELEMETRY CONTRACTS
# =============================================================================

def test_contract_observability_health():
    """Verify GET /api/observability/health contract."""
    resp = client.get("/api/observability/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "api_status" in data
    assert "readiness_status" in data
    assert "backend_status" in data


def test_contract_observability_summary():
    """Verify GET /api/observability/summary contract."""
    resp = client.get("/api/observability/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "system_health" in data
    assert "runtime_metrics" in data
    assert "forecast_operations" in data
