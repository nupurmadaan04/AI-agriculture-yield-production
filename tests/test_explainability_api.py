"""
Integration tests for Day 13 Explainable AI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_global_importance():
    resp = client.get("/api/explainability/global")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "exogenous_rf_forecaster"
    assert len(data["features"]) == 10


def test_api_explain_prediction():
    payload = {
        "state": "Punjab",
        "district": "Ludhiana",
        "area_1000_ha": 310.0,
        "year": 2017
    }
    resp = client.post("/api/explainability/prediction", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_kg_ha" in data
    assert len(data["feature_contributions"]) == 10


def test_api_sensitivity():
    payload = {
        "state": "Punjab",
        "target_features": ["RICE_YIELD_LAG1", "RICE AREA (1000 ha)"]
    }
    resp = client.post("/api/explainability/sensitivity", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sensitivity_curves" in data


def test_api_explain_alert():
    resp = client.get("/api/explainability/alert/ALR-000183")
    assert resp.status_code == 200
    data = resp.json()
    assert data["alert_id"] == "ALR-000183"


def test_api_explain_scenario():
    resp = client.post(
        "/api/explainability/scenario/SCEN-001?state=Punjab&baseline_yield=3950&simulated_yield=4150",
        json={"RICE_AREA_SHARE": 0.55}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario_id"] == "SCEN-001"
    assert data["simulated_delta_kg_ha"] == 200.0


def test_api_audit():
    resp = client.get("/api/explainability/audit/EXP-000183")
    assert resp.status_code == 200
    data = resp.json()
    assert "explanation_id" in data


def test_api_methodology():
    resp = client.get("/api/explainability/methodology")
    assert resp.status_code == 200
    data = resp.json()
    assert "supported_methods" in data


def test_api_validation():
    resp = client.get("/api/explainability/validation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is True
    assert data["passed_checks"] >= 6
