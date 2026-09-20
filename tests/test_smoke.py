"""
Day 15 Backend Smoke Test Suite.

Verifies that all core API endpoints, health checks, readiness probes,
system metadata, and primary analytical pipelines respond successfully.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_smoke_health_endpoints(client):
    """Verifies root health and readiness probes."""
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] == "ok"
    assert "version" in data_health

    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    data_ready = res_ready.json()
    assert data_ready["status"] == "ready"
    assert "dataset" in data_ready["components"]
    assert "forecast_model" in data_ready["components"]


def test_smoke_system_info(client):
    """Verifies system info endpoint returns canonical versions and metadata."""
    res = client.get("/api/system/info")
    assert res.status_code == 200
    data = res.json()
    assert data["application_version"] == "1.0.0"
    assert "ICRISAT" in data["dataset_version"]
    assert "exogenous_rf_forecaster" in data["registered_models"]


def test_smoke_agriculture_core(client):
    """Verifies filters and summary endpoints."""
    res_filters = client.get("/api/filters")
    assert res_filters.status_code == 200
    assert len(res_filters.json()["states"]) > 0

    res_summary = client.get("/api/summary")
    assert res_summary.status_code == 200
    assert res_summary.json()["total_records"] > 0


def test_smoke_forecasting(client):
    """Verifies pre-season forecasting endpoint."""
    res = client.get("/api/forecast/state/Punjab?crop=Rice&horizon=3")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "Punjab"
    assert len(data["forecasts"]) == 3


def test_smoke_monitoring_and_geospatial(client):
    """Verifies temporal monitoring overview and geospatial overview."""
    res_mon = client.get("/api/monitoring/overview")
    assert res_mon.status_code == 200
    assert "active_alerts_count" in res_mon.json()

    res_geo = client.get("/api/geospatial/overview")
    assert res_geo.status_code == 200
    assert res_geo.json()["total_states_monitored"] > 0


def test_smoke_scenario_and_xai(client):
    """Verifies scenario simulation and local XAI attribution."""
    res_scn = client.post("/api/scenario/simulate", json={
        "year": 2017,
        "state": "Punjab",
        "district": "Ludhiana",
        "baseline_rice_area": 250.0,
        "scenario_rice_area": 275.0
    })
    assert res_scn.status_code == 200
    assert "scenario_id" in res_scn.json()

    res_xai = client.post("/api/explainability/prediction", json={
        "state": "Punjab",
        "district": "Ludhiana",
        "year": 2017,
        "area_1000_ha": 310.0
    })
    assert res_xai.status_code == 200
    assert "prediction_kg_ha" in res_xai.json()
    assert len(res_xai.json()["feature_contributions"]) > 0


def test_smoke_decision_intelligence(client):
    """Verifies Day 14 Decision Intelligence master synthesis endpoint."""
    res = client.post("/api/decision/analyze", json={
        "state": "Punjab",
        "district": "Ludhiana",
        "year": 2017
    })
    assert res.status_code == 200
    data = res.json()
    assert data["decision_id"].startswith("DEC-")
    assert "brief" in data
    assert len(data["brief"]["evidence_items"]) > 0
    assert len(data["brief"]["analytical_priorities"]) > 0
    assert len(data["brief"]["decision_options"]) > 0
    assert "audit_record" in data["brief"]
