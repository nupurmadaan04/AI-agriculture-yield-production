"""
Integration tests for Day 12 Monitoring REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_monitoring_overview():
    response = client.get("/api/monitoring/overview")
    assert response.status_code == 200
    data = response.json()
    assert "active_alerts_count" in data
    assert "summary" in data
    assert data["model_monitoring_status"] == "HEALTHY"


def test_api_monitoring_timeline():
    response = client.get("/api/monitoring/timeline?state=Punjab&metric=yield")
    assert response.status_code == 200
    data = response.json()
    assert data["metric_name"] == "yield"
    assert "rolling_3yr_mean" in data


def test_api_monitoring_alerts():
    response = client.get("/api/monitoring/alerts?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        alert_id = data[0]["alert_id"]
        detail_resp = client.get(f"/api/monitoring/alerts/{alert_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["alert_id"] == alert_id


def test_api_monitoring_warning_map():
    response = client.get("/api/monitoring/warning-map")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 20


def test_api_monitoring_health():
    response = client.get("/api/monitoring/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "overall_health_score" in data


def test_api_monitoring_query():
    response = client.post("/api/monitoring/query", json={"state": "Punjab", "limit": 5})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_api_monitoring_backtest():
    response = client.post("/api/monitoring/backtest", json={
        "yield_drop_threshold_pct": -10.0,
        "warning_zscore_threshold": 1.2,
        "lead_time_years": 1
    })
    assert response.status_code == 200
    data = response.json()
    assert "precision" in data
    assert "recall" in data
    assert data["is_chronologically_valid"] is True
