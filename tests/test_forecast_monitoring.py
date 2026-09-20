"""
Tests for Forecast Operations Telemetry, Prediction Distributions, and Monitoring Health (Day 30).
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_monitoring_summary_structure():
    response = client.get("/api/monitoring/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["monitoring_status"] in ["HEALTHY", "WATCH", "DRIFT_DETECTED", "EVALUATION_UNAVAILABLE"]
    assert "status_reason" in data
    assert isinstance(data["total_forecast_requests"], int)
    assert isinstance(data["successful_forecasts"], int)
    assert isinstance(data["rejected_requests"], int)
    assert isinstance(data["evaluated_outcomes_count"], int)
    assert data["monitored_crops_count"] == 14
    assert data["dataset_version"] == "AGRI_PANEL_1.0"


def test_forecast_operations_aggregation():
    response = client.get("/api/monitoring/operations")
    assert response.status_code == 200
    data = response.json()
    assert data["total_requests"] >= 0
    assert data["successful_requests"] >= 0
    assert data["rejected_requests"] >= 0
    assert data["failed_requests"] >= 0
    assert 0.0 <= data["success_rate_pct"] <= 100.0
    assert data["semantic_classification"] == "MONITORING"
    assert isinstance(data["crop_breakdown"], list)
    assert isinstance(data["strategy_breakdown"], list)


def test_forecast_operations_filtered_by_crop():
    response = client.get("/api/monitoring/operations?crop=Oilseeds")
    assert response.status_code == 200
    data = response.json()
    assert data["semantic_classification"] == "MONITORING"


def test_prediction_distributions_moments():
    response = client.get("/api/monitoring/distributions?crop=Oilseeds")
    assert response.status_code == 200
    data = response.json()
    assert data["total_monitored_crops"] >= 1
    dist = data["distributions"][0]
    assert dist["crop"] == "Oilseeds"
    assert dist["unit"] == "kg/ha"
    assert dist["semantic_classification"] == "MONITORING"
    
    # Verify historical reference statistical moments
    hist = dist["historical_reference"]
    assert hist["count"] > 0
    assert hist["mean"] > 0
    assert hist["max_val"] >= hist["min_val"]
    assert hist["p90"] >= hist["p10"]


def test_monitoring_alerts_evidence_structure():
    response = client.get("/api/monitoring/forecast-alerts")
    assert response.status_code == 200
    data = response.json()
    assert data["semantic_classification"] == "MONITORING"
    assert isinstance(data["active_alerts"], list)
    assert data["total_alerts"] == len(data["active_alerts"])
    for alert in data["active_alerts"]:
        assert "alert_id" in alert
        assert alert["severity"] in ["INFO", "WATCH", "WARNING", "CRITICAL"]
        assert alert["category"] in ["DRIFT", "BIAS", "OPERATIONAL", "INTEGRITY", "DATASET"]
        assert "metric" in alert
        assert "observed_value" in alert
        assert "evidence" in alert
        assert "recommended_action" in alert


def test_monitoring_health():
    response = client.get("/api/monitoring/forecast-health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["subsystem"] == "forecast-monitoring-engine"
    assert data["outcomes_dataset_available"] is True

