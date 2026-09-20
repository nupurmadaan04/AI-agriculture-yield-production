"""
Unit & integration tests for operational alert rule evaluations and thresholds.
Verifies Day 28 Alerting requirements.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.observability_engine import get_observability_engine

client = TestClient(app)


def test_alerts_endpoint_structure():
    """Verify /api/observability/alerts returns configured thresholds and active alert states."""
    response = client.get("/api/observability/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "active_alerts" in data
    assert "resolved_alerts" in data
    assert "configured_thresholds" in data
    
    thresholds = data["configured_thresholds"]
    assert "OBSERVABILITY_ERROR_RATE_THRESHOLD_PCT" in thresholds
    assert "OBSERVABILITY_P95_LATENCY_THRESHOLD_MS" in thresholds
    assert "OBSERVABILITY_MEMORY_THRESHOLD_PERCENT" in thresholds


def test_alert_rule_evaluation_under_normal_conditions():
    """Verify alert structure evaluation under normal conditions."""
    engine = get_observability_engine()
    alerts_data = engine.evaluate_alerts()
    assert "active_alerts" in alerts_data
    assert "active_alerts_count" in alerts_data


def test_alert_rule_evaluation_on_error_spike():
    """Verify high error rate alert triggers when error threshold is breached."""
    engine = get_observability_engine()
    
    # Inject synthetic errors to trigger error rate alert
    for i in range(25):
        engine.record_request_telemetry(
            request_id=f"test-alert-err-{i}",
            method="POST",
            endpoint="/api/forecast/predict",
            status_code=500,
            duration_ms=15.0,
            error_type="INTERNAL_SERVER_ERROR"
        )
    
    alerts_data = engine.evaluate_alerts()
    high_err_alert = next((a for a in alerts_data["active_alerts"] if a["alert_id"] == "ALT-ERROR-RATE"), None)
    assert high_err_alert is not None
    assert high_err_alert["severity"] == "CRITICAL"
    assert "exceeds" in high_err_alert["message"] or "breached" in high_err_alert["message"]
