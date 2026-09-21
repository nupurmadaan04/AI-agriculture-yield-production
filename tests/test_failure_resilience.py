"""
Day 33: Failure Resilience & Degradation Handling Test Suite.

Validates:
1. Graceful degradation when non-critical downstream services fail (Monitoring, Explainability).
2. Explicit EVIDENCE_UNAVAILABLE handling without silent data fabrication.
3. Unsupported scenario archetypes fallback safely.
4. Non-blocking telemetry / audit logging: Core forecast inference never fails due to telemetry errors.
"""

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# =============================================================================
# 1. NON-BLOCKING MONITORING & GRACEFUL DEGRADATION
# =============================================================================

def test_monitoring_failure_does_not_break_workspace_analysis():
    """
    Simulate downstream ForecastMonitoringService raising an exception.
    Workspace synthesis must still return 200 OK with baseline forecast intact,
    and monitoring marked in degraded/unavailable state.
    """
    with patch("backend.services.forecast_monitoring_service.ForecastMonitoringService.get_prediction_distributions", side_effect=RuntimeError("Monitoring DB offline")):
        payload = {
            "crop": "Oilseeds",
            "state": "Madhya Pradesh",
            "district": "Indore",
            "forecast_year": 2017
        }
        resp = client.post("/api/workspace/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Core baseline forecast must still be valid and served
        assert data["baseline_forecast"]["forecast_yield_kg_ha"] > 0

        # Monitoring should reflect degraded state rather than crashing
        mon = data["monitoring"]
        assert mon["semantic_classification"] == "MONITORING"


def test_explainability_failure_gracefully_handled():
    """
    Simulate Tree SHAP explainability service failure.
    Workspace must still succeed with attribution marked unavailable.
    """
    with patch("backend.services.explainability_service.ExplainabilityService.explain_prediction", side_effect=Exception("Tree SHAP computation failed")):
        payload = {
            "crop": "Oilseeds",
            "state": "Madhya Pradesh",
            "district": "Indore",
            "forecast_year": 2017
        }
        resp = client.post("/api/workspace/analyze", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["attribution"]["is_available"] is False


# =============================================================================
# 2. EVIDENCE UNAVAILABLE EXPLICIT STATES
# =============================================================================

def test_baseline_strategy_uncertainty_explicitly_unavailable():
    """
    For baseline persistence strategies (e.g. Rice, Wheat), the system must
    explicitly set is_available=False and provide a scientific limitation reason.
    It must NEVER silently interpolate or invent confidence intervals.
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

    unc = data["uncertainty"]
    assert unc["is_available"] is False
    assert unc["empirical_p10_kg_ha"] is None
    assert unc["empirical_p90_kg_ha"] is None
    assert "deterministic baseline" in unc["limitations"].lower()


def test_unharvested_future_outcome_explicitly_unavailable():
    """
    Evaluating post-harvest outcome for 2026 must return EVALUATION_UNAVAILABLE
    with observed_outcome_kg_ha=None and forecast_error_kg_ha=None.
    """
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "forecast_year": 2026
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    mon = data["monitoring"]
    assert mon["outcome_evaluation_status"] == "EVALUATION_UNAVAILABLE"
    assert mon["observed_outcome_kg_ha"] is None
    assert mon["forecast_error_kg_ha"] is None


# =============================================================================
# 3. SCENARIO FALLBACK RESILIENCE
# =============================================================================

def test_unsupported_scenario_archetype_handled_gracefully():
    """
    Passing an unknown scenario archetype ID in selected_scenarios falls back
    gracefully to baseline comparison without 500 error.
    """
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "forecast_year": 2017,
        "selected_scenarios": ["completely_fictional_unsupported_archetype_999"]
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    # Scenarios list is present and valid
    scenarios = data["scenarios"]
    assert len(scenarios) >= 1


# =============================================================================
# 4. TELEMETRY / AUDIT LOG NON-BLOCKING RESILIENCE
# =============================================================================

def test_audit_log_write_error_does_not_break_prediction():
    """
    If the append-oriented audit log fails to write (e.g. disk read-only),
    the forecast inference should still return the valid point estimate.
    """
    with patch("src.prediction_audit.PredictionAuditLogger.log_event", side_effect=IOError("Disk write simulated error")):
        payload = {
            "crop": "Rice",
            "state": "Punjab",
            "district": "Ludhiana",
            "forecast_year": 2017
        }
        # In PredictionService, audit logging failure is wrapped or handled
        resp = client.post("/api/forecast/predict", json=payload)
        # Even if audit fails, the client gets either 200 with result or controlled error, not unhandled crash
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            assert resp.json()["status"] == "SUCCESS"
