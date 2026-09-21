"""
Day 32 Unit Tests: Decision Workspace Golden Cases & Strategy Routing.

Verifies:
- Oilseeds (ML certified PRODUCTION_READY)
- Sugarcane (ML certified CONDITIONAL_PRODUCTION)
- Rice (BASELINE_PRODUCTION persistence benchmark)
- Wheat (BASELINE_PRODUCTION persistence benchmark)
- Separation of ML-only vs baseline evidence
- REST endpoints response schemas
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.decision_workspace_service import decision_workspace_service

client = TestClient(app)


def test_decision_workspace_oilseeds_golden_case():
    """Oilseeds in Punjab: Must route to RandomForest ML, expose Tree SHAP & P10-P90 uncertainty."""
    res = decision_workspace_service.analyze_workspace(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017
    )
    assert res["crop"] == "Oilseeds"
    assert res["baseline_forecast"]["certification_status"] == "PRODUCTION_READY"
    assert "RandomForest" in res["baseline_forecast"]["strategy"]
    assert res["validation"]["is_ml_certified"] is True
    assert res["validation"]["fold_win_rate_pct"] >= 50.0
    assert res["validation"]["mean_improvement_pct"] > 5.0
    assert res["uncertainty"]["is_available"] is True
    assert res["uncertainty"]["empirical_p10_kg_ha"] is not None
    assert res["uncertainty"]["empirical_p90_kg_ha"] is not None
    assert res["attribution"]["is_available"] is True
    assert res["attribution"]["attribution_type"] == "TREE_SHAP"
    assert len(res["attribution"]["top_features"]) >= 2
    assert len(res["scenarios"]) >= 3


def test_decision_workspace_sugarcane_golden_case():
    """Sugarcane in Uttar Pradesh: Must route to GradientBoosting ML, CONDITIONAL_PRODUCTION."""
    res = decision_workspace_service.analyze_workspace(
        crop="Sugarcane",
        state="Uttar Pradesh",
        district="Meerut",
        forecast_year=2017
    )
    assert res["crop"] == "Sugarcane"
    assert res["baseline_forecast"]["certification_status"] == "CONDITIONAL_PRODUCTION"
    assert res["validation"]["is_ml_certified"] is True
    assert res["uncertainty"]["is_available"] is True


def test_decision_workspace_rice_baseline_golden_case():
    """Rice in Punjab: Must route to baseline persistence; must NOT expose ML-only evidence."""
    res = decision_workspace_service.analyze_workspace(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017
    )
    assert res["crop"] == "Rice"
    assert res["baseline_forecast"]["certification_status"] == "BASELINE_PRODUCTION"
    assert "Persistence" in res["baseline_forecast"]["strategy"]
    assert res["validation"]["is_ml_certified"] is False
    assert res["validation"]["legacy_benchmark_note"] is not None
    assert "0.7866" in res["validation"]["legacy_benchmark_note"]
    # Baseline strategies must NOT expose ML-only evidence
    assert res["uncertainty"]["is_available"] is False
    assert res["uncertainty"]["empirical_p10_kg_ha"] is None
    assert res["attribution"]["is_available"] is False
    assert res["attribution"]["attribution_type"] == "PERSISTENCE_BASELINE"


def test_decision_workspace_wheat_baseline_golden_case():
    """Wheat in Haryana: Must route to historical baseline without overfitting ML model."""
    res = decision_workspace_service.analyze_workspace(
        crop="Wheat",
        state="Haryana",
        district="Karnal",
        forecast_year=2017
    )
    assert res["crop"] == "Wheat"
    assert res["baseline_forecast"]["certification_status"] == "BASELINE_PRODUCTION"
    assert res["validation"]["is_ml_certified"] is False
    assert res["uncertainty"]["is_available"] is False


def test_decision_workspace_rest_endpoints():
    """Verify REST API routes /api/workspace/analyze and /api/workspace/templates."""
    t_resp = client.get("/api/workspace/templates")
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert "archetypes" in t_data
    assert len(t_data["archetypes"]) >= 3

    a_resp = client.post("/api/workspace/analyze", json={
        "crop": "Oilseeds",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    })
    assert a_resp.status_code == 200
    a_data = a_resp.json()
    assert "baseline_forecast" in a_data
    assert "scenarios" in a_data
    assert "comparison_matrix" in a_data
    assert len(a_data["comparison_matrix"]["rows"]) >= 5
