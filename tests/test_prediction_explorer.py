"""
Day 29 Prediction Explorer & Forecast Explainability Tests.
Verifies historical context extraction, verified evidence serving, empirical uncertainty limits,
and baseline vs ML attribution distinctions.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_forecast_context_valid_oilseeds():
    """Verifies that /api/forecast/context extracts real historical observations for Oilseeds in Ludhiana."""
    response = client.get("/api/forecast/context?crop=Oilseeds&state=Punjab&district=Ludhiana&forecast_year=2018")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Oilseeds"
    assert data["state"] == "Punjab"
    assert data["district"] == "Ludhiana"
    assert data["forecast_year"] == 2018
    assert data["historical_observations_count"] > 0
    assert data["district_historical_mean"] is not None
    assert data["has_sufficient_history"] is True
    assert len(data["recent_observations"]) > 0
    for obs in data["recent_observations"]:
        assert obs["year"] < 2018
        assert "yield_kg_ha" in obs
        assert obs["observation_type"] == "OBSERVED"


def test_forecast_context_sparse_or_unknown():
    """Verifies safe degradation for unknown or unsupported geographies."""
    response = client.get("/api/forecast/context?crop=Oilseeds&state=UnknownState&district=UnknownDistrict&forecast_year=2018")
    assert response.status_code == 200
    data = response.json()
    assert data["historical_observations_count"] == 0
    assert data["has_sufficient_history"] is False
    assert "No prior historical observations" in data["context_notes"]


def test_forecast_evidence_ml_crop_oilseeds():
    """Verifies that ML production crop returns verified feature importance, walk-forward evidence, and empirical spread."""
    response = client.get("/api/forecast/evidence/Oilseeds")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Oilseeds"
    assert data["is_ml_strategy"] is True
    assert data["certification_status"] == "PRODUCTION_READY"
    assert data["validation_protocol"] == "4-Origin Expanding Walk-Forward (2014-2017)"
    assert data["mean_mae"] == 549.67
    assert data["baseline_mae"] == 616.60
    assert data["mean_improvement_pct"] == 10.85
    assert data["empirical_p10_p90_spread"] == 797.62
    assert len(data["feature_importance"]) > 0
    # Confirm top feature is yield_lag_1
    top_feature = data["feature_importance"][0]
    assert top_feature["feature_name"] == "yield_lag_1"
    assert top_feature["importance_pct"] > 40.0
    assert "TreeSHAP" not in data["explanation_notice"]
    assert "SHAP" not in data["explanation_notice"]


def test_forecast_evidence_baseline_crop_rice():
    """Verifies that statistical baseline crop does NOT manufacture fake ML feature importances or confidence bounds."""
    response = client.get("/api/forecast/evidence/Rice")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Rice"
    assert data["is_ml_strategy"] is False
    assert data["certification_status"] == "BASELINE_PRODUCTION"
    assert data["empirical_p10_p90_spread"] is None
    assert len(data["feature_importance"]) == 0
    assert "Feature-level ML attribution is not applicable" in data["explanation_notice"]


def test_forecast_evidence_unsupported_crop():
    """Verifies handling for crops outside the governed catalog."""
    response = client.get("/api/forecast/evidence/NonExistentCrop")
    assert response.status_code == 200
    data = response.json()
    assert data["certification_status"] == "UNSUPPORTED"
    assert data["is_ml_strategy"] is False
