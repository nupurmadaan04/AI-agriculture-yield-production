"""
Tests for Directional Bias Diagnostics and Stratified Error Decomposition (Day 30).
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_bias_diagnostics_structure():
    response = client.get("/api/monitoring/bias")
    assert response.status_code == 200
    data = response.json()
    assert data["semantic_classification"] == "POST_OUTCOME_EVALUATION"
    assert len(data["crops"]) >= 14
    for b in data["crops"]:
        assert b["bias_status"] in ["OVER_PREDICTION_BIAS", "UNDER_PREDICTION_BIAS", "NO_CLEAR_BIAS"]
        assert b["mean_actual_yield"] > 0
        assert "bias_threshold_rule" in b
        assert b["sample_count"] > 0


def test_bias_diagnostics_single_crop():
    response = client.get("/api/monitoring/bias?crop=Oilseeds")
    assert response.status_code == 200
    data = response.json()
    assert len(data["crops"]) == 1
    assert data["crops"][0]["crop"] == "Oilseeds"


def test_error_decomposition_temporal():
    response = client.get("/api/monitoring/errors?crop=Oilseeds")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Oilseeds"
    assert data["semantic_classification"] == "POST_OUTCOME_EVALUATION"
    assert len(data["temporal_breakdown"]) > 0
    for tb in data["temporal_breakdown"]:
        assert 2014 <= tb["year"] <= 2017
        assert tb["mae"] > 0
        assert tb["rmse"] >= tb["mae"]


def test_error_decomposition_regimes():
    response = client.get("/api/monitoring/errors?crop=Oilseeds")
    assert response.status_code == 200
    data = response.json()
    assert len(data["regime_breakdown"]) > 0
    for reg in data["regime_breakdown"]:
        assert reg["sample_count"] > 0
        assert reg["mae"] > 0
