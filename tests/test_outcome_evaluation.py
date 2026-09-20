"""
Tests for Post-Outcome Evaluation, Temporal Leak Prevention, and Error Metrics (Day 30).
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_outcome_evaluation_historical_horizon():
    response = client.get("/api/monitoring/outcomes?crop=Oilseeds&forecast_year=2017")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "EVALUATED"
    assert data["semantic_classification"] == "POST_OUTCOME_EVALUATION"
    assert data["summary"] is not None
    assert data["summary"]["mae"] > 0
    assert data["summary"]["rmse"] >= data["summary"]["mae"]
    assert len(data["records"]) > 0
    
    # Check individual record signed and absolute error mathematics
    rec = data["records"][0]
    assert rec["forecast_origin"] < rec["forecast_year"]
    expected_signed = round(rec["predicted_yield"] - rec["observed_yield"], 2)
    assert abs(rec["signed_error"] - expected_signed) < 0.01


def test_outcome_evaluation_future_horizon_leak_prevention():
    # Attempting to evaluate an unharvested future year must return EVALUATION_UNAVAILABLE
    response = client.get("/api/monitoring/outcomes?crop=Oilseeds&forecast_year=2026")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "EVALUATION_UNAVAILABLE"
    assert "unavailable" in data["reason"].lower()
    assert data["summary"] is None
    assert len(data["records"]) == 0
    assert "Strict Pre-Forecast Freezing" in data["temporal_boundary_rule"]


def test_outcome_evaluation_all_crops():
    response = client.get("/api/monitoring/outcomes")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "EVALUATED"
    assert data["total_records"] > 0
    assert data["summary"] is not None
    assert data["summary"]["evaluated_samples"] == data["total_records"]


def test_outcome_evaluation_unsupported_crop():
    response = client.get("/api/monitoring/outcomes?crop=InvalidNonExistentCrop")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "EVALUATION_UNAVAILABLE"
    assert len(data["records"]) == 0
