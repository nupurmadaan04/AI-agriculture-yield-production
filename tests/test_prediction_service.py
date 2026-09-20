"""
Test suite for Day 24 Prediction Service.
Verifies end-to-end inference coordination, provenance attachment, and rejection handling.
"""

import pytest
from src.prediction_service import PredictionService


@pytest.fixture
def service():
    return PredictionService()


def test_prediction_service_success_flow(service):
    res = service.predict_forecast(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017,
    )
    assert res["status"] == "SUCCESS"
    assert res["prediction"] is not None
    assert res["prediction"] > 0
    assert res["certification_status"] == "PRODUCTION_READY"
    assert res["provenance"] is not None
    assert res["provenance"]["provenance_hash"].startswith("SHA256:")


def test_prediction_service_rejection_flow(service):
    res = service.predict_forecast(
        crop="Dragonfruit",
        state="Goa",
        district="Panaji",
    )
    assert res["status"] == "REJECTED"
    assert res["error_code"] == "UNSUPPORTED_CROP"
    assert res["prediction"] is None
    assert res["provenance"] is None


def test_prediction_service_baseline_crop(service):
    res = service.predict_forecast(
        crop="Chickpea",
        state="Madhya Pradesh",
        district="Indore",
        forecast_year=2017,
    )
    assert res["status"] == "SUCCESS"
    assert res["certification_status"] == "BASELINE_PRODUCTION"
    assert res["evidence_type"] == "HISTORICAL_BASELINE"
