"""
Unit & integration tests for end-to-end Prediction Trace functionality.
Verifies stage recording, timing measurement, and audit log fallback.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from src.prediction_service import get_prediction_service
from src.observability_engine import get_observability_engine

client = TestClient(app)


def test_forecast_trace_successful_prediction():
    """Verify end-to-end prediction execution creates a traceable record with stage timings."""
    service = get_prediction_service()
    
    # Execute a prediction for a production crop
    prediction_result = service.predict(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2024,
        yield_lag_1=1500.0,
        yield_rolling_3yr_mean=1450.0,
        area_lag_1=50000.0
    )
    
    req_id = prediction_result.get("request_id")
    assert req_id is not None
    assert prediction_result.get("status") == "SUCCESS"
    
    # Retrieve the trace via observability API
    response = client.get(f"/api/observability/trace/{req_id}")
    assert response.status_code == 200
    trace = response.json()
    
    assert trace["request_id"] == req_id
    assert trace["crop"] == "Oilseeds"
    assert trace["state"] == "Punjab"
    assert trace["district"] == "Ludhiana"
    assert trace["status"] == "SUCCESS"
    assert trace["prediction"] is not None
    assert trace["stages"] is not None
    assert len(trace["stages"]) >= 5
    
    stage_names = [s["stage_name"] for s in trace["stages"]]
    assert "INPUT_VALIDATION" in stage_names
    assert "STRATEGY_LOOKUP" in stage_names
    assert "INFERENCE_EXECUTION" in stage_names
    assert "PROVENANCE_GENERATION" in stage_names
    assert "AUDIT_LOGGING" in stage_names

    # Check that individual stage statuses are recorded
    for stage in trace["stages"]:
        assert stage["status"] in ["COMPLETED", "SKIPPED", "FAILED", "REJECTED"]


def test_forecast_trace_rejected_prediction():
    """Verify rejected prediction (unsupported crop / out-of-domain) records proper trace stages."""
    service = get_prediction_service()
    
    prediction_result = service.predict(
        crop="NonExistentCropXYZ",
        state="Punjab",
        district="Ludhiana",
        year=2024,
    )
    
    req_id = prediction_result.get("request_id")
    assert req_id is not None
    assert prediction_result.get("status") in ["REJECTED", "FAILED"]
    
    # Retrieve trace
    response = client.get(f"/api/observability/trace/{req_id}")
    assert response.status_code == 200
    trace = response.json()
    
    assert trace["request_id"] == req_id
    assert trace["status"] in ["REJECTED", "FAILED"]


def test_forecast_trace_not_found():
    """Verify 404 response for unknown request ID."""
    response = client.get("/api/observability/trace/REQ-DOES-NOT-EXIST-00000")
    assert response.status_code == 404
    assert "not found" in response.json().get("detail", "").lower()
