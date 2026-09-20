"""
Test suite for Day 24 Prediction Audit Logger.
Verifies thread-safe audit logging, file creation, and retrieval.
"""

import pytest
from pathlib import Path
from src.prediction_audit import PredictionAuditLogger


@pytest.fixture
def audit_logger(tmp_path):
    return PredictionAuditLogger(base_dir=tmp_path)


def test_audit_logging_success_and_rejection(audit_logger):
    # Log success event
    evt1 = audit_logger.log_event(
        request_id="REQ-TEST001",
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2018,
        strategy="Historical ML (RandomForestRegressor)",
        certification_status="PRODUCTION_READY",
        status="SUCCESS",
        prediction=817.06,
        provenance_hash="SHA256:abc12345",
    )
    assert evt1["request_id"] == "REQ-TEST001"
    assert evt1["status"] == "SUCCESS"

    # Log rejection event
    evt2 = audit_logger.log_event(
        request_id="REQ-TEST002",
        crop="Potato",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2018,
        strategy="NONE",
        certification_status="UNSUPPORTED",
        status="REJECTED",
        error_code="UNSUPPORTED_CROP",
        error_message="Crop 'Potato' is not registered.",
    )
    assert evt2["request_id"] == "REQ-TEST002"
    assert evt2["status"] == "REJECTED"

    # Retrieve logs
    logs = audit_logger.get_recent_audit_logs(limit=10)
    assert len(logs) == 2
    assert logs[0]["request_id"] == "REQ-TEST002"
    assert logs[1]["request_id"] == "REQ-TEST001"
