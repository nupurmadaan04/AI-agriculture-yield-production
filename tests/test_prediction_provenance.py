"""
Test suite for Day 24 Prediction Provenance.
Verifies lineage metadata generation, hash integrity, and validation limitations.
"""

import pytest
from src.prediction_provenance import PredictionProvenanceBuilder


@pytest.fixture
def provenance_builder():
    return PredictionProvenanceBuilder()


def test_build_provenance_structure(provenance_builder):
    strategy_meta = {
        "primary_strategy": "Historical ML (RandomForestRegressor)",
        "model_name": "RandomForestRegressor",
        "model_version": "oilseeds_rf_v23",
        "certification_status": "PRODUCTION_READY",
        "strategy_mae": 549.67,
        "baseline_mae": 616.60,
        "gain_vs_baseline_pct": 10.85,
        "fold_win_rate_pct": 75.0,
    }
    prediction_result = {
        "prediction": 817.06,
        "unit": "kg/ha",
        "fallback_used": False,
        "evidence_type": "PREDICTED_ML",
    }
    prov = provenance_builder.build_provenance(
        request_id="REQ-TEST12345",
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2018,
        strategy_meta=strategy_meta,
        prediction_result=prediction_result,
    )

    assert prov["request_id"] == "REQ-TEST12345"
    assert prov["crop"] == "Oilseeds"
    assert prov["prediction"] == 817.06
    assert prov["certification_status"] == "PRODUCTION_READY"
    assert "provenance_hash" in prov
    assert prov["provenance_hash"].startswith("SHA256:")
    assert "validation_boundary_notice" in prov
