import pytest
from src.scenario_audit import scenario_audit_engine

def test_scenario_audit_record_generation():
    rec = scenario_audit_engine.generate_audit_record(
        location="Punjab (Ludhiana)",
        horizon=1,
        scenario_type="moderate_improvement",
        modified_features=[
            {'feature_key': 'rice_area', 'baseline_value': 200.0, 'scenario_value': 224.0, 'percent_change': 12.0}
        ],
        baseline_prediction=3200.0,
        scenario_prediction=3550.0
    )
    assert rec['scenario_id'].startswith("SCN-")
    assert rec['is_reproducible'] is True
    assert rec['model_version'] == 'exogenous_rf_forecaster_v2.1.0'
    assert rec['validation_r2'] == 0.7866
    assert rec['drift_status'] == 'NORMAL'
    assert rec['yield_delta'] == 350.0
