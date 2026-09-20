import pytest
from src.sensitivity_analysis import sensitivity_analysis_engine

def test_sensitivity_analysis_execution():
    res = sensitivity_analysis_engine.analyze_sensitivity(
        state="Punjab",
        district="Ludhiana",
        horizon=1
    )

    assert "Punjab" in res['location']
    assert "Ludhiana" in res['location']
    assert res['horizon'] == 1
    assert res['baseline_prediction'] > 0
    assert len(res['sensitivity_matrix']) >= 4
    assert res['most_sensitive_feature'] is not None

    # Check perturbation values
    first_feat = res['sensitivity_matrix'][0]
    steps = [p['perturbation_pct'] for p in first_feat['perturbation_responses']]
    assert steps == [-20.0, -10.0, 0.0, 10.0, 20.0]
    assert first_feat['elasticity_index'] >= 0.0
