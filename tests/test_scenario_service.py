import pytest
from backend.services.scenario_service import scenario_service
from backend.services.sensitivity_service import sensitivity_service
from backend.services.optimization_service import optimization_service

def test_scenario_service_multi_simulate():
    res = scenario_service.simulate(
        state="Punjab",
        district="Ludhiana",
        horizon=1,
        scenario_type="conservative_improvement"
    )
    assert res['scenario_id'].startswith('SCN-')
    assert "Punjab" in res['location'] and "Ludhiana" in res['location']
    assert res['baseline_prediction'] > 0
    assert res['scenario_prediction'] > 0
    assert res['validation_context']['validation_r2'] == 0.7866

def test_scenario_service_compare():
    res = scenario_service.compare(
        state="Punjab",
        horizon=1
    )
    assert res['location'] == "Punjab"
    assert len(res['comparison_matrix']) >= 4

def test_sensitivity_service_run():
    res = sensitivity_service.run(
        state="Punjab",
        horizon=1
    )
    assert res['location'] == "Punjab"
    assert len(res['sensitivity_matrix']) >= 3
    assert res['most_sensitive_feature'] is not None

def test_optimization_service_solve():
    res = optimization_service.solve(
        state="Punjab",
        horizon=1,
        weights={
            'yield_improvement': 0.40,
            'risk_reduction': 0.25,
            'resource_efficiency': 0.20,
            'model_reliability': 0.15
        },
        constraints={
            'max_risk_score': 60.0
        }
    )
    assert res['recommended_scenario'] is not None
    assert len(res['pareto_alternatives']) >= 1
