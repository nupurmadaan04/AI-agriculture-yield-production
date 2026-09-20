"""
Unit tests for backend/services/explainability_service.py.
"""

import pytest
from backend.services.explainability_service import explainability_service


def test_service_global_importance():
    res = explainability_service.get_global_feature_importance()
    assert isinstance(res, dict)
    assert 'features' in res
    assert res['model_id'] == 'exogenous_rf_forecaster'


def test_service_explain_prediction():
    res = explainability_service.explain_prediction(state_val="Punjab", area=310.0, district="Ludhiana")
    assert isinstance(res, dict)
    assert 'prediction_kg_ha' in res
    assert 'feature_contributions' in res
    assert 'explanation_id' in res
    assert res['explanation_id'].startswith('EXP-')


def test_service_sensitivity():
    res = explainability_service.get_feature_sensitivity(state_val="Punjab")
    assert isinstance(res, dict)
    assert 'sensitivity_curves' in res
    assert res['base_prediction_kg_ha'] > 0


def test_service_explain_alert():
    res = explainability_service.explain_alert("ALR-000183")
    assert isinstance(res, dict)
    assert res['alert_id'] == 'ALR-000183'
    assert 'signal_breakdown' in res
    assert 'evidence_chain' in res


def test_service_explain_scenario():
    res = explainability_service.explain_scenario(
        scenario_id="SCEN-001",
        state="Punjab",
        baseline_yield=3950.0,
        simulated_yield=4150.0,
        changed_features={'RICE_AREA_SHARE': 0.55}
    )
    assert isinstance(res, dict)
    assert res['scenario_id'] == 'SCEN-001'
    assert res['simulated_delta_kg_ha'] == 200.0
    assert len(res['changed_inputs']) == 1
