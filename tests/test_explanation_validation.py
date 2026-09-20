"""
Unit tests for src/explanation_validation.py.
Verifies the 7-point scientific validation engine.
"""

import pytest
from src.explanation_validation import explanation_validator
from src.explainability_engine import explainability_engine


def test_explanation_validation_success():
    features = {
        'Year': 2017.0,
        'State Code': 12.0,
        'RICE AREA (1000 ha)': 300.0,
        'TOTAL_CROPPED_AREA': 600.0,
        'RICE_AREA_SHARE': 0.50,
        'WHEAT AREA (1000 ha)': 180.0,
        'COTTON AREA (1000 ha)': 40.0,
        'SUGARCANE AREA (1000 ha)': 25.0,
        'RICE_YIELD_LAG1': 3000.0,
        'RICE_YIELD_ROLL3': 2900.0
    }

    explanation = explainability_engine.explain_local_prediction(features, entity="Punjab")
    val_res = explanation_validator.validate_explanation(explanation, raw_features=features)

    assert isinstance(val_res, dict)
    assert val_res['is_valid'] is True
    assert val_res['passed_checks'] == 7
    assert val_res['validation_score_pct'] == 100.0


def test_explanation_validation_negative_area_failure():
    features = {
        'Year': 2017.0,
        'State Code': 12.0,
        'RICE AREA (1000 ha)': -50.0,  # Invalid negative area
    }

    explanation = explainability_engine.explain_local_prediction(features, entity="Punjab")
    val_res = explanation_validator.validate_explanation(explanation, raw_features=features)

    # Check that perturbation validity flagged the negative area
    rule_check = [c for c in val_res['checks'] if c['rule'] == 'Perturbation Validity'][0]
    assert rule_check['passed'] is False
    assert val_res['is_valid'] is False
