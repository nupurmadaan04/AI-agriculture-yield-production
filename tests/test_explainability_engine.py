"""
Unit tests for src/explainability_engine.py.
Verifies global feature importance, local prediction attribution, and controlled sensitivity sweeps.
"""

import pytest
from src.explainability_engine import explainability_engine, ExplainabilityEngine


def test_engine_initialization():
    engine = ExplainabilityEngine()
    assert engine.model_path.exists()


def test_reference_baseline():
    baseline = explainability_engine.get_reference_baseline()
    assert isinstance(baseline, dict)
    assert 'RICE AREA (1000 ha)' in baseline
    assert 'State Code' in baseline
    assert baseline['RICE AREA (1000 ha)'] > 0


def test_global_feature_importance():
    res = explainability_engine.compute_global_feature_importance(n_repeats=2)
    assert isinstance(res, dict)
    assert res['model_id'] == 'exogenous_rf_forecaster'
    assert res['version'] == '2.1.0'
    assert len(res['features']) == 10

    # Verify normalization
    native_sum = sum(f['native_importance'] for f in res['features'])
    assert pytest.approx(native_sum, abs=0.05) == 1.0

    # Top feature check
    assert res['top_feature'] is not None
    assert 'scientific_disclaimer' in res


def test_local_prediction_explanation():
    sample_features = {
        'Year': 2017.0,
        'State Code': 12.0,
        'RICE AREA (1000 ha)': 310.0,
        'TOTAL_CROPPED_AREA': 600.0,
        'RICE_AREA_SHARE': 0.51,
        'WHEAT AREA (1000 ha)': 180.0,
        'COTTON AREA (1000 ha)': 40.0,
        'SUGARCANE AREA (1000 ha)': 25.0,
        'RICE_YIELD_LAG1': 3000.0,
        'RICE_YIELD_ROLL3': 2950.0
    }

    res = explainability_engine.explain_local_prediction(sample_features, entity='Punjab (Ludhiana)')
    assert isinstance(res, dict)
    assert res['prediction_kg_ha'] > 0
    assert res['baseline_reference_kg_ha'] > 0
    assert len(res['feature_contributions']) == 10
    assert len(res['top_positive_features']) <= 3

    # Check that contributions sum relative influence to 100%
    rel_sum = sum(c['relative_influence_pct'] for c in res['feature_contributions'])
    assert pytest.approx(rel_sum, abs=1.0) == 100.0


def test_feature_sensitivity():
    sample_features = {
        'Year': 2017.0,
        'State Code': 12.0,
        'RICE AREA (1000 ha)': 310.0,
        'RICE_YIELD_LAG1': 3000.0
    }

    res = explainability_engine.compute_feature_sensitivity(
        sample_features,
        target_features=['RICE_YIELD_LAG1', 'RICE AREA (1000 ha)']
    )

    assert isinstance(res, dict)
    assert 'RICE_YIELD_LAG1' in res['sensitivity_curves']
    curve = res['sensitivity_curves']['RICE_YIELD_LAG1']
    assert len(curve) == 5  # -10%, -5%, 0, +5%, +10%

    # Baseline step (0%) delta must be 0.0
    base_step = [p for p in curve if p['step_pct'] == 0.0][0]
    assert pytest.approx(base_step['prediction_delta_kg_ha'], abs=1e-3) == 0.0
