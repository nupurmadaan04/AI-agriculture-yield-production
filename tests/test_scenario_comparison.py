import pytest
from src.scenario_comparison import scenario_comparison_engine

def test_scenario_comparison_generation():
    baseline = {
        'scenario_id': 'SCN-BASE',
        'scenario_name': 'Baseline Scenario',
        'scenario_type': 'baseline',
        'location': 'Punjab (Ludhiana)',
        'horizon': 1,
        'scenario_prediction': 3200.0,
        'risk_score': 35.0,
        'warning_score': 20.0,
        'prediction_spread': 250.0
    }

    scenarios = [
        {
            'scenario_id': 'SCN-MOD',
            'scenario_name': 'Moderate Expansion',
            'scenario_type': 'moderate_improvement',
            'scenario_prediction': 3520.0,
            'risk_score': 38.0,
            'warning_score': 22.0,
            'prediction_spread': 280.0
        },
        {
            'scenario_id': 'SCN-STRESS',
            'scenario_name': 'Acreage Contraction',
            'scenario_type': 'stress_scenario',
            'scenario_prediction': 2800.0,
            'risk_score': 55.0,
            'warning_score': 45.0,
            'prediction_spread': 320.0
        }
    ]

    res = scenario_comparison_engine.compare_scenarios(
        baseline_result=baseline,
        scenario_results=scenarios
    )

    assert res['location'] == "Punjab (Ludhiana)"
    assert res['horizon'] == 1
    assert res['baseline_yield'] == 3200.0
    assert len(res['comparison_matrix']) == 3
    assert res['yield_range_kg_ha'] == 720.0  # 3520 - 2800

    mod_row = next(r for r in res['comparison_matrix'] if r['scenario_id'] == 'SCN-MOD')
    assert mod_row['yield_delta'] == 320.0
    assert mod_row['yield_percent_change'] == 10.0
    assert mod_row['risk_delta'] == 3.0
