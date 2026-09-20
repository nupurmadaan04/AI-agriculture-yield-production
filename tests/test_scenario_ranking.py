import pytest
from src.scenario_ranking import scenario_ranking_engine

def test_scenario_ranking_and_tradeoffs():
    candidates = [
        {
            'scenario_id': 'SCN-MOD',
            'scenario_name': 'Moderate Expansion',
            'scenario_type': 'moderate_improvement',
            'scenario_prediction': 3550.0,
            'risk_score': 32.0,
            'resource_change_pct': 10.0,
            'reliability_r2': 0.7866,
            'prediction_spread': 280.0
        },
        {
            'scenario_id': 'SCN-STRESS',
            'scenario_name': 'Acreage Contraction',
            'scenario_type': 'stress_scenario',
            'scenario_prediction': 2700.0,
            'risk_score': 65.0,
            'resource_change_pct': 15.0,
            'reliability_r2': 0.7866,
            'prediction_spread': 350.0
        }
    ]

    ranked = scenario_ranking_engine.rank_scenarios(
        candidates=candidates,
        baseline_yield=3200.0,
        baseline_risk=35.0,
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

    assert len(ranked) == 2
    # Moderate expansion passes constraint and has positive yield -> should be rank 1
    assert ranked[0]['scenario_id'] == 'SCN-MOD'
    assert ranked[0]['is_feasible'] is True
    assert ranked[0]['rank'] == 1
    assert len(ranked[0]['strengths']) > 0

    # Stress scenario violates max risk score 60 -> should be rank 2 (infeasible)
    assert ranked[1]['scenario_id'] == 'SCN-STRESS'
    assert ranked[1]['is_feasible'] is False
