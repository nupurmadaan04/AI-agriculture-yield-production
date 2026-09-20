import pytest
from src.decision_optimizer import decision_optimizer

def test_decision_optimizer_linear_scalarization():
    candidate = {
        'projected_yield': 3300.0,
        'risk_score': 30.0,
        'resource_change_pct': 5.0,
        'validation_r2': 0.7866
    }
    score = decision_optimizer.compute_objective_score(
        candidate=candidate,
        baseline_yield=3000.0,
        baseline_risk=40.0,
        weights={
            'yield_improvement': 0.40,
            'risk_reduction': 0.25,
            'resource_efficiency': 0.20,
            'model_reliability': 0.15
        }
    )
    assert 0.0 <= score <= 100.0

def test_decision_optimizer_constraints():
    # Feasible candidate
    cand_feasible = {
        'projected_yield': 3500.0,
        'risk_score': 40.0,
        'resource_change_pct': 10.0,
        'validation_r2': 0.7866
    }
    feasible, status = decision_optimizer.evaluate_constraints(
        candidate=cand_feasible,
        min_yield=3000.0,
        max_risk_score=50.0,
        max_resource_change_pct=15.0,
        min_reliability_score=0.70
    )
    assert feasible is True
    assert status['min_yield']['passed'] is True
    assert status['max_risk']['passed'] is True

    # Infeasible candidate exceeding risk and violating min yield
    cand_infeasible = {
        'projected_yield': 2500.0,
        'risk_score': 65.0,
        'resource_change_pct': 10.0,
        'validation_r2': 0.7866
    }
    infeasible, status = decision_optimizer.evaluate_constraints(
        candidate=cand_infeasible,
        min_yield=3000.0,
        max_risk_score=50.0
    )
    assert infeasible is False
    assert status['min_yield']['passed'] is False
    assert status['max_risk']['passed'] is False

def test_decision_optimizer_pareto_frontier():
    candidates = [
        # Candidate 1: High yield gain (3600), low risk (25), low resource (5%) -> Dominates C3
        {
            'scenario_id': 'C1',
            'scenario_name': 'Candidate 1',
            'projected_yield': 3600.0,
            'risk_score': 25.0,
            'resource_change_pct': 5.0
        },
        # Candidate 2: Very high yield gain (3800), medium risk (40), higher resource (15%) -> Trade-off (Pareto)
        {
            'scenario_id': 'C2',
            'scenario_name': 'Candidate 2',
            'projected_yield': 3800.0,
            'risk_score': 40.0,
            'resource_change_pct': 15.0
        },
        # Candidate 3: Lower yield (3400), higher risk (35), higher resource (10%) -> Dominated by C1
        {
            'scenario_id': 'C3',
            'scenario_name': 'Candidate 3',
            'projected_yield': 3400.0,
            'risk_score': 35.0,
            'resource_change_pct': 10.0
        }
    ]

    pareto = decision_optimizer.identify_pareto_frontier(candidates)
    pareto_ids = [c['scenario_id'] for c in pareto]
    assert 'C1' in pareto_ids
    assert 'C2' in pareto_ids
    assert 'C3' not in pareto_ids  # C3 is dominated by C1
