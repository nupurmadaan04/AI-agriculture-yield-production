"""
Unit tests for src/decision_options.py.
"""

import pytest
from src.decision_options import decision_options_engine


def test_build_decision_options():
    scenarios = [
        {
            "scenario_id": "SCEN-001",
            "name": "Moderate Improvement",
            "scenario_type": "moderate_improvement",
            "simulated_yield": 4120.0,
            "production_delta_pct": 5.2,
            "modifications": {"rice_area_share": 0.50}
        }
    ]

    opt_res = {
        "scenario_id": "SCEN-OPT-01",
        "optimal_solution": {
            "simulated_yield": 4180.0,
            "production_delta_pct": 6.8
        },
        "weights": {"yield_weight": 0.6}
    }

    ev_items = [{
        "evidence_id": "EV-SCEN-0001",
        "category": "scenario",
        "statement": "Scenario test",
        "value": 4120,
        "unit": "kg/ha",
        "source_module": "scenario_service",
        "source_method": "simulation",
        "evidence_type": "SIMULATED",
        "confidence_status": "VALIDATED",
        "timestamp": "2026-01-01",
        "model_version": "v2.1.0",
        "dataset_version": "ICRISAT"
    }]

    options = decision_options_engine.build_decision_options(
        base_yield_kg_ha=3950.0,
        base_area_1000_ha=300.0,
        scenario_results=scenarios,
        optimization_result=opt_res,
        evidence_items=ev_items
    )

    assert isinstance(options, list)
    assert len(options) >= 3
    assert options[0]["option_id"] == "OPT-STATUS-QUO"
    assert any(o["option_id"] == "OPT-PARETO-OPTIMAL" for o in options)
