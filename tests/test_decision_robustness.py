"""
Unit tests for src/decision_robustness.py.
"""

import pytest
from src.decision_robustness import decision_robustness_engine


def test_decision_robustness_classification():
    option = {
        "option_id": "OPT-01",
        "title": "Acreage Adjustment",
        "projected_yield_delta_kg_ha": 120.0
    }

    sens_matrix = {
        "sensitivity_curves": {
            "RICE_AREA_SHARE": [
                {"step_pct": -0.1, "prediction_delta_kg_ha": -35.0},
                {"step_pct": 0.1, "prediction_delta_kg_ha": 40.0}
            ]
        }
    }

    res = decision_robustness_engine.evaluate_option_robustness(option, sens_matrix)
    assert res["classification"] == "ROBUST"
    assert res["is_favorable"] is True
