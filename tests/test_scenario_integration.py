"""
Day 32 Unit Tests: Scenario Integration & Comparison Matrix.

Verifies:
- Supported scenario archetypes (conservative, moderate, stress, custom)
- Quantitative delta calculations vs baseline
- Comparison matrix without subjective ranking labels
- Rejection or partial support of invalid/unsupported parameters
"""

import pytest
from backend.services.decision_workspace_service import decision_workspace_service


def test_scenario_archetypes_and_deltas():
    """Verify quantitative delta arithmetic and evidence types across standard archetypes."""
    res = decision_workspace_service.analyze_workspace(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017
    )
    base_yield = res["baseline_forecast"]["forecast_yield_kg_ha"]
    scenarios = res["scenarios"]
    assert len(scenarios) == 3

    for sc in scenarios:
        assert sc["evidence_type"] == "SCENARIO"
        assert sc["is_simulated"] is True
        assert sc["status"] == "SUPPORTED"
        expected_delta = round(sc["scenario_output_kg_ha"] - base_yield, 1)
        assert abs(sc["yield_delta_kg_ha"] - expected_delta) < 0.2
        expected_pct = round((expected_delta / base_yield * 100.0), 2)
        assert abs(sc["yield_percent_change"] - expected_pct) < 0.2


def test_scenario_comparison_matrix_no_subjective_ranking():
    """Comparison matrix must only display quantitative metrics, never ranking labels."""
    res = decision_workspace_service.analyze_workspace(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017
    )
    matrix = res["comparison_matrix"]
    assert "scenario_headers" in matrix
    assert len(matrix["scenario_headers"]) == 3
    assert len(matrix["rows"]) >= 5

    # Check for forbidden ranking words
    forbidden = ["BEST", "WORST", "WINNER", "RECOMMENDED", "OPTIMAL CHOICE"]
    matrix_str = str(matrix).upper()
    for word in forbidden:
        assert word not in matrix_str, f"Forbidden ranking word '{word}' found in comparison matrix"


def test_custom_scenario_modifications():
    """Verify custom what-if parameter modification handling."""
    custom_mods = {"rice_area_pct": 12.0, "historical_yield_lag_pct": 8.0}
    res = decision_workspace_service.analyze_workspace(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017,
        custom_modifications=custom_mods
    )
    assert len(res["scenarios"]) == 4
    custom_scen = res["scenarios"][-1]
    assert custom_scen["scenario_type"] == "custom"
    assert custom_scen["status"] == "SUPPORTED"
    assert custom_scen["yield_delta_kg_ha"] > 0
