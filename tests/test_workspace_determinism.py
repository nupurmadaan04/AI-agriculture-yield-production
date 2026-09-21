"""
Day 32 Unit Tests: Bitwise Deterministic Workspace Inference.

Verifies:
- Repeated queries with identical inputs yield identical baseline forecasts,
  scenario projections, quantitative deltas, and validation metrics.
"""

import pytest
from backend.services.decision_workspace_service import decision_workspace_service


def test_workspace_analytical_determinism():
    """Duplicate workspace executions must produce identical analytical outputs."""
    res1 = decision_workspace_service.analyze_workspace(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017
    )
    res2 = decision_workspace_service.analyze_workspace(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2017
    )

    # Invariant analytical values
    assert res1["baseline_forecast"]["forecast_yield_kg_ha"] == res2["baseline_forecast"]["forecast_yield_kg_ha"]
    assert res1["baseline_forecast"]["strategy"] == res2["baseline_forecast"]["strategy"]
    assert res1["baseline_forecast"]["provenance_hash"] == res2["baseline_forecast"]["provenance_hash"]
    assert res1["historical_context"]["historical_mean_yield_kg_ha"] == res2["historical_context"]["historical_mean_yield_kg_ha"]
    assert res1["validation"]["mae_kg_ha"] == res2["validation"]["mae_kg_ha"]

    assert len(res1["scenarios"]) == len(res2["scenarios"])
    for s1, s2 in zip(res1["scenarios"], res2["scenarios"]):
        assert s1["scenario_output_kg_ha"] == s2["scenario_output_kg_ha"]
        assert s1["yield_delta_kg_ha"] == s2["yield_delta_kg_ha"]
        assert s1["yield_percent_change"] == s2["yield_percent_change"]
