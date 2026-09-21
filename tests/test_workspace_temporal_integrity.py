"""
Day 32 Unit Tests: Temporal Boundaries & Post-Outcome Isolation.

Verifies:
- All historical baseline reference observations strictly satisfy Year < forecast_year
- Future/unharvested years (2026) return EVALUATION_UNAVAILABLE
- Historical evaluation preserves leak-free separation
"""

import pytest
from backend.services.decision_workspace_service import decision_workspace_service


def test_temporal_boundary_isolation():
    """All historical points must have Year < forecast_year."""
    forecast_year = 2015
    res = decision_workspace_service.analyze_workspace(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        forecast_year=forecast_year
    )
    hist = res["historical_context"]
    assert hist["end_year"] < forecast_year
    for obs in hist["recent_observations"]:
        assert obs["year"] < forecast_year, f"Found lookahead leak: {obs['year']} >= {forecast_year}"


def test_future_unharvested_horizon_handling():
    """Unharvested future year 2026 must return EVALUATION_UNAVAILABLE without fabricated outcomes."""
    res = decision_workspace_service.analyze_workspace(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        forecast_year=2026
    )
    monitoring = res["monitoring"]
    assert monitoring["outcome_evaluation_status"] == "EVALUATION_UNAVAILABLE"
    assert monitoring["observed_outcome_kg_ha"] is None
    assert monitoring["forecast_error_kg_ha"] is None
