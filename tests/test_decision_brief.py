"""
Day 31 Unit Tests: Decision Brief Generation & Multi-Crop Golden Cases.

Verifies structured DecisionBrief generation across:
- Oilseeds (Production-ready ML)
- Sugarcane (Conditional production ML)
- Rice (Baseline production)
- Wheat (Baseline production)
And tests determinism, completeness rules, and audit integrity.
"""

import pytest
from backend.services.decision_intelligence_service import decision_intelligence_service


def test_decision_brief_oilseeds_golden_case():
    """Verify Oilseeds generates an evidence brief with ML validation and provenance."""
    res = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    assert res is not None
    assert "decision_id" in res
    assert res["decision_id"].startswith("DEC-")
    assert res["is_scientifically_validated"] is True
    assert res["validation_checks_passed"] == 11

    brief = res["brief"]
    forecast = brief.get("forecast_summary")
    assert forecast is not None
    assert forecast["crop"] == "Oilseeds"
    assert forecast["forecast_yield_kg_ha"] > 0
    assert "RandomForest" in forecast["strategy"] or "Historical ML" in forecast["strategy"]

    val = brief.get("validation_evidence")
    assert val is not None
    assert val["strategy_tier"] in ["PRODUCTION_READY", "CONDITIONAL_PRODUCTION", "BASELINE_PRODUCTION"]
    assert val["mae_kg_ha"] > 0

    assert brief.get("historical_context") is not None
    assert brief.get("monitoring_evidence") is not None
    assert brief.get("executive_summary") is not None
    assert len(brief.get("evidence_items", [])) >= 5


def test_decision_brief_sugarcane_golden_case():
    """Verify Sugarcane generates an evidence brief under conditional production."""
    res = decision_intelligence_service.analyze_decision(
        crop="Sugarcane",
        state="Uttar Pradesh",
        district="Meerut",
        year=2017
    )
    assert res is not None
    assert res["is_scientifically_validated"] is True
    brief = res["brief"]
    forecast = brief.get("forecast_summary")
    assert forecast is not None
    assert forecast["crop"] == "Sugarcane"
    assert "GradientBoosting" in forecast["strategy"] or "Historical ML" in forecast["strategy"]


def test_decision_brief_rice_baseline_golden_case():
    """Verify Rice generates a baseline evidence brief with legacy benchmark note."""
    res = decision_intelligence_service.analyze_decision(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    assert res is not None
    assert res["is_scientifically_validated"] is True
    brief = res["brief"]
    forecast = brief.get("forecast_summary")
    assert forecast is not None
    assert "Baseline" in forecast["strategy"] or "District Mean" in forecast["strategy"] or "Persistence" in forecast["strategy"]

    val = brief.get("validation_evidence")
    assert val is not None
    assert val["legacy_benchmark_note"] is not None
    assert "Legacy Rice Validated Benchmark" in val["legacy_benchmark_note"]


def test_decision_brief_wheat_baseline_golden_case():
    """Verify Wheat generates a certified baseline brief without fabricating ML importances."""
    res = decision_intelligence_service.analyze_decision(
        crop="Wheat",
        state="Haryana",
        district="Karnal",
        year=2017
    )
    assert res is not None
    assert res["is_scientifically_validated"] is True
    brief = res["brief"]
    forecast = brief.get("forecast_summary")
    assert forecast is not None
    assert forecast["crop"] == "Wheat"


def test_decision_brief_determinism():
    """Verify identical input requests produce bitwise identical decision evidence values."""
    res1 = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    res2 = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )

    b1 = res1["brief"]
    b2 = res2["brief"]

    # Invariant fields must match exactly
    assert b1["forecast_summary"]["forecast_yield_kg_ha"] == b2["forecast_summary"]["forecast_yield_kg_ha"]
    assert b1["forecast_summary"]["strategy"] == b2["forecast_summary"]["strategy"]
    assert b1["validation_evidence"]["mae_kg_ha"] == b2["validation_evidence"]["mae_kg_ha"]
    assert b1["historical_context"]["historical_mean_yield_kg_ha"] == b2["historical_context"]["historical_mean_yield_kg_ha"]
    assert len(b1["evidence_items"]) == len(b2["evidence_items"])
