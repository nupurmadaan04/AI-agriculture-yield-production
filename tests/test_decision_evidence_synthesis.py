"""
Day 31 Unit Tests: Decision Evidence Synthesis & Temporal Boundary Audits.

Verifies:
- Strict temporal isolation (historical observations strictly < forecast_year)
- Unharvested future horizon handling (EVALUATION_UNAVAILABLE)
- Semantic tagging of evidence items
- Uncertainty empirical ensemble spread and mandatory disclaimer
- Rule-based evidence completeness
"""

import pytest
from backend.services.decision_intelligence_service import decision_intelligence_service


def test_temporal_boundary_isolation():
    """Historical context observations must strictly precede forecast year."""
    target_year = 2017
    res = decision_intelligence_service.analyze_decision(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        year=target_year
    )
    brief = res["brief"]
    hist = brief.get("historical_context")
    assert hist is not None
    assert hist["end_year"] < target_year
    for pt in hist.get("recent_observations", []):
        assert pt["year"] < target_year


def test_future_unharvested_horizon_handling():
    """Forecast year beyond ground truth (e.g. 2026) must set post_outcome status to EVALUATION_UNAVAILABLE."""
    future_year = 2026
    res = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=future_year
    )
    brief = res["brief"]
    mon = brief.get("monitoring_evidence")
    assert mon is not None
    assert mon["post_outcome_evaluation_status"] == "EVALUATION_UNAVAILABLE"


def test_evidence_semantic_classifications():
    """All generated evidence items must carry explicit valid semantic classifications."""
    valid_types = {
        "OBSERVED", "PREDICTED", "SIMULATED", "DERIVED",
        "MODEL_ATTRIBUTION", "VALIDATION", "MONITORING",
        "PROVENANCE", "DECISION_EVIDENCE", "HISTORICAL_REFERENCE",
        "ASSUMPTION", "LIMITATION"
    }
    res = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    brief = res["brief"]
    ev_items = brief.get("evidence_items", [])
    assert len(ev_items) >= 5
    for item in ev_items:
        assert item["evidence_type"] in valid_types
        assert item["value"] is not None
        assert "evidence_id" in item
        assert "source_module" in item


def test_uncertainty_disclaimer_presence():
    """Uncertainty evidence must include the required non-confidence interval disclaimer."""
    res = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    brief = res["brief"]
    unc = brief.get("uncertainty_evidence")
    assert unc is not None
    if unc["is_available"]:
        assert "not a formal confidence interval" in unc["disclaimer"].lower()
        assert unc["empirical_p10_kg_ha"] is not None
        assert unc["empirical_p90_kg_ha"] is not None


def test_rule_based_evidence_completeness():
    """Completeness level must be evaluated from explicit deterministic rules."""
    res = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    status = res["brief"]["evidence_status"]
    assert status["completeness_level"] in [
        "STRONG_EVIDENCE", "PARTIAL_EVIDENCE", "LIMITED_EVIDENCE", "INSUFFICIENT_EVIDENCE"
    ]
