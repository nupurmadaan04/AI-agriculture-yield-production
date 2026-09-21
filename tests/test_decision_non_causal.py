"""
Day 31 Unit Tests: Non-Causal Language Guard & Scientific Compliance Audits.

Verifies:
- Scans all generated decision text across 4 commodities for forbidden causal phrases
- Verifies scenario options are marked SIMULATED
- Checks assumptions and limitations are explicitly listed
"""

import pytest
from src.decision_validation import decision_validator
from backend.services.decision_intelligence_service import decision_intelligence_service


@pytest.mark.parametrize("crop,state,district", [
    ("Oilseeds", "Punjab", "Ludhiana"),
    ("Sugarcane", "Uttar Pradesh", "Meerut"),
    ("Rice", "Punjab", "Ludhiana"),
    ("Wheat", "Haryana", "Karnal"),
])
def test_no_causal_language_in_decision_brief(crop, state, district):
    """Generated decision brief text must contain zero unscientific causal claims."""
    res = decision_intelligence_service.analyze_decision(
        crop=crop,
        state=state,
        district=district,
        year=2017
    )
    brief = res["brief"]

    # Gather all analytical prose
    prose_blocks = [
        str(brief.get("executive_summary", {})),
        str(brief.get("footer_disclaimer", "")),
        " ".join([s.get("content", "") for s in brief.get("sections", [])]),
        " ".join([e.get("statement", "") for e in brief.get("evidence_items", [])]),
        " ".join([o.get("tradeoffs", "") for o in brief.get("decision_options", [])])
    ]
    full_text = " ".join(prose_blocks)

    violations = decision_validator.scan_for_causal_language(full_text)
    assert len(violations) == 0, f"Detected causal violations in {crop}: {violations}"


def test_scenario_options_marked_simulated():
    """All scenario options must be strictly marked as simulated."""
    res = decision_intelligence_service.analyze_decision(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    brief = res["brief"]
    options = brief.get("decision_options", [])
    assert len(options) > 0
    for opt in options:
        assert opt.get("is_simulated") is True
        assert opt.get("semantic_classification") == "DERIVED"


def test_assumptions_and_limitations_present():
    """Decision brief must explicitly enumerate assumptions and limitations."""
    res = decision_intelligence_service.analyze_decision(
        crop="Oilseeds",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    brief = res["brief"]
    assert len(brief.get("assumptions", [])) >= 2
    assert len(brief.get("limitations", [])) >= 2
