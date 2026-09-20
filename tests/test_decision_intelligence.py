"""
Unit tests for src/decision_intelligence.py and backend/services/decision_intelligence_service.py.
"""

import pytest
from src.decision_intelligence import decision_intelligence_engine
from backend.services.decision_intelligence_service import decision_intelligence_service


def test_collect_evidence_structure():
    res = decision_intelligence_engine.collect_evidence(
        crop="Rice",
        state="Punjab",
        year=2017
    )
    assert isinstance(res, dict)
    assert "context" in res
    assert "evidence_items" in res
    assert "metrics" in res
    assert len(res["evidence_items"]) >= 5


def test_analyze_decision_end_to_end():
    res = decision_intelligence_service.analyze_decision(
        crop="Rice",
        state="Punjab",
        district="Ludhiana",
        year=2017
    )
    assert isinstance(res, dict)
    assert "decision_id" in res
    assert res["decision_id"].startswith("DEC-")
    assert "brief" in res
    assert res["is_scientifically_validated"] is True
    assert res["validation_checks_passed"] == 11


def test_decision_history_and_retrieve():
    res = decision_intelligence_service.analyze_decision(state="Haryana")
    dec_id = res["decision_id"]
    retrieved = decision_intelligence_service.get_decision_by_id(dec_id)
    assert retrieved is not None
    assert retrieved["decision_id"] == dec_id
