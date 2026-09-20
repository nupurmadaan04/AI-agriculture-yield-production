"""
Unit tests for evidence normalization and typing.
"""

import pytest
from src.decision_intelligence import decision_intelligence_engine


def test_evidence_classification_types():
    res = decision_intelligence_engine.collect_evidence(state="Punjab")
    ev_items = res["evidence_items"]

    types = set(e["evidence_type"] for e in ev_items)
    assert "OBSERVED" in types
    assert "PREDICTED" in types
    assert "SIMULATED" in types
    assert "DERIVED" in types
    assert "VALIDATION" in types

    for e in ev_items:
        assert "evidence_id" in e
        assert e["evidence_id"].startswith("EV-")
        assert "statement" in e
        assert "source_module" in e
        assert "value" in e
