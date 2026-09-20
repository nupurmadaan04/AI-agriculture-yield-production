"""
Unit tests for src/decision_validation.py.
"""

import pytest
from src.decision_validation import decision_validator


def test_causal_language_guard():
    bad_text = "This intervention will increase yield and was caused by fertilizer."
    violations = decision_validator.scan_for_causal_language(bad_text)
    assert len(violations) >= 2

    good_text = "The model estimates an increase in yield associated with historical baselines."
    good_violations = decision_validator.scan_for_causal_language(good_text)
    assert len(good_violations) == 0


def test_decision_validation_success():
    brief = {
        "executive_summary": "The model estimates stable output.",
        "decision_options": []
    }
    ev_items = [
        {"evidence_id": f"EV-0{i}", "value": 2000, "category": "reliability" if i == 0 else "forecast", "evidence_type": "PREDICTED" if i % 2 == 0 else "SIMULATED"}
        for i in range(6)
    ]
    provenance = {"nodes": [1, 2], "edges": [1]}
    audit = {
        "decision_id": "DEC-1234567890",
        "model_version": "exogenous_rf_forecaster v2.1.0",
        "dataset_version": "ICRISAT 1966-2017"
    }

    res = decision_validator.validate_decision_brief(brief, ev_items, provenance, audit)
    assert res["is_valid"] is True
    assert res["passed_rules"] == 11
