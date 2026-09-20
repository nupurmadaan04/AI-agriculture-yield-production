"""
Unit tests for src/decision_audit.py.
"""

import pytest
from src.decision_audit import decision_audit_logger


def test_decision_audit_deterministic_hash():
    context = {"crop": "Rice", "state": "Punjab", "district": "Ludhiana", "year": 2017}
    dataset_ver = "ICRISAT 1966-2017"
    model_ver = "exogenous_rf_forecaster v2.1.0"
    ev_ids = ["EV-01", "EV-02"]
    sc_ids = ["SCEN-01"]
    exp_ids = ["EXP-01"]

    id1 = decision_audit_logger.generate_decision_id(
        context, dataset_ver, model_ver, ev_ids, sc_ids, exp_ids
    )
    id2 = decision_audit_logger.generate_decision_id(
        context, dataset_ver, model_ver, ev_ids, sc_ids, exp_ids
    )
    assert id1 == id2
    assert id1.startswith("DEC-")


def test_create_and_fetch_audit_record():
    context = {"state": "Haryana", "year": 2017}
    rec = decision_audit_logger.create_audit_record(
        context=context,
        dataset_version="ICRISAT 1966-2017",
        model_version="v2.1.0",
        evidence_ids=["EV-01"],
        scenario_ids=["SCEN-01"],
        explanation_ids=["EXP-01"],
        brief_summary={"test": True},
        limitations=["Non-causal"]
    )
    assert rec["decision_id"].startswith("DEC-")
    fetched = decision_audit_logger.get_audit_record(rec["decision_id"])
    assert fetched is not None
    assert fetched["decision_id"] == rec["decision_id"]
