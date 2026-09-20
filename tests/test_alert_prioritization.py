"""
Unit tests for AlertPrioritization (Day 12).
"""

import pytest
from src.alert_prioritization import alert_prioritization
from src.alert_severity import SeverityTier


def test_alert_prioritization_ranking():
    alerts = [
        {
            "alert_id": "ALR-01",
            "state": "Punjab",
            "district": "Ludhiana",
            "severity": SeverityTier.WATCH.value,
            "composite_risk_score": 25.0,
            "signal_count": 1,
            "year": 2017
        },
        {
            "alert_id": "ALR-02",
            "state": "Haryana",
            "district": "Karnal",
            "severity": SeverityTier.CRITICAL.value,
            "composite_risk_score": 85.0,
            "signal_count": 3,
            "year": 2017
        },
        {
            "alert_id": "ALR-03",
            "state": "Punjab",
            "district": "Patiala",
            "severity": SeverityTier.ELEVATED.value,
            "composite_risk_score": 50.0,
            "signal_count": 2,
            "year": 2017
        }
    ]

    ranked = alert_prioritization.rank_alerts(alerts)
    assert len(ranked) == 3
    assert ranked[0]["alert_id"] == "ALR-02"
    assert ranked[0]["priority_rank"] == 1
    assert ranked[1]["alert_id"] == "ALR-03"
    assert ranked[2]["alert_id"] == "ALR-01"


def test_alert_prioritization_filtering():
    alerts = [
        {"alert_id": "A1", "state": "Punjab", "district": "Ludhiana", "severity": "HIGH", "composite_risk_score": 60.0, "signal_count": 2, "year": 2017, "dominant_signal": "Decline"},
        {"alert_id": "A2", "state": "Haryana", "district": "Karnal", "severity": "WATCH", "composite_risk_score": 20.0, "signal_count": 1, "year": 2017, "dominant_signal": "Variance"},
    ]

    res_punjab = alert_prioritization.rank_alerts(alerts, state_filter="Punjab")
    assert len(res_punjab) == 1
    assert res_punjab[0]["state"] == "Punjab"

    res_sev = alert_prioritization.rank_alerts(alerts, severity_filter="HIGH")
    assert len(res_sev) == 1
    assert res_sev[0]["severity"] == "HIGH"
