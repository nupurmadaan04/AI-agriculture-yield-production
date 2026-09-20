"""
Unit tests for Copilot monitoring intents & evidence responses (Day 12).
"""

import pytest
from backend.services.copilot_service import copilot_service
from backend.services.query_service import query_service


def test_copilot_monitoring_intent_classification():
    q1 = "Which regions are currently under elevated warning alerts?"
    parsed1 = query_service.classify_intent(q1)
    assert parsed1["intent"] == "alert_search"

    q2 = "What is the historical backtest precision and lead time of warnings?"
    parsed2 = query_service.classify_intent(q2)
    assert parsed2["intent"] == "warning_backtest"

    q3 = "What is the overall monitoring health and data freshness?"
    parsed3 = query_service.classify_intent(q3)
    assert parsed3["intent"] == "monitoring_health"

    q4 = "Run CUSUM change detection and trend break analysis for Punjab"
    parsed4 = query_service.classify_intent(q4)
    assert parsed4["intent"] == "change_detection"


def test_copilot_monitoring_queries_execution():
    res_alert = copilot_service.answer_query("Show active alerts under watch in Punjab")
    assert res_alert["intent"] == "alert_search"
    assert "Punjab" in res_alert["answer"]
    assert len(res_alert["evidence"]) > 0

    res_backtest = copilot_service.answer_query("Run historical warning backtest")
    assert res_backtest["intent"] == "warning_backtest"
    assert "Precision" in res_backtest["answer"]
    assert "Lead Time" in res_backtest["answer"]

    res_health = copilot_service.answer_query("Check monitoring health status")
    assert res_health["intent"] == "monitoring_health"
    assert "Health Index" in res_health["answer"]
