"""
Unit tests for Day 12 backend services.
"""

import pytest
from backend.services.temporal_monitoring_service import temporal_monitoring_service
from backend.services.alert_service import alert_service
from backend.services.change_detection_service import change_detection_service
from backend.services.monitoring_health_service import monitoring_health_service
from backend.services.warning_backtest_service import warning_backtest_service


def test_temporal_monitoring_service():
    res = temporal_monitoring_service.get_timeline_metrics(state="Punjab", district="Ludhiana", metric="yield")
    assert res["record_count"] > 0
    assert res["state"] == "Punjab"
    assert res["district"] == "Ludhiana"
    assert "rolling_3yr_mean" in res


def test_alert_service_overview_and_ranking():
    overview = alert_service.get_overview()
    assert overview["active_alerts_count"] >= 0
    assert overview["model_monitoring_status"] == "HEALTHY"
    assert "summary" in overview

    alerts = alert_service.get_ranked_alerts(limit=10)
    assert len(alerts) > 0
    first = alerts[0]
    assert "alert_id" in first
    assert "severity" in first
    assert "evidence_chain" in first

    by_id = alert_service.get_alert_by_id(first["alert_id"])
    assert by_id is not None
    assert by_id["alert_id"] == first["alert_id"]


def test_alert_service_warning_map():
    map_data = alert_service.get_warning_map_data()
    assert len(map_data) == 20
    assert all("state" in item and "severity" in item for item in map_data)


def test_change_detection_service():
    res = change_detection_service.analyze_region_change(state="Punjab")
    assert "cusum_analysis" in res
    assert "trend_break_analysis" in res


def test_monitoring_health_service():
    health = monitoring_health_service.get_monitoring_health()
    assert health["status"] in ["HEALTHY", "WATCH", "DEGRADED", "REVIEW_REQUIRED"]
    assert health["overall_health_score"] > 50.0
    assert "data_freshness_label" in health


def test_warning_backtest_service():
    res = warning_backtest_service.run_backtest(lead_time_years=1)
    assert res["total_evaluations"] > 0
    assert "precision" in res
