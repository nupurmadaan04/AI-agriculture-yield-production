import pytest
from src.early_warning_engine import early_warning_engine
from backend.services.early_warning_service import early_warning_service

def test_early_warning_engine_calculation():
    score, sev, triggers, comps = early_warning_engine.calculate_score(
        trend_direction="STRONG DECREASING",
        trend_slope=-55.0,
        forecast_change_pct=-18.0,
        historical_z_score=-3.2,
        is_anomaly=True,
        anomaly_score=85.0,
        prediction_spread_pct=35.0
    )
    assert 75.0 <= score <= 100.0
    assert sev == "CRITICAL"
    assert len(triggers) >= 3
    assert 'trend_signal_score' in comps

def test_early_warning_engine_low_risk():
    score, sev, triggers, comps = early_warning_engine.calculate_score(
        trend_direction="INCREASING",
        trend_slope=35.0,
        forecast_change_pct=2.5,
        historical_z_score=0.2,
        is_anomaly=False,
        anomaly_score=10.0,
        prediction_spread_pct=12.0
    )
    assert score < 25.0
    assert sev == "LOW"

def test_early_warning_service_assess():
    res = early_warning_service.assess_region(state="Punjab")
    assert res['state'] == "Punjab"
    assert 'warning_score' in res
    assert res['severity'] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert len(res['trigger_signals']) >= 1
