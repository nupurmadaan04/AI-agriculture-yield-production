"""
Unit tests for ChangeDetectionEngine (Day 12).
"""

import pytest
from src.change_detection import change_detection_engine, ChangeDetectionEngine


def test_cusum_shift_detection():
    # Stable baseline followed by strong drop
    stable = [2000.0] * 10
    drop = [1200.0] * 10
    series = stable + drop

    res = change_detection_engine.detect_cusum_shift(series, threshold=3.0)
    assert res["change_detected"] is True
    assert res["change_type"] == "DOWNWARD_REGIME_SHIFT"
    assert res["max_cusum_statistic"] >= 3.0


def test_trend_break_detection():
    years = list(range(2000, 2016))
    # Pre-break: steady increase; post-break: sharp decline
    values = [2000 + i * 50 for i in range(8)] + [2400 - i * 100 for i in range(8)]

    res = change_detection_engine.detect_trend_break(years, values)
    assert res["trend_break_detected"] is True
    assert res["inflection_year"] is not None
    assert res["slope_delta"] > 0


def test_analyze_series_changes():
    years = list(range(2000, 2015))
    values = [2500.0 + (i % 3) * 50 for i in range(15)]

    res = change_detection_engine.analyze_series_changes(years, values)
    assert "cusum_analysis" in res
    assert "trend_break_analysis" in res
    assert "volatility_shift_cv_delta" in res
    assert "scientific_disclaimer" in res
