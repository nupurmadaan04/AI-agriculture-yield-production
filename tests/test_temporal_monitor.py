"""
Unit tests for TemporalMonitor engine (Day 12).
"""

import pytest
import numpy as np
from src.temporal_monitor import temporal_monitor, TemporalMonitor


def test_temporal_monitor_basic():
    years = [2010, 2011, 2012, 2013, 2014, 2015]
    values = [2000.0, 2100.0, 2200.0, 2150.0, 2300.0, 2400.0]

    metrics = temporal_monitor.calculate_temporal_metrics(years, values, metric_name="yield")
    assert metrics["record_count"] == 6
    assert metrics["latest_year"] == 2015
    assert metrics["latest_value"] == 2400.0
    assert metrics["yoy_change_absolute"] == 100.0
    assert metrics["yoy_change_pct"] > 0
    assert metrics["rolling_3yr_mean"] > 0
    assert metrics["trend_slope"] > 0
    assert len(metrics["trajectory"]) == 6


def test_temporal_monitor_empty_and_single():
    empty_res = temporal_monitor.calculate_temporal_metrics([], [])
    assert empty_res["record_count"] == 0
    assert empty_res["latest_year"] is None

    single_res = temporal_monitor.calculate_temporal_metrics([2015], [3000.0])
    assert single_res["record_count"] == 1
    assert single_res["latest_value"] == 3000.0
    assert single_res["yoy_change_pct"] == 0.0


def test_temporal_monitor_chronological_ordering():
    years = [2015, 2010, 2012]
    values = [3000.0, 2000.0, 2500.0]

    res = temporal_monitor.calculate_temporal_metrics(years, values)
    assert res["latest_year"] == 2015
    assert res["latest_value"] == 3000.0
    assert res["trajectory"][0]["year"] == 2010
    assert res["trajectory"][-1]["year"] == 2015
