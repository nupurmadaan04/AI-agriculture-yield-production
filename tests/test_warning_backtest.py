"""
Unit tests for WarningBacktester (Day 12).
"""

import pytest
import pandas as pd
from src.warning_backtest import warning_backtester, WarningBacktester
from backend.utils.data_loader import data_loader


def test_warning_backtest_with_panel_data():
    df = data_loader.dataframe
    res = warning_backtester.run_backtest(
        df=df,
        yield_drop_threshold_pct=-10.0,
        warning_zscore_threshold=1.2,
        lead_time_years=1
    )

    assert res["total_evaluations"] > 1000
    assert res["is_chronologically_valid"] is True
    assert 0.0 <= res["precision"] <= 100.0
    assert 0.0 <= res["recall"] <= 100.0
    assert 0.0 <= res["f1_score"] <= 100.0
    assert 0.0 <= res["false_positive_rate"] <= 100.0
    assert res["mean_lead_time_years"] == 1
    assert "scientific_disclaimer" in res


def test_warning_backtest_empty():
    empty_df = pd.DataFrame()
    res = warning_backtester.run_backtest(empty_df)
    assert res["total_evaluations"] == 0
    assert res["precision"] == 0.0
