"""
Unit tests for Day 23 Residual Diagnostics, Yield Quantiles & Systematic Bias Detection.
"""

from pathlib import Path
import pytest
import pandas as pd


def test_residual_diagnostics_structure():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "Datasets" / "metadata" / "residual_diagnostics.csv"
    assert p.exists()

    df = pd.read_csv(p)
    assert len(df) == 14
    for col in [
        "mean_residual", "median_residual", "std_residual", "mae", "rmse",
        "p25_abs_error", "p50_abs_error", "p75_abs_error", "p90_abs_error",
        "p95_abs_error", "mae_q1_lowest_yield", "mae_q4_highest_yield"
    ]:
        assert col in df.columns


def test_prediction_bias_classification():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "Datasets" / "metadata" / "prediction_bias_analysis.csv"
    assert p.exists()

    df = pd.read_csv(p)
    assert len(df) == 14
    valid_statuses = {"OVER_PREDICTION_BIAS", "UNDER_PREDICTION_BIAS", "NO_CLEAR_BIAS", "INSUFFICIENT_EVIDENCE"}
    for status in df["bias_status"]:
        assert status in valid_statuses


def test_empirical_intervals_labeling():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "Datasets" / "metadata" / "empirical_interval_analysis.csv"
    assert p.exists()

    df = pd.read_csv(p)
    assert not df.empty
    for label in df["calibration_label"]:
        assert "Empirical ensemble interval, not a statistically calibrated prediction interval." in label
