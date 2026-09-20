"""
Unit Tests for Temporal Alignment & Pre-Season Contract Enforcement (Day 22).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.exogenous.temporal_alignment import TemporalAlignmentEngine, TEMPORAL_CONTRACTS


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_temporal_contracts_cutoff(base_dir):
    engine = TemporalAlignmentEngine(base_dir)
    p = engine.export_temporal_audit()
    assert p.exists()

    df = pd.read_csv(p)
    assert len(df) >= 7

    # Check that monsoon rainfall is flagged UNSAFE
    monsoon_row = df[df["feature"] == "monsoon_rainfall_june_sept_mm"]
    assert len(monsoon_row) > 0
    assert monsoon_row["timing_status"].iloc[0] == "UNSAFE"

    # Check that preseason rainfall is flagged SAFE
    pre_row = df[df["feature"] == "preseason_rainfall_mm"]
    assert len(pre_row) > 0
    assert pre_row["timing_status"].iloc[0] == "SAFE"


def test_filter_safe_features(base_dir):
    engine = TemporalAlignmentEngine(base_dir)
    dummy_df = pd.DataFrame({
        "preseason_rainfall_mm": [100.0, 120.0],
        "monsoon_rainfall_june_sept_mm": [800.0, 950.0],
        "harvest_ndvi_max": [0.85, 0.88]
    })
    filtered = engine.filter_safe_features(dummy_df)
    assert "preseason_rainfall_mm" in filtered.columns
    assert "monsoon_rainfall_june_sept_mm" not in filtered.columns
    assert "harvest_ndvi_max" not in filtered.columns
