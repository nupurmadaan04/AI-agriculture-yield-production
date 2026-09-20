"""
Unit tests for feature timing audit and pre-season availability (Day 21).
Validates that features used for forecasting are verified as pre-season safe.
"""
import pytest
import pandas as pd
from pathlib import Path

METADATA_DIR = Path("Datasets/metadata")


def test_feature_timing_audit_csv():
    """Verify multicrop_feature_timing_audit.csv contains audited features and statuses."""
    path = METADATA_DIR / "multicrop_feature_timing_audit.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) >= 7
    expected_cols = [
        "feature", "observation_time", "available_before_forecast",
        "fold_safe", "timing_status", "timing_notes"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify lag and rolling mean features are SAFE
    safe_features = df[df["timing_status"] == "SAFE"]["feature"].tolist()
    assert "yield_lag_1" in safe_features
    assert "yield_rolling_3yr_mean" in safe_features
    assert "area_lag_1" in safe_features


def test_spatial_cluster_flagged_unsafe():
    """Verify spatial_cluster_id or global fit features are flagged UNSAFE if fit across folds."""
    path = METADATA_DIR / "multicrop_feature_timing_audit.csv"
    df = pd.read_csv(path)
    if "spatial_cluster_id" in df["feature"].values:
        status = df[df["feature"] == "spatial_cluster_id"]["timing_status"].values[0]
        assert status == "UNSAFE", "spatial_cluster_id fit globally should be marked UNSAFE"
