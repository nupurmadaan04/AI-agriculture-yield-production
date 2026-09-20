"""
Unit Tests for Agronomic Exogenous Feature Engineering (Day 22).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.exogenous.feature_engineering import ExogenousFeatureEngineeringEngine


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_feature_registry_schema(base_dir):
    engine = ExogenousFeatureEngineeringEngine(base_dir)
    p = engine.export_feature_registry()
    assert p.exists()

    df = pd.read_csv(p)
    required_cols = [
        "feature", "source", "spatial_level", "temporal_resolution",
        "observation_period", "availability_date", "forecast_origin",
        "lag", "unit", "transformation", "leakage_status"
    ]
    for c in required_cols:
        assert c in df.columns

    # All registered features must be SAFE
    assert (df["leakage_status"] == "SAFE").all()


def test_feature_transformations(base_dir):
    engine = ExogenousFeatureEngineeringEngine(base_dir)
    dummy_df = pd.DataFrame({
        "preseason_rainfall_mm": [120.0, 180.0],
        "preseason_rainfall_anomaly_pct": [5.0, -10.0],
        "preseason_temp_mean_c": [30.0, 32.0],
        "preseason_temp_max_c": [39.0, 41.0],
        "preseason_temp_anomaly_c": [0.5, 1.2],
        "preseason_soil_moisture_index": [0.45, 0.35],
        "preseason_dry_spell_days": [20, 35],
        "rainfall_lag1_annual_mm": [900.0, 750.0],
        "irrigation_ratio_lag1": [0.40, 0.45]
    })
    feat_df = engine.engineer_exogenous_features(dummy_df)
    assert "preseason_aridity_index" in feat_df.columns
    assert "preseason_rainfall_total" in feat_df.columns
    assert "preseason_soil_moisture" in feat_df.columns
