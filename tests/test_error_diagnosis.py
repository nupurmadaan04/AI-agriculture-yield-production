"""
Unit tests for multi-level error diagnosis and regime decomposition (Day 21).
Validates multi-origin error statistics, quantiles, error bands, and district/year regimes.
"""
import pytest
import pandas as pd
from pathlib import Path

METADATA_DIR = Path("Datasets/metadata")


def test_error_diagnosis_csv_exists_and_complete():
    """Verify multicrop_error_diagnosis.csv contains all 14 evaluated crops."""
    path = METADATA_DIR / "multicrop_error_diagnosis.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 14
    expected_cols = [
        "crop", "selected_ml_model", "total_folds", "total_test_observations",
        "historical_median_yield", "ml_wins", "ml_losses", "win_rate",
        "mean_mae_improvement_pct", "median_mae_improvement_pct",
        "worst_fold_degradation_pct", "best_fold_improvement_pct",
        "ml_mae_mean", "ml_mae_median", "ml_mae_std", "ml_mae_cv",
        "base_mae_mean", "base_mae_median", "base_mae_std", "base_mae_cv",
        "p25_error_ml", "p50_error_ml", "p75_error_ml", "p90_error_ml", "p95_error_ml",
        "pct_errors_lt_100", "pct_errors_lt_250", "pct_errors_lt_500", "pct_errors_gt_1000",
        "normalized_error_p50", "normalized_error_p90"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in multicrop_error_diagnosis.csv"


def test_error_regimes_breakdown():
    """Verify multicrop_error_regimes.csv has 42 regime breakdown records (3 per crop)."""
    path = METADATA_DIR / "multicrop_error_regimes.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 42
    assert set(df["regime"].unique()) == {"Low Yield", "Normal Yield", "High Yield"}
    assert set(df["regime_status"].unique()).issubset({"ML_ADVANTAGE", "BASELINE_ADVANTAGE"})


def test_year_error_analysis():
    """Verify multicrop_year_error_analysis.csv covers all 4 walk-forward test origins."""
    path = METADATA_DIR / "multicrop_year_error_analysis.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 56  # 14 crops * 4 years
    assert set(df["year"].unique()) == {2014, 2015, 2016, 2017}
    # Check 2016 flag exists
    y2016_regimes = df[df["year"] == 2016]["temporal_regime"].tolist()
    assert any("2016" in str(r) for r in y2016_regimes)


def test_district_error_analysis():
    """Verify multicrop_district_error_analysis.csv contains district-level records with N >= 3."""
    path = METADATA_DIR / "multicrop_district_error_analysis.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) > 1000
    assert (df["observations_count"] >= 3).all(), "Every district record must satisfy N >= 3"
    assert "is_best_ml_district" in df.columns
    assert "is_worst_ml_district" in df.columns
    assert "is_high_error_district" in df.columns
