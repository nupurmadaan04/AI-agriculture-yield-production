"""
Unit tests for deterministic model selection logic (Day 21).
Validates that classification follows deterministic empirical rules and preserves lineage.
"""
import pytest
import pandas as pd
from pathlib import Path

METADATA_DIR = Path("Datasets/metadata")


def test_model_selection_csv_structure():
    """Verify multicrop_model_selection.csv contains all 14 crops with 3-day lineage."""
    path = METADATA_DIR / "multicrop_model_selection.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 14
    expected_cols = [
        "crop", "day19_status", "day20_status", "day21_status",
        "win_rate", "mean_mae_improvement_pct", "median_mae_improvement_pct",
        "worst_fold_degradation_pct", "ml_mae_cv", "feature_timing_status",
        "total_test_observations", "decision_basis", "methodology_version"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_model_selection_categories():
    """Verify Day 21 status classifications match empirical thresholds."""
    path = METADATA_DIR / "multicrop_model_selection.csv"
    df = pd.read_csv(path)
    statuses = set(df["day21_status"].unique())
    valid_statuses = {"ROBUST_ML", "ML_WITH_CONDITIONS", "BASELINE_PREFERRED", "RESEARCH_CANDIDATE", "INSUFFICIENT_EVIDENCE"}
    assert statuses.issubset(valid_statuses)

    # Check Oilseeds is ROBUST_ML
    oilseeds_row = df[df["crop"] == "Oilseeds"].iloc[0]
    assert oilseeds_row["day21_status"] == "ROBUST_ML"
    assert oilseeds_row["win_rate"] >= 75.0
    assert oilseeds_row["mean_mae_improvement_pct"] > 0.0

    # Check Chickpea is ML_WITH_CONDITIONS
    chickpea_row = df[df["crop"] == "Chickpea"].iloc[0]
    assert chickpea_row["day21_status"] == "ML_WITH_CONDITIONS"

    # Check Baseline preferred count is 7
    bp_count = (df["day21_status"] == "BASELINE_PREFERRED").sum()
    assert bp_count == 7
