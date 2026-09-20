"""
Unit tests for feature stability across walk-forward folds (Day 21).
Validates feature predictive contributions, ranks, rank variance, and Feature Stability Score.
"""
import pytest
import pandas as pd
from pathlib import Path

METADATA_DIR = Path("Datasets/metadata")


def test_feature_stability_csv_structure():
    """Verify multicrop_feature_stability.csv has valid columns and stability scores in [0, 1]."""
    path = METADATA_DIR / "multicrop_feature_stability.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 84  # 14 crops * 6 features
    expected_cols = [
        "crop", "feature", "model", "mean_importance", "std_importance",
        "mean_rank", "rank_variance", "feature_stability_score", "interpretative_role"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify stability scores are bounded in [0, 1]
    assert (df["feature_stability_score"] >= 0.0).all()
    assert (df["feature_stability_score"] <= 1.0).all()


def test_feature_importance_language():
    """Ensure interpretive roles avoid causal claims."""
    path = METADATA_DIR / "multicrop_feature_stability.csv"
    df = pd.read_csv(path)
    for role in df["interpretative_role"]:
        role_lower = str(role).lower()
        assert "cause" not in role_lower or "predictive contribution" in role_lower, (
            f"Feature role '{role}' should avoid causal language."
        )


def test_primary_feature_presence():
    """Verify standard autoregressive features are present for each crop."""
    path = METADATA_DIR / "multicrop_feature_stability.csv"
    df = pd.read_csv(path)
    crops = df["crop"].unique()
    for c in crops:
        crop_features = df[df["crop"] == c]["feature"].tolist()
        assert "yield_lag_1" in crop_features
        assert "yield_rolling_3yr_mean" in crop_features
