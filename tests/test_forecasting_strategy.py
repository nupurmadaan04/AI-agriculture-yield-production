"""
Unit tests for crop-specific forecasting policies, fallback architectures, and model registry lineage (Day 21).
"""
import pytest
import pandas as pd
import json
from pathlib import Path

METADATA_DIR = Path("Datasets/metadata")
REGISTRY_PATH = Path("Models/multicrop/model_registry.json")


def test_forecasting_strategy_csv():
    """Verify multicrop_forecasting_strategy.csv contains policy for all 14 crops."""
    path = METADATA_DIR / "multicrop_forecasting_strategy.csv"
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert len(df) == 14
    expected_cols = [
        "crop", "day21_status", "primary_forecasting_model", "fallback_model",
        "operating_conditions", "diagnostic_notes", "missing_information_gaps",
        "evidence_required", "evidence_strength_score", "evidence_interpretation"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify evidence strength score in [0, 100]
    assert (df["evidence_strength_score"] >= 0).all()
    assert (df["evidence_strength_score"] <= 100).all()


def test_model_registry_lineage_preserved():
    """Verify Models/multicrop/model_registry.json retains Day 19, Day 20, and Day 21 lineage."""
    assert REGISTRY_PATH.exists(), f"Missing {REGISTRY_PATH}"
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)

    models = registry.get("models", {})
    assert len(models) == 14

    for crop_name, model_info in models.items():
        assert "history" in model_info, f"Missing history array for {crop_name}"
        history = model_info["history"]
        phases = [h.get("phase") for h in history]
        assert "Multi-Crop Lag Modeling" in phases
        assert "Temporal Walk-Forward Validation" in phases
        assert "Error Diagnosis & Strategy Selection" in phases

        assert "day21_diagnosis" in model_info, f"Missing day21_diagnosis in {crop_name}"
        day21 = model_info["day21_diagnosis"]
        assert "day21_status" in day21
        assert "primary_forecasting_model" in day21
        assert "fallback_model" in day21
