"""
Unit tests for Day 23 Final Validation Coordinator & Temporal Range Audits.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.final_validation import FinalValidationCoordinator


def test_temporal_range_statement():
    coordinator = FinalValidationCoordinator()
    panel_path = coordinator.base_dir / "Datasets" / "processed" / "agricultural_panel.csv"
    assert panel_path.exists()

    df = pd.read_csv(panel_path, usecols=["year"])
    max_year = int(df["year"].max())
    assert max_year == 2017


def test_final_validation_metadata_artifacts_exist():
    base_dir = Path(__file__).resolve().parent.parent
    meta_dir = base_dir / "Datasets" / "metadata"

    expected_files = [
        "final_strategy_results.csv",
        "final_validation_results.csv",
        "residual_diagnostics.csv",
        "residual_year_analysis.csv",
        "residual_district_analysis.csv",
        "prediction_bias_analysis.csv",
        "empirical_interval_analysis.csv",
        "reproducibility_audit.csv",
        "final_model_certification.csv",
    ]

    for f in expected_files:
        p = meta_dir / f
        assert p.exists(), f"Missing required Day 23 metadata file: {f}"
        df = pd.read_csv(p)
        assert not df.empty, f"File {f} is empty"
