"""
Unit Tests for Exogenous Ablation Benchmarks (Day 22).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.exogenous_ablation import ABLATION_EXPERIMENTS, EVALUATED_CROPS, FOLDS


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_ablation_experiment_definitions():
    assert len(ABLATION_EXPERIMENTS) == 5
    exp_ids = [e["experiment_id"] for e in ABLATION_EXPERIMENTS]
    assert exp_ids == ["EXP-22A", "EXP-22B", "EXP-22C", "EXP-22D", "EXP-22E"]


def test_ablation_results_structure(base_dir):
    ablation_csv = base_dir / "Datasets" / "metadata" / "exogenous_ablation_results.csv"
    if ablation_csv.exists():
        df = pd.read_csv(ablation_csv)
        assert len(df) > 0
        assert "crop" in df.columns
        assert "experiment_id" in df.columns
        assert "mean_mae" in df.columns
        assert "mean_improvement_vs_historical_pct" in df.columns
