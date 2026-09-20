"""
Unit tests for Day 23 Strategy Evaluation Engine (Policy vs ML vs Baseline).
"""

from pathlib import Path
import pytest
import pandas as pd

from src.strategy_evaluation import StrategyEvaluationEngine, OPERATIONAL_POLICIES
from src.exogenous_ablation import EVALUATED_CROPS


def test_operational_policies_coverage():
    assert len(OPERATIONAL_POLICIES) == len(EVALUATED_CROPS)
    for crop in EVALUATED_CROPS:
        assert crop in OPERATIONAL_POLICIES
        pol = OPERATIONAL_POLICIES[crop]
        assert "primary_model" in pol
        assert "fallback_model" in pol
        assert "policy_type" in pol
        assert "operating_rule" in pol


def test_strategy_results_metrics():
    base_dir = Path(__file__).resolve().parent.parent
    strat_csv = base_dir / "Datasets" / "metadata" / "final_strategy_results.csv"
    assert strat_csv.exists()

    df = pd.read_csv(strat_csv)
    assert len(df) == 14
    for _, row in df.iterrows():
        assert row["strategy_mean_mae"] > 0
        assert row["baseline_mean_mae"] > 0
        assert row["ml_mean_mae"] > 0
