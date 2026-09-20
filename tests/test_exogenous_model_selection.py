"""
Unit Tests for Exogenous Model Selection Decisions (Day 22).
"""

from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_model_selection_classes(base_dir):
    sel_csv = base_dir / "Datasets" / "metadata" / "exogenous_model_selection.csv"
    if sel_csv.exists():
        df = pd.read_csv(sel_csv)
        assert len(df) == 14
        valid_statuses = {"EXOGENOUS_ROBUST", "EXOGENOUS_CONDITIONAL", "NO_MEANINGFUL_GAIN", "INSUFFICIENT_COVERAGE"}
        assert set(df["day22_status"].unique()).issubset(valid_statuses)
