"""
Unit Tests for Exogenous Data Coverage Audit (Day 22).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.exogenous.coverage_audit import ExogenousCoverageAuditEngine


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_coverage_audit_metrics(base_dir):
    cov_csv = base_dir / "Datasets" / "metadata" / "exogenous_coverage_audit.csv"
    assert cov_csv.exists()

    df = pd.read_csv(cov_csv)
    assert len(df) == 14  # 14 evaluated crops

    assert "crop" in df.columns
    assert "weather_coverage_pct" in df.columns
    assert "soil_coverage_pct" in df.columns
    assert "overall_exogenous_coverage_pct" in df.columns
    assert "coverage_status" in df.columns

    # High coverage across all crops
    assert df["overall_exogenous_coverage_pct"].min() >= 90.0
