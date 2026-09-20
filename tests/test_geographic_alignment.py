"""
Unit Tests for Geographic Alignment & Spatial Mapping (Day 22).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.exogenous.geographic_alignment import GeographicAlignmentEngine


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_geographic_audit_integrity(base_dir):
    engine = GeographicAlignmentEngine(base_dir)
    geo_audit_csv = base_dir / "Datasets" / "metadata" / "exogenous_geographic_audit.csv"
    assert geo_audit_csv.exists()

    df = pd.read_csv(geo_audit_csv)
    assert len(df) == 20  # 20 states evaluated
    assert "match_percentage" in df.columns
    assert "spatial_alignment_status" in df.columns

    # High match rate across states
    assert df["match_percentage"].mean() >= 95.0
