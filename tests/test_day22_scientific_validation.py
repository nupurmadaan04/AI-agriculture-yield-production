"""
Comprehensive Scientific & Statistical Integrity Validation Tests (Day 22).
"""

from pathlib import Path
import json
import pandas as pd
import pytest


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_rice_benchmark_preserved(base_dir):
    """Verifies that the Day 9 Rice model remains intact."""
    metrics_path = base_dir / "Models" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        assert metrics.get("mae") == 353.01 or abs(metrics.get("mae", 0) - 353.01) < 0.1
        assert metrics.get("r2") == 0.7866 or abs(metrics.get("r2", 0) - 0.7866) < 0.01


def test_exogenous_pipeline_artifacts_exist(base_dir):
    metadata_dir = base_dir / "Datasets" / "metadata"
    processed_dir = base_dir / "Datasets" / "processed"

    assert (processed_dir / "exogenous_features.csv").exists()
    assert (metadata_dir / "exogenous_source_registry.csv").exists()
    assert (metadata_dir / "exogenous_feature_registry.csv").exists()
    assert (metadata_dir / "exogenous_temporal_audit.csv").exists()
    assert (metadata_dir / "exogenous_geographic_audit.csv").exists()
    assert (metadata_dir / "exogenous_coverage_audit.csv").exists()
    assert (metadata_dir / "exogenous_leakage_audit.csv").exists()
