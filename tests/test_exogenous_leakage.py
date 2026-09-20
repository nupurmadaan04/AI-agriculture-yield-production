"""
Unit Tests for Exogenous Anti-Leakage Certifications (Day 22).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.exogenous.leakage_audit import ExogenousLeakageAuditEngine, LEAKAGE_AUDIT_RULES


@pytest.fixture
def base_dir():
    return Path(__file__).resolve().parent.parent


def test_leakage_audit_rules_passed(base_dir):
    engine = ExogenousLeakageAuditEngine(base_dir)
    p = engine.export_leakage_audit()
    assert p.exists()

    df = pd.read_csv(p)
    assert len(df) >= 6
    assert (df["audit_result"] == "PASS").all()
    assert (df["status"] == "SAFE").all()


def test_no_forbidden_keywords_in_features(base_dir):
    feat_reg_csv = base_dir / "Datasets" / "metadata" / "exogenous_feature_registry.csv"
    assert feat_reg_csv.exists()

    df = pd.read_csv(feat_reg_csv)
    features = df["feature"].tolist()

    for f in features:
        assert "monsoon" not in f.lower() or "preseason" in f.lower()
        assert "harvest" not in f.lower()
        assert "production" not in f.lower()
