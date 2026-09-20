"""
Unit tests for Day 23 Dual-Run Bitwise Reproducibility Verification.
"""

from pathlib import Path
import json
import pytest
import pandas as pd


def test_reproducibility_audit_bitwise_match():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "Datasets" / "metadata" / "reproducibility_audit.csv"
    assert p.exists()

    df = pd.read_csv(p)
    assert len(df) == 14
    for _, row in df.iterrows():
        assert row["bitwise_reproducible"] is True or row["bitwise_reproducible"] == "True"
        assert row["status"] == "VERIFIED_BITWISE"
        assert row["max_absolute_prediction_diff"] == 0.0


def test_reproducibility_certificate_json():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "Models" / "multicrop" / "reproducibility_certificate.json"
    assert p.exists()

    with open(p, "r", encoding="utf-8") as f:
        cert = json.load(f)

    assert cert["overall_status"] == "ALL_14_CROPS_REPRODUCIBLE"
    assert len(cert["crop_certificates"]) == 14
