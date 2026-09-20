"""
Unit tests for Day 23 Final Model Certification & Status Taxonomy.
"""

from pathlib import Path
import json
import pytest
import pandas as pd


def test_final_model_certification_taxonomy():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "Datasets" / "metadata" / "final_model_certification.csv"
    assert p.exists()

    df = pd.read_csv(p)
    assert len(df) == 14

    valid_statuses = {
        "PRODUCTION_READY",
        "CONDITIONAL_PRODUCTION",
        "BASELINE_PRODUCTION",
        "RESEARCH_ONLY",
        "NOT_READY",
    }

    for status in df["final_status"]:
        assert status in valid_statuses

    oilseeds = df[df["crop"] == "Oilseeds"].iloc[0]
    assert oilseeds["final_status"] == "PRODUCTION_READY"

    sugarcane = df[df["crop"] == "Sugarcane"].iloc[0]
    assert sugarcane["final_status"] == "CONDITIONAL_PRODUCTION"

    chickpea = df[df["crop"] == "Chickpea"].iloc[0]
    assert chickpea["final_status"] == "BASELINE_PRODUCTION"


def test_model_registry_lineage_preserved():
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "Models" / "multicrop" / "model_registry.json"
    assert p.exists()

    with open(p, "r", encoding="utf-8") as f:
        reg = json.load(f)

    assert "crops" in reg
    assert "Oilseeds" in reg["crops"]
    assert "day23" in reg["crops"]["Oilseeds"]
    assert reg["crops"]["Oilseeds"]["day23"]["final_status"] == "PRODUCTION_READY"
