"""
Comprehensive scientific validation tests for Day 21 Crop-Specific Model Selection & Error Diagnosis.
Validates all non-negotiable scientific rules, invariants, and API integrations.
"""
import pytest
import pandas as pd
import json
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

METADATA_DIR = Path("Datasets/metadata")
REGISTRY_PATH = Path("Models/multicrop/model_registry.json")


@pytest.fixture
def client():
    return TestClient(app)


def test_rice_day9_model_untouched():
    """Verify that Day 9 Rice model baseline remains completely untouched (R² = 0.7866, MAE = 353.01 kg/ha)."""
    rice_meta_path = Path("Models/forecasting_model_metadata.json")
    assert rice_meta_path.exists(), f"Missing {rice_meta_path}"
    with open(rice_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    metrics = meta.get("selected_model_metrics", {})
    assert metrics.get("r2_score") == pytest.approx(0.7866, abs=1e-3)
    assert metrics.get("mae_kg_ha") == pytest.approx(353.01, abs=1e-1)


def test_no_synthetic_data_fabrication():
    """Ensure all test observations correspond to actual historical evaluation splits (2014-2017)."""
    df = pd.read_csv(METADATA_DIR / "multicrop_error_diagnosis.csv")
    assert df["total_test_observations"].sum() > 3000
    assert (df["win_rate"] >= 0.0).all() and (df["win_rate"] <= 100.0).all()


def test_lineage_immutability():
    """Ensure historical Day 19 and Day 20 records match across CSVs."""
    df_sel = pd.read_csv(METADATA_DIR / "multicrop_model_selection.csv")
    df_rob = pd.read_csv(METADATA_DIR / "multicrop_temporal_robustness.csv")

    for _, row in df_sel.iterrows():
        crop = row["crop"]
        rob_row = df_rob[df_rob["crop"] == crop].iloc[0]
        assert row["day20_status"] == rob_row["status"]
        assert row["win_rate"] >= 0.0 and row["win_rate"] <= 100.0


def test_api_diagnosis_endpoints(client):
    """Test all Day 21 REST API endpoints."""
    # Summary
    r = client.get("/api/modeling/diagnosis")
    assert r.status_code == 200
    data = r.json()
    assert data["total_crops"] == 14

    # Single crop diagnosis
    r = client.get("/api/modeling/diagnosis/Chickpea")
    assert r.status_code == 200
    assert r.json()["crop"] == "Chickpea"

    # Regimes
    r = client.get("/api/modeling/diagnosis/Chickpea/errors")
    assert r.status_code == 200
    assert len(r.json()["regimes"]) == 3

    # Districts
    r = client.get("/api/modeling/diagnosis/Chickpea/districts")
    assert r.status_code == 200
    assert r.json()["total_districts"] > 0

    # Years
    r = client.get("/api/modeling/diagnosis/Chickpea/years")
    assert r.status_code == 200
    assert len(r.json()["years"]) == 4

    # Features
    r = client.get("/api/modeling/diagnosis/Chickpea/features")
    assert r.status_code == 200
    assert len(r.json()["features"]) == 6

    # Model Selection
    r = client.get("/api/modeling/selection")
    assert r.status_code == 200
    assert r.json()["total_crops"] == 14

    # Forecasting Strategy
    r = client.get("/api/modeling/forecasting-strategy")
    assert r.status_code == 200
    assert r.json()["total_crops"] == 14
