"""
Tests for Day 19: Multi-Crop Forecasting, Baselines Comparison & Artifact Registry
==================================================================================
Verifies zero-leakage feature engineering, chronological splits, model acceptance rules,
uncertainty dispersion, Rice model regression safety, and backend REST APIs.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from src.multicrop_pipeline import CropFeaturePipeline
from src.train_multicrop_models import get_model_ready_crops

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_MULTICROP_DIR = os.path.join(BASE_DIR, "Models", "multicrop")
METADATA_DIR = os.path.join(BASE_DIR, "Datasets", "metadata")
PROCESSED_DIR = os.path.join(BASE_DIR, "Datasets", "processed")

client = TestClient(app)


def test_dynamic_crop_loading():
    """Verifies that the 14 MODEL_READY crops are loaded dynamically from metadata."""
    crops = get_model_ready_crops()
    assert len(crops) == 14
    assert "Maize" in crops
    assert "Rice" in crops
    assert "Wheat" in crops
    assert "Sugarcane" in crops
    assert "Sesamum" in crops


def test_feature_pipeline_anti_leakage_and_shifts():
    """Verifies that lag and rolling features are shifted and zero same-period lookahead occurs."""
    sample_df = pd.DataFrame([
        {"state": "Bihar", "district": "Patna", "year": 2011, "yield_kg_ha": 2000.0, "area_ha": 10000.0},
        {"state": "Bihar", "district": "Patna", "year": 2012, "yield_kg_ha": 2500.0, "area_ha": 11000.0},
        {"state": "Bihar", "district": "Patna", "year": 2013, "yield_kg_ha": 3000.0, "area_ha": 12000.0},
    ])

    pipe = CropFeaturePipeline()
    pipe.fit(sample_df)
    transformed = pipe.transform(sample_df)

    # In 2012, yield_lag_1 must equal 2011 yield (2000.0), NOT 2012 yield (2500.0)
    row_2012 = transformed[transformed["year"] == 2012].iloc[0]
    assert row_2012["yield_lag_1"] == 2000.0

    # In 2013, yield_lag_1 must equal 2012 yield (2500.0), NOT 2013 yield (3000.0)
    row_2013 = transformed[transformed["year"] == 2013].iloc[0]
    assert row_2013["yield_lag_1"] == 2500.0

    # No production column allowed in features
    assert "production_tonnes" not in CropFeaturePipeline.FEATURE_NAMES
    assert "current_year_production" not in CropFeaturePipeline.FEATURE_NAMES


def test_model_results_and_leaderboard_integrity():
    """Verifies that model results CSV and leaderboard exist with consistent classifications."""
    results_path = os.path.join(METADATA_DIR, "multicrop_model_results.csv")
    leaderboard_path = os.path.join(METADATA_DIR, "multicrop_model_leaderboard.csv")

    assert os.path.exists(results_path), "multicrop_model_results.csv must exist."
    assert os.path.exists(leaderboard_path), "multicrop_model_leaderboard.csv must exist."

    df = pd.read_csv(results_path)
    assert len(df) == 14

    accepted = df[df["model_status"] == "ACCEPTED"]
    baseline_pref = df[df["model_status"] == "BASELINE_PREFERRED"]

    assert len(accepted) == 4
    assert len(baseline_pref) == 10
    assert set(accepted["crop"]) == {"Maize", "Sesamum", "Pigeonpea", "Sugarcane"}


def test_model_artifacts_and_registry_serialization():
    """Verifies that model artifacts exist for all crops and registry contains valid SHA-256 hashes."""
    registry_path = os.path.join(MODELS_MULTICROP_DIR, "model_registry.json")
    assert os.path.exists(registry_path), "model_registry.json must exist."

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert registry["total_models_registered"] == 14
    for crop_name, meta in registry["models"].items():
        assert "sha256" in meta
        assert len(meta["sha256"]) == 64
        assert os.path.exists(os.path.join(BASE_DIR, meta["artifact_path"]))


def test_rice_model_regression_safety():
    """Guarantees that the historical Rice model metrics remain strictly preserved."""
    rice_meta_path = os.path.join(BASE_DIR, "Models", "forecasting_model_metadata.json")
    assert os.path.exists(rice_meta_path)
    with open(rice_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    metrics = meta.get("selected_model_metrics", {})
    assert metrics.get("r2_score") == 0.7866
    assert metrics.get("mae_kg_ha") == 353.01
    assert metrics.get("rmse_kg_ha") == 513.11
    assert metrics.get("mape_percent") == 18.04


def test_api_get_models_and_leaderboard():
    """Verifies GET /api/modeling/models and GET /api/modeling/leaderboard."""
    r_models = client.get("/api/modeling/models")
    assert r_models.status_code == 200
    data_models = r_models.json()
    assert data_models["total_models"] == 14
    assert data_models["accepted_count"] == 4
    assert data_models["baseline_preferred_count"] == 10

    r_lead = client.get("/api/modeling/leaderboard")
    assert r_lead.status_code == 200
    data_lead = r_lead.json()
    assert len(data_lead["leaderboard"]) == 14


def test_api_get_crop_specific_details():
    """Verifies crop detail, comparison, metrics, and features endpoints for Maize."""
    r_detail = client.get("/api/modeling/models/Maize")
    assert r_detail.status_code == 200
    assert r_detail.json()["crop"] == "Maize"

    r_comp = client.get("/api/modeling/models/Maize/comparison")
    assert r_comp.status_code == 200
    assert r_comp.json()["ml_winner"] == "RandomForestRegressor"
    assert r_comp.json()["model_status"] == "ACCEPTED"

    r_metrics = client.get("/api/modeling/models/Maize/metrics")
    assert r_metrics.status_code == 200
    assert "error_analysis" in r_metrics.json()

    r_features = client.get("/api/modeling/models/Maize/features")
    assert r_features.status_code == 200
    assert "yield_lag_1" in r_features.json()["feature_importance_native"]


def test_api_predict_crop_yield():
    """Verifies leak-free pre-season yield forecast with uncertainty dispersion."""
    payload = {
        "crop": "Maize",
        "state": "Bihar",
        "district": "Patna",
        "year": 2018,
        "yield_lag_1": 2850.0,
        "yield_lag_2": 2700.0,
        "yield_rolling_3yr_mean": 2750.0,
        "area_lag_1": 45000.0
    }
    r = client.post("/api/modeling/models/Maize/predict", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert res["crop"] == "Maize"
    assert res["predicted_yield_kg_ha"] > 0
    assert res["p10_lower_kg_ha"] is not None
    assert res["p90_upper_kg_ha"] is not None
    assert res["p10_lower_kg_ha"] <= res["p90_upper_kg_ha"]
    assert "sha256" in res["provenance"]


def test_api_unknown_crop_error():
    """Verifies 404/400 handling when querying or predicting an unknown crop."""
    r = client.get("/api/modeling/models/UnknownCrop123")
    assert r.status_code == 404

    r_pred = client.post("/api/modeling/models/UnknownCrop123/predict", json={
        "crop": "UnknownCrop123",
        "state": "State",
        "district": "Dist"
    })
    assert r_pred.status_code == 400
