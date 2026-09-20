"""
Unit and Integration Tests for Multi-Crop Modeling Readiness & Baselines (Day 18)
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from src.multicrop_readiness import MultiCropReadinessEngine, load_config
from backend.services.modeling_service import ModelingReadinessService

client = TestClient(app)


class TestMultiCropReadinessEngine:
    """Tests for core crop profiling, temporal continuity, and baseline modeling engine."""

    @pytest.fixture(scope="class")
    def engine(self):
        return MultiCropReadinessEngine()

    def test_config_loader(self):
        config = load_config()
        assert "criteria" in config
        assert config["criteria"]["min_total_records"] == 200
        assert config["criteria"]["min_years"] == 5
        assert config["criteria"]["min_yield_completeness"] == 0.90
        assert config["criteria"]["train_period"] == [2010, 2015]
        assert config["criteria"]["test_period"] == [2016, 2017]

    def test_crop_profiling(self, engine):
        profiles_df = engine.profile_all_crops()
        assert len(profiles_df) == 29
        assert "crop" in profiles_df.columns
        assert "total_records" in profiles_df.columns
        assert "yield_mean" in profiles_df.columns
        assert "active_districts" in profiles_df.columns
        assert (profiles_df["total_records"] == 2469).all()

    def test_temporal_continuity(self, engine):
        cont_df = engine.evaluate_temporal_continuity()
        assert len(cont_df) == 29
        assert "continuous_district_count" in cont_df.columns
        assert "median_continuity" in cont_df.columns
        assert (cont_df["median_continuity"] >= 0.70).all()

    def test_readiness_classification(self, engine):
        readiness_df, target_df = engine.classify_readiness()
        assert len(readiness_df) == 29
        assert len(target_df) == 29

        status_counts = readiness_df["readiness_status"].value_counts()
        assert status_counts["MODEL_READY"] == 14
        assert status_counts["ANALYTICS_READY"] == 9
        assert status_counts["INSUFFICIENT_DATA"] == 6

        # Check that Rice, Wheat, Maize are MODEL_READY
        m_ready_crops = readiness_df[readiness_df["readiness_status"] == "MODEL_READY"]["crop"].tolist()
        assert "Rice" in m_ready_crops
        assert "Wheat" in m_ready_crops
        assert "Maize" in m_ready_crops

        # Check that Fruit/Veg aggregates are INSUFFICIENT_DATA
        insufficient_crops = readiness_df[readiness_df["readiness_status"] == "INSUFFICIENT_DATA"]["crop"].tolist()
        assert "Fruits and Vegetables" in insufficient_crops
        assert "Fodder" in insufficient_crops

    def test_baseline_evaluation_and_zero_leakage(self, engine):
        baselines_df = engine.evaluate_baselines()
        assert len(baselines_df) > 0
        evaluated_df = baselines_df[baselines_df["status"] == "EVALUATED"]
        assert len(evaluated_df) > 0

        # Check required models are evaluated
        models = evaluated_df["model"].unique()
        assert "Historical District Mean" in models
        assert "Naive Persistence (t-1)" in models
        assert "Historical Crop Mean" in models
        assert "Linear District Trend" in models

        # Check metric ranges
        assert (evaluated_df["mae"] > 0).all()
        assert (evaluated_df["rmse"] > 0).all()


class TestModelingReadinessAPI:
    """Tests for FastAPI modeling readiness endpoints."""

    def test_api_readiness_summary(self):
        r = client.get("/api/modeling/readiness/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total_crops"] == 29
        assert data["model_ready_count"] == 14
        assert data["analytics_ready_count"] == 9
        assert data["insufficient_data_count"] == 6
        assert data["active_dataset_version"] == "AGRI_PANEL_1.0"

    def test_api_crop_readiness_all(self):
        r = client.get("/api/modeling/crops/readiness")
        assert r.status_code == 200
        data = r.json()
        assert data["total_crops"] == 29
        assert len(data["crops"]) == 29

    def test_api_crop_readiness_filter(self):
        r = client.get("/api/modeling/crops/readiness?status=MODEL_READY")
        assert r.status_code == 200
        data = r.json()
        assert data["total_crops"] == 14
        for c in data["crops"]:
            assert c["readiness_status"] == "MODEL_READY"

    def test_api_crop_readiness_single(self):
        r = client.get("/api/modeling/crops/Wheat/readiness")
        assert r.status_code == 200
        data = r.json()
        assert data["crop"] == "Wheat"
        assert data["readiness_status"] == "MODEL_READY"
        assert data["readiness_score"] >= 80.0

    def test_api_crop_baselines(self):
        r = client.get("/api/modeling/crops/Wheat/baselines")
        assert r.status_code == 200
        data = r.json()
        assert data["crop"] == "Wheat"
        assert len(data["baselines"]) == 4
        assert data["best_model_by_mae"] is not None
        assert data["best_mae"] is not None

    def test_api_feature_compatibility(self):
        r = client.get("/api/modeling/feature-compatibility")
        assert r.status_code == 200
        data = r.json()
        assert data["total_features_audited"] >= 6
        features = {f["feature_name"]: f for f in data["features"]}
        assert features["current_year_production"]["leakage_risk"] == "CRITICAL (100% LEAKAGE)"
        assert features["yield_lag_1"]["pre_season_valid"] is True

    def test_api_architecture_decision(self):
        r = client.get("/api/modeling/architecture-decision")
        assert r.status_code == 200
        data = r.json()
        assert data["decision"] == "SEPARATE_CROP_SPECIFIC_REGRESSORS"
        assert data["crop_specific_justified"] is True
        assert data["global_model_justified"] is False

    def test_rice_model_regression_safety(self):
        """Ensures Rice model metrics remain exactly preserved in model registry."""
        r = client.get("/api/models/registry")
        assert r.status_code == 200
        data = r.json()
        primary_model = [m for m in data["models"] if m.get("is_primary")][0]
        metrics = primary_model["test_metrics"]
        assert round(metrics["r2"], 4) == 0.7866
        assert round(metrics["mae"], 2) == 353.01
        assert round(metrics["rmse"], 2) == 513.11
        assert round(metrics["mape"], 2) == 18.04
