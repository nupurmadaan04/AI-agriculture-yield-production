"""
Tests for Day 20: Multi-Crop Temporal Robustness & Walk-Forward Cross-Validation
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.modeling_service import ModelingReadinessService
from src.temporal_robustness_engine import TemporalWalkForwardEngine
from src.multicrop_pipeline import CropFeaturePipeline


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def service():
    return ModelingReadinessService()


@pytest.fixture
def engine():
    return TemporalWalkForwardEngine()


class TestWalkForwardFoldDefinition:
    """Validates expanding-window fold structure and temporal integrity."""

    def test_fold_structure(self, engine):
        df = engine.df[engine.df["crop"] == "Chickpea"]
        folds = engine.get_valid_folds(df)
        assert len(folds) == 4, f"Expected 4 folds, got {len(folds)}"

        # Verify expanding windows
        for i, fold in enumerate(folds):
            assert fold["fold_id"] == i + 1
            train_years = fold["train_years"]
            test_year = fold["test_year"]
            assert test_year > max(train_years), f"Fold {i+1} has test year {test_year} not strictly after train {train_years}"
            assert min(train_years) == 2011
            assert max(train_years) == 2013 + i
            assert test_year == 2014 + i


class TestLeakFreePreprocessing:
    """Validates that preprocessing is fitted strictly on training data."""

    def test_pipeline_zero_leakage(self, engine):
        df = engine.df[engine.df["crop"] == "Maize"].copy()
        train_raw = df[df["year"] <= 2013]
        panel_up_to_test = df[df["year"] <= 2014]

        pipeline = CropFeaturePipeline()
        pipeline.fit(train_raw)

        # Transformed features
        trans = pipeline.transform(panel_up_to_test)
        test_proc = trans[trans["year"] == 2014]

        assert not test_proc.empty
        # Target should not be in feature names
        for feat in CropFeaturePipeline.FEATURE_NAMES:
            assert feat != "yield_kg_ha"
            assert feat != "production_tonnes"
            assert feat != "spatial_cluster_id"
            assert not test_proc[feat].isna().any(), f"Feature {feat} contains NaNs in test slice"


class TestBaselineCalculations:
    """Validates mathematical correctness of statistical baseline forecasts."""

    def test_evaluate_baselines_on_fold(self, engine):
        df = engine.df[engine.df["crop"] == "Chickpea"].copy()
        train_raw = df[df["year"] <= 2013].dropna(subset=["yield_kg_ha"])
        
        pipeline = CropFeaturePipeline()
        pipeline.fit(train_raw)
        trans = pipeline.transform(df[df["year"] <= 2014])
        test_proc = trans[trans["year"] == 2014].dropna(subset=["yield_kg_ha"])

        baselines = engine.evaluate_baselines_on_fold(train_raw, test_proc)
        assert "Historical District Mean" in baselines
        assert "Naive Persistence" in baselines
        assert "Linear District Trend" in baselines

        for name, res in baselines.items():
            assert res["mae"] > 0
            assert res["rmse"] > 0
            assert "residuals" in res
            assert len(res["residuals"]) == len(test_proc)


class TestTemporalRobustnessOutputs:
    """Validates generated metadata files and metrics distributions."""

    def test_temporal_robustness_csv_exists(self):
        csv_path = os.path.join("Datasets", "metadata", "multicrop_temporal_robustness.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 14
        assert "win_rate" in df.columns
        assert "mean_mae_improvement" in df.columns
        assert "status" in df.columns

        valid_statuses = {"ROBUST_ACCEPTED", "SPLIT_SENSITIVE", "BASELINE_PREFERRED"}
        for st in df["status"]:
            assert st in valid_statuses

    def test_fold_results_csv_exists(self):
        csv_path = os.path.join("Datasets", "metadata", "multicrop_fold_results.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) >= 56  # 14 crops * 4 folds minimum
        assert set(df["test_year"].unique()) == {2014, 2015, 2016, 2017}

    def test_robustness_scores_csv_exists(self):
        csv_path = os.path.join("Datasets", "metadata", "model_robustness_scores.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 14
        assert "robustness_score" in df.columns
        for score in df["robustness_score"]:
            assert 0.0 <= score <= 100.0


class TestModelRegistryLineage:
    """Validates that model_registry.json preserves Day 19 lineage alongside Day 20 validation."""

    def test_registry_lineage_preserved(self):
        reg_path = os.path.join("Models", "multicrop", "model_registry.json")
        assert os.path.exists(reg_path)
        with open(reg_path, "r", encoding="utf-8") as f:
            reg = json.load(f)

        assert "models" in reg
        assert len(reg["models"]) >= 14
        for crop, mdata in reg["models"].items():
            assert "day20_robustness" in mdata
            rob = mdata["day20_robustness"]
            assert "baseline_win_rate" in rob
            assert "robustness_status" in rob
            assert "walk_forward_folds" in rob
            assert rob["walk_forward_folds"] == 4


class TestPreservedLegacyRiceModel:
    """Ensures Day 9 legacy validated Rice model remains completely regression-free."""

    def test_legacy_rice_model_metrics(self):
        model_card = os.path.join("Models", "model_card.json")
        if os.path.exists(model_card):
            with open(model_card, "r", encoding="utf-8") as f:
                card = json.load(f)
            if "validation" in card and "r2" in card["validation"]:
                assert card["validation"]["r2"] >= 0.78, "Legacy Rice model R2 must remain >= 0.78"


class TestRobustnessBackendAPI:
    """Validates all Day 20 REST API endpoints."""

    def test_get_all_crop_robustness(self, client):
        resp = client.get("/api/modeling/robustness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_crops"] == 14
        assert data["robust_accepted_count"] + data["split_sensitive_count"] + data["baseline_preferred_count"] == 14
        assert len(data["crops"]) == 14

    def test_get_robustness_summary(self, client):
        resp = client.get("/api/modeling/robustness/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_crops_evaluated"] == 14
        assert data["robust_accepted_count"] == 2
        assert "Chickpea" in data["robust_accepted_crops"]
        assert "Oilseeds" in data["robust_accepted_crops"]
        assert data["split_sensitive_count"] == 10
        assert data["baseline_preferred_count"] == 2

    def test_get_crop_robustness_single(self, client):
        resp = client.get("/api/modeling/robustness/Chickpea")
        assert resp.status_code == 200
        data = resp.json()
        assert data["crop"] == "Chickpea"
        assert data["status"] == "ROBUST_ACCEPTED"
        assert data["win_rate"] == 75.0
        assert data["mean_mae_improvement"] > 0

    def test_get_crop_folds(self, client):
        resp = client.get("/api/modeling/robustness/Chickpea/folds")
        assert resp.status_code == 200
        data = resp.json()
        assert data["crop"] == "Chickpea"
        assert data["fold_count"] >= 4
        assert len(data["folds"]) >= 4
        for fold in data["folds"]:
            assert fold["crop"] == "Chickpea"
            assert fold["test_year"] in [2014, 2015, 2016, 2017]

    def test_get_crop_robustness_detail(self, client):
        resp = client.get("/api/modeling/robustness/Chickpea/comparison")
        assert resp.status_code == 200
        data = resp.json()
        assert data["crop"] == "Chickpea"
        assert data["robustness_status"] == "ROBUST_ACCEPTED"
        assert len(data["folds"]) >= 4
        assert len(data["feature_stability"]) > 0

    def test_get_crop_robustness_not_found(self, client):
        resp = client.get("/api/modeling/robustness/FictionalNonExistentCrop")
        assert resp.status_code == 404
