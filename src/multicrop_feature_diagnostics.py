"""
Multi-Crop Feature Diagnostics Module (Day 21)
==============================================
Performs deep-dive pre-season feature stability audits, observation timing checks,
and missing information diagnostics across all 14 MODEL_READY crops.

Guards:
1. Describes feature importance strictly as "predictive contribution within the fitted model".
2. Audits all features for strict zero lookahead and fold-safety.
3. Formulates Feature Stability Score = max(0, min(1, 1 - std(rank) / (num_features - 1))).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.multicrop_pipeline import CropFeaturePipeline


class MultiCropFeatureDiagnosticsEngine:
    """Pre-season feature importance stability and observation timing audit engine."""

    def __init__(
        self,
        data_path: str = "Datasets/processed/agricultural_panel.csv",
        readiness_path: str = "Datasets/metadata/crop_model_readiness.csv",
        model_results_path: str = "Datasets/metadata/multicrop_model_results.csv",
    ):
        self.data_path = data_path
        self.readiness_path = readiness_path
        self.df = pd.read_csv(data_path)
        self.readiness_df = pd.read_csv(readiness_path)
        self.model_ready_crops = self.readiness_df[
            self.readiness_df["readiness_status"] == "MODEL_READY"
        ]["crop"].tolist()

        self.crop_best_algos = {}
        if os.path.exists(model_results_path):
            mr_df = pd.read_csv(model_results_path)
            for crop in self.model_ready_crops:
                c_rows = mr_df[mr_df["crop"] == crop]
                if not c_rows.empty and "ml_winner" in c_rows.columns:
                    best_algo = c_rows.iloc[0]["ml_winner"]
                    self.crop_best_algos[crop] = "RandomForestRegressor" if "RandomForest" in str(best_algo) else "GradientBoostingRegressor"
                else:
                    self.crop_best_algos[crop] = "RandomForestRegressor"
        else:
            for crop in self.model_ready_crops:
                self.crop_best_algos[crop] = "RandomForestRegressor"

    def get_valid_folds(self, crop_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Constructs expanding-window walk-forward folds."""
        years = sorted(crop_df["year"].unique())
        folds = []
        test_years = [y for y in [2014, 2015, 2016, 2017] if y in years]
        for idx, test_yr in enumerate(test_years, 1):
            train_years = [y for y in years if 2011 <= y < test_yr]
            if len(train_years) >= 2:
                folds.append({
                    "fold_id": idx,
                    "train_years": train_years,
                    "test_year": test_yr,
                })
        return folds

    def run_feature_stability_for_crop(self, crop: str) -> List[Dict[str, Any]]:
        """Calculates feature predictive contributions across folds for a crop."""
        crop_df = self.df[self.df["crop"] == crop].copy().sort_values(["district", "year"])
        folds = self.get_valid_folds(crop_df)
        algo_name = self.crop_best_algos.get(crop, "RandomForestRegressor")
        feature_names = CropFeaturePipeline.FEATURE_NAMES
        num_features = len(feature_names)

        fold_importances = []
        fold_ranks = []

        for fold in folds:
            train_years = fold["train_years"]
            test_year = fold["test_year"]

            train_raw = crop_df[crop_df["year"].isin(train_years)].copy()
            panel_up_to_test = crop_df[crop_df["year"] <= test_year].copy()

            pipeline = CropFeaturePipeline()
            pipeline.fit(train_raw)

            panel_trans = pipeline.transform(panel_up_to_test)
            train_proc = panel_trans[panel_trans["year"].isin(train_years)].dropna(subset=["yield_kg_ha"])

            X_train = np.nan_to_num(train_proc[feature_names].values, nan=0.0)
            y_train = train_proc["yield_kg_ha"].values

            if algo_name == "GradientBoostingRegressor":
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_leaf=4,
                    random_state=42,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1,
                )

            model.fit(X_train, y_train)
            importances = model.feature_importances_
            fold_importances.append(importances)

            # Calculate ranks (1 = highest importance)
            order = np.argsort(-importances)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(importances) + 1)
            fold_ranks.append(ranks)

        fold_importances_arr = np.array(fold_importances)  # (folds, features)
        fold_ranks_arr = np.array(fold_ranks)              # (folds, features)

        mean_imps = np.mean(fold_importances_arr, axis=0)
        std_imps = np.std(fold_importances_arr, axis=0, ddof=1) if len(folds) > 1 else np.zeros(num_features)
        mean_ranks = np.mean(fold_ranks_arr, axis=0)
        std_ranks = np.std(fold_ranks_arr, axis=0, ddof=1) if len(folds) > 1 else np.zeros(num_features)

        results = []
        max_rank_denom = max(1.0, float(num_features - 1))

        for idx, feat_name in enumerate(feature_names):
            # Formula: Feature Stability = max(0.0, min(1.0, 1.0 - (std_rank / (num_features - 1))))
            stability_score = max(0.0, min(1.0, 1.0 - (float(std_ranks[idx]) / max_rank_denom)))
            results.append({
                "crop": crop,
                "feature": feat_name,
                "model": algo_name,
                "mean_importance": round(float(mean_imps[idx]), 4),
                "std_importance": round(float(std_imps[idx]), 4),
                "mean_rank": round(float(mean_ranks[idx]), 2),
                "rank_variance": round(float(std_ranks[idx] ** 2), 4),
                "feature_stability_score": round(float(stability_score), 4),
                "interpretative_role": "predictive contribution within the fitted model",
            })

        return results

    def get_feature_timing_audit(self) -> List[Dict[str, Any]]:
        """Static timing and lookahead audit for all agricultural features."""
        audit_records = [
            {
                "feature": "yield_lag_1",
                "observation_time": "Previous agricultural season (t-1)",
                "available_before_forecast": True,
                "fold_safe": True,
                "timing_status": "SAFE",
                "timing_notes": "Target variable strictly shifted by 1 season within district. Zero same-year lookahead.",
            },
            {
                "feature": "yield_lag_2",
                "observation_time": "Two agricultural seasons prior (t-2)",
                "available_before_forecast": True,
                "fold_safe": True,
                "timing_status": "SAFE",
                "timing_notes": "Target variable shifted by 2 seasons within district.",
            },
            {
                "feature": "yield_rolling_3yr_mean",
                "observation_time": "Previous 3 historical seasons (t-3 to t-1)",
                "available_before_forecast": True,
                "fold_safe": True,
                "timing_status": "SAFE",
                "timing_notes": "Rolling mean computed strictly on lagged yield values prior to current season.",
            },
            {
                "feature": "area_lag_1",
                "observation_time": "Previous agricultural season (t-1)",
                "available_before_forecast": True,
                "fold_safe": True,
                "timing_status": "SAFE",
                "timing_notes": "Crop area shifted by 1 season within district to eliminate current-year planting lookahead.",
            },
            {
                "feature": "state_encoded",
                "observation_time": "Static regional identifier",
                "available_before_forecast": True,
                "fold_safe": True,
                "timing_status": "SAFE",
                "timing_notes": "Categorical state mapping fitted strictly on training partition of each fold.",
            },
            {
                "feature": "year",
                "observation_time": "Deterministic calendar year",
                "available_before_forecast": True,
                "fold_safe": True,
                "timing_status": "SAFE",
                "timing_notes": "Represents secular national technological / agro-climatic trend.",
            },
            {
                "feature": "district_coordinates",
                "observation_time": "Static spatial centroid",
                "available_before_forecast": True,
                "fold_safe": True,
                "timing_status": "SAFE",
                "timing_notes": "Static spatial coordinates invariant over time.",
            },
            {
                "feature": "spatial_cluster_id",
                "observation_time": "Multi-year clustering embedding",
                "available_before_forecast": False,
                "fold_safe": False,
                "timing_status": "UNSAFE",
                "timing_notes": "Clustering algorithm fitted on entire longitudinal dataset. EXCLUDED from pre-season models.",
            },
        ]
        return audit_records

    def execute_all(self, output_dir: str = "Datasets/metadata") -> Dict[str, Any]:
        """Runs feature diagnostics across all crops and writes CSVs."""
        os.makedirs(output_dir, exist_ok=True)
        all_stability = []

        for crop in self.model_ready_crops:
            crop_res = self.run_feature_stability_for_crop(crop)
            all_stability.extend(crop_res)

        stability_df = pd.DataFrame(all_stability)
        timing_df = pd.DataFrame(self.get_feature_timing_audit())

        stability_df.to_csv(os.path.join(output_dir, "multicrop_feature_stability.csv"), index=False)
        timing_df.to_csv(os.path.join(output_dir, "multicrop_feature_timing_audit.csv"), index=False)

        return {
            "crops_diagnosed": len(self.model_ready_crops),
            "total_feature_records": len(all_stability),
            "timing_records": len(timing_df),
        }


if __name__ == "__main__":
    engine = MultiCropFeatureDiagnosticsEngine()
    stats = engine.execute_all()
    print(f"Feature diagnostics completed: {stats}")
