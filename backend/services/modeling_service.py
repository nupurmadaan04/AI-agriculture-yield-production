"""
Modeling Readiness & Multi-Crop Forecasting Domain Service
==========================================================
Provides domain queries for multi-crop modeling readiness,
crop-specific baseline benchmarking, feature compatibility audits,
architecture decision records, and multi-crop model inferences.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from backend.schemas.modeling import (
    CropReadinessItem,
    CropReadinessResponse,
    CropBaselineItem,
    CropBaselinesResponse,
    ReadinessSummaryResponse,
    FeatureCompatibilityItem,
    FeatureCompatibilityResponse,
    ArchitectureDecisionResponse,
    MultiCropModelItem,
    MultiCropModelsResponse,
    CropModelComparisonResponse,
    CropModelMetricsResponse,
    CropModelFeaturesResponse,
    MultiCropLeaderboardItem,
    MultiCropLeaderboardResponse,
    MultiCropRegistryResponse,
    CropPredictionRequest,
    CropPredictionResponse,
    FoldResultItem,
    CropRobustnessItem,
    CropRobustnessResponse,
    CropFoldsResponse,
    CropRobustnessDetailResponse,
    RobustnessSummaryResponse,
    CropDiagnosisItem,
    CropDiagnosisSummaryResponse,
    CropErrorRegimeItem,
    CropErrorRegimesResponse,
    CropDistrictErrorItem,
    CropDistrictErrorsResponse,
    CropYearErrorItem,
    CropYearErrorsResponse,
    CropFeatureStabilityItem,
    CropFeatureStabilityResponse,
    CropModelSelectionItem,
    CropModelSelectionResponse,
    CropForecastingStrategyItem,
    CropForecastingStrategyResponse,
    ExogenousSourceItem,
    ExogenousSourcesResponse,
    ExogenousCoverageItem,
    ExogenousCoverageResponse,
    ExogenousFeatureItem,
    ExogenousFeaturesResponse,
    ExogenousAblationItem,
    ExogenousAblationResponse,
    ExogenousFoldResultItem,
    ExogenousCropFoldsResponse,
    ExogenousCropResultItem,
    ExogenousCropResultResponse,
    ExogenousModelSelectionItem,
    ExogenousModelSelectionResponse,
    ExogenousSummaryResponse,
    FinalStrategyItem,
    FinalValidationFoldItem,
    FinalValidationResponse,
    SingleCropFinalValidationResponse,
    ResidualQuantileItem,
    ResidualYearItem,
    ResidualDiagnosticsResponse,
    PredictionBiasItem,
    PredictionBiasResponse,
    ReproducibilityItem,
    ReproducibilityResponse,
    FinalModelCertificationItem,
    FinalModelCertificationResponse,
    ForecastPredictRequest,
    ForecastPredictResponse,
    ForecastStrategyItem,
    ForecastStrategiesResponse,
    ForecastCoverageItem,
    ForecastCoverageResponse,
    ForecastCertificationSummaryResponse,
    ForecastAuditItem,
    ForecastAuditResponse,
    ForecastHealthResponse,
)

logger = logging.getLogger("agricultural_platform")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METADATA_DIR = os.path.join(BASE_DIR, "Datasets", "metadata")
PROCESSED_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
MODELS_MULTICROP_DIR = os.path.join(BASE_DIR, "Models", "multicrop")


class ModelingReadinessService:
    """Encapsulates multi-crop modeling readiness, baseline queries, forecasting, and temporal robustness."""

    def __init__(self):
        self.readiness_csv = os.path.join(METADATA_DIR, "crop_model_readiness.csv")
        self.target_csv = os.path.join(METADATA_DIR, "crop_target_profiles.csv")
        self.temporal_csv = os.path.join(METADATA_DIR, "crop_temporal_readiness.csv")
        self.baselines_csv = os.path.join(PROCESSED_DIR, "multicrop_baseline_results.csv")
        self.results_csv = os.path.join(METADATA_DIR, "multicrop_model_results.csv")
        self.leaderboard_csv = os.path.join(METADATA_DIR, "multicrop_model_leaderboard.csv")
        self.registry_json = os.path.join(MODELS_MULTICROP_DIR, "model_registry.json")
        self.robustness_csv = os.path.join(METADATA_DIR, "multicrop_temporal_robustness.csv")
        self.fold_results_csv = os.path.join(METADATA_DIR, "multicrop_fold_results.csv")
        self.robustness_scores_csv = os.path.join(METADATA_DIR, "model_robustness_scores.csv")
        self.diagnosis_csv = os.path.join(METADATA_DIR, "multicrop_error_diagnosis.csv")
        self.regimes_csv = os.path.join(METADATA_DIR, "multicrop_error_regimes.csv")
        self.year_errors_csv = os.path.join(METADATA_DIR, "multicrop_year_error_analysis.csv")
        self.district_errors_csv = os.path.join(METADATA_DIR, "multicrop_district_error_analysis.csv")
        self.feature_stability_csv = os.path.join(METADATA_DIR, "multicrop_feature_stability.csv")
        self.feature_timing_csv = os.path.join(METADATA_DIR, "multicrop_feature_timing_audit.csv")
        self.selection_csv = os.path.join(METADATA_DIR, "multicrop_model_selection.csv")
        self.forecasting_strategy_csv = os.path.join(METADATA_DIR, "multicrop_forecasting_strategy.csv")
        self.exo_source_csv = os.path.join(METADATA_DIR, "exogenous_source_registry.csv")
        self.exo_feature_csv = os.path.join(METADATA_DIR, "exogenous_feature_registry.csv")
        self.exo_coverage_csv = os.path.join(METADATA_DIR, "exogenous_coverage_audit.csv")
        self.exo_temporal_csv = os.path.join(METADATA_DIR, "exogenous_temporal_audit.csv")
        self.exo_leakage_csv = os.path.join(METADATA_DIR, "exogenous_leakage_audit.csv")
        self.exo_ablation_csv = os.path.join(METADATA_DIR, "exogenous_ablation_results.csv")
        self.exo_folds_csv = os.path.join(METADATA_DIR, "exogenous_fold_results.csv")
        self.exo_crop_results_csv = os.path.join(METADATA_DIR, "exogenous_crop_results.csv")
        self.exo_selection_csv = os.path.join(METADATA_DIR, "exogenous_model_selection.csv")

    def _get_readiness_df(self) -> pd.DataFrame:
        if os.path.exists(self.readiness_csv):
            return pd.read_csv(self.readiness_csv)
        return pd.DataFrame()

    def _get_target_df(self) -> pd.DataFrame:
        if os.path.exists(self.target_csv):
            return pd.read_csv(self.target_csv)
        return pd.DataFrame()

    def _get_baselines_df(self) -> pd.DataFrame:
        if os.path.exists(self.baselines_csv):
            return pd.read_csv(self.baselines_csv)
        return pd.DataFrame()

    def _get_results_df(self) -> pd.DataFrame:
        if os.path.exists(self.results_csv):
            return pd.read_csv(self.results_csv)
        return pd.DataFrame()

    def _get_leaderboard_df(self) -> pd.DataFrame:
        if os.path.exists(self.leaderboard_csv):
            return pd.read_csv(self.leaderboard_csv)
        return pd.DataFrame()

    def _get_registry_dict(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_json):
            with open(self.registry_json, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"version": "1.0.0", "models": {}}

    def get_all_crop_readiness(self, status_filter: Optional[str] = None) -> CropReadinessResponse:
        """Returns readiness assessments and scores for all crops."""
        df = self._get_readiness_df()
        target_df = self._get_target_df().set_index("crop") if os.path.exists(self.target_csv) else pd.DataFrame()
        baselines_df = self._get_baselines_df()

        if df.empty:
            return CropReadinessResponse(total_crops=0, crops=[])

        best_baselines = {}
        if not baselines_df.empty:
            eval_df = baselines_df[baselines_df["status"] == "EVALUATED"].copy()
            for crop_name, group in eval_df.groupby("crop"):
                best_row = group.sort_values("mae").iloc[0]
                best_baselines[crop_name] = {
                    "model": best_row["model"],
                    "mae": float(best_row["mae"]),
                    "r2": float(best_row["r2"]) if pd.notna(best_row["r2"]) else None,
                }

        items = []
        for _, row in df.iterrows():
            crop = row["crop"]
            status = row["readiness_status"]

            if status_filter and status_filter.upper() != "ALL" and status != status_filter.upper():
                continue

            t_info = target_df.loc[crop] if crop in target_df.index else {}
            b_info = best_baselines.get(crop, {})

            items.append(CropReadinessItem(
                crop=crop,
                readiness_score=float(row["readiness_score"]),
                readiness_status=status,
                data_volume_score=float(row["data_volume_score"]),
                temporal_score=float(row["temporal_score"]),
                geographic_score=float(row["geographic_score"]),
                target_quality_score=float(row["target_quality_score"]),
                validation_score=float(row["validation_score"]),
                feature_score=float(row["feature_score"]),
                blocking_reasons=str(row["blocking_reasons"]),
                recommendation=str(row["recommendation"]),
                total_records=int(t_info.get("total_records")) if isinstance(t_info, pd.Series) and pd.notna(t_info.get("total_records")) else None,
                active_districts=int(t_info.get("active_districts")) if isinstance(t_info, pd.Series) and pd.notna(t_info.get("active_districts")) else None,
                zero_yield_pct=float(t_info.get("zero_yield_pct", 0.0)) if isinstance(t_info, pd.Series) and pd.notna(t_info.get("zero_yield_pct")) else None,
                best_baseline_model=b_info.get("model"),
                best_baseline_mae=b_info.get("mae"),
                best_baseline_r2=b_info.get("r2"),
            ))

        return CropReadinessResponse(total_crops=len(items), crops=items)

    def get_crop_readiness(self, crop: str) -> Optional[CropReadinessItem]:
        """Returns readiness assessment for a specific crop."""
        all_res = self.get_all_crop_readiness()
        for item in all_res.crops:
            if item.crop.lower() == crop.lower():
                return item
        return None

    def get_crop_baselines(self, crop: str) -> CropBaselinesResponse:
        """Returns baseline benchmarking results for a specific crop."""
        baselines_df = self._get_baselines_df()
        if baselines_df.empty:
            return CropBaselinesResponse(crop=crop, baselines=[])

        filtered = baselines_df[baselines_df["crop"].str.lower() == crop.lower()]
        items = []
        best_model = None
        best_mae = float("inf")

        for _, row in filtered.iterrows():
            mae = float(row["mae"]) if pd.notna(row["mae"]) else None
            if mae is not None and mae < best_mae and row["status"] == "EVALUATED":
                best_mae = mae
                best_model = str(row["model"])

            items.append(CropBaselineItem(
                crop=str(row["crop"]),
                model=str(row["model"]),
                train_period=str(row["train_period"]),
                test_period=str(row["test_period"]),
                train_records=int(row["train_records"]),
                test_records=int(row["test_records"]),
                mae=mae,
                rmse=float(row["rmse"]) if pd.notna(row["rmse"]) else None,
                r2=float(row["r2"]) if pd.notna(row["r2"]) else None,
                mape=float(row["mape"]) if pd.notna(row["mape"]) else None,
                smape=float(row["smape"]) if pd.notna(row["smape"]) else None,
                valid_predictions=int(row["valid_predictions"]),
                invalid_predictions=int(row["invalid_predictions"]),
                notes=str(row["notes"]),
                status=str(row["status"]),
            ))

        return CropBaselinesResponse(
            crop=crop,
            baselines=items,
            best_model_by_mae=best_model,
            best_mae=round(best_mae, 2) if best_mae != float("inf") else None,
        )

    def get_readiness_summary(self) -> ReadinessSummaryResponse:
        """Returns aggregate modeling readiness KPIs."""
        df = self._get_readiness_df()
        if df.empty:
            return ReadinessSummaryResponse(
                total_crops=0,
                model_ready_count=0,
                analytics_ready_count=0,
                insufficient_data_count=0,
                model_ready_crops=[],
                analytics_ready_crops=[],
                insufficient_data_crops=[],
                total_records_evaluated=0,
                active_dataset_version="AGRI_PANEL_1.0",
            )

        mr = df[df["readiness_status"] == "MODEL_READY"]["crop"].tolist()
        ar = df[df["readiness_status"] == "ANALYTICS_READY"]["crop"].tolist()
        id_crops = df[df["readiness_status"] == "INSUFFICIENT_DATA"]["crop"].tolist()
        total_rec = int(df["total_records"].sum()) if "total_records" in df.columns else 71601

        return ReadinessSummaryResponse(
            total_crops=len(df),
            model_ready_count=len(mr),
            analytics_ready_count=len(ar),
            insufficient_data_count=len(id_crops),
            model_ready_crops=mr,
            analytics_ready_crops=ar,
            insufficient_data_crops=id_crops,
            total_records_evaluated=total_rec,
            active_dataset_version="AGRI_PANEL_1.0",
        )

    def get_feature_compatibility(self) -> FeatureCompatibilityResponse:
        """Returns the scientific feature compatibility and anti-leakage audit matrix."""
        features = [
            FeatureCompatibilityItem(
                feature_name="yield_lag_1",
                source_type="Historical Panel",
                timing="Prior Crop Season (t-1)",
                pre_season_valid=True,
                leakage_risk="ZERO",
                classification="UNIVERSAL",
                recommendation="Admissible for all crops with consecutive district time-series.",
            ),
            FeatureCompatibilityItem(
                feature_name="yield_rolling_3yr_mean",
                source_type="Historical Panel",
                timing="Prior 3 Seasons (t-3 to t-1)",
                pre_season_valid=True,
                leakage_risk="ZERO",
                classification="UNIVERSAL",
                recommendation="Primary baseline feature for historical trend persistence.",
            ),
            FeatureCompatibilityItem(
                feature_name="area_lag_1",
                source_type="Historical Panel",
                timing="Prior Crop Season (t-1)",
                pre_season_valid=True,
                leakage_risk="ZERO",
                classification="UNIVERSAL",
                recommendation="Admissible proxy for district crop land allocation.",
            ),
            FeatureCompatibilityItem(
                feature_name="spatial_cluster_id",
                source_type="Agro-Climatic",
                timing="Static Spatial Baseline",
                pre_season_valid=True,
                leakage_risk="ZERO",
                classification="UNIVERSAL",
                recommendation="K-Means (k=4) cluster embeddings apply universally across India.",
            ),
            FeatureCompatibilityItem(
                feature_name="current_year_area",
                source_type="Observed Survey",
                timing="Early Sowing Season (t)",
                pre_season_valid=False,
                leakage_risk="LOW",
                classification="EARLY_SEASON",
                recommendation="Admissible only if sowing survey precedes yield formation.",
            ),
            FeatureCompatibilityItem(
                feature_name="current_year_production",
                source_type="Observed Survey",
                timing="Harvest Season (t)",
                pre_season_valid=False,
                leakage_risk="CRITICAL (100% LEAKAGE)",
                classification="LEAKAGE_RISK",
                recommendation="Strictly prohibited: mathematically reconstructs Yield = Production / Area.",
            ),
            FeatureCompatibilityItem(
                feature_name="annual_rainfall",
                source_type="Meteorological",
                timing="Post-Harvest (t)",
                pre_season_valid=False,
                leakage_risk="HIGH",
                classification="POST_HARVEST_ONLY",
                recommendation="Post-harvest benchmarking only; invalid before planting.",
            ),
        ]
        return FeatureCompatibilityResponse(total_features_audited=len(features), features=features)

    def get_architecture_decision(self) -> ArchitectureDecisionResponse:
        """Returns the empirical architectural decision record."""
        return ArchitectureDecisionResponse(
            decision="SEPARATE_CROP_SPECIFIC_REGRESSORS",
            recommended_architecture="Dedicated Random Forest / Gradient Boosted Trees per Crop (Option B)",
            global_model_justified=False,
            crop_specific_justified=True,
            hierarchical_justified=False,
            summary="A single global pooled model is scientifically disfavored due to extreme target scale disparity (100x between Sugarcane and Cotton), biological mechanism incompatibility, and loss gradient domination. Separate crop-specific regressors are empirically justified.",
            empirical_justification=[
                "Target scale disparity: Sugarcane yields (60,000 kg/ha) vs Chickpea (920 kg/ha) vs Cotton (350 kg/ha).",
                "Loss gradient bias: Raw MSE on pooled crops prioritizes high-yield outliers while neglecting pulses.",
                "Biological divergence: Paddy water response differs completely from dryland pulse agro-ecology.",
                "Baseline evidence: Local district historical mean outperforms national crop pooling (R2=0.42 vs R2=-0.05).",
            ],
        )

    # -----------------------------------------------------------------------
    # Day 19 Multi-Crop Forecasting Domain Services
    # -----------------------------------------------------------------------

    def get_all_models(self, status_filter: Optional[str] = None) -> MultiCropModelsResponse:
        """Returns all trained crop-specific models and candidate results."""
        df = self._get_results_df()
        if df.empty:
            return MultiCropModelsResponse(total_models=0, accepted_count=0, baseline_preferred_count=0, models=[])

        items = []
        for _, r in df.iterrows():
            status = str(r["model_status"])
            if status_filter and status_filter.upper() != "ALL" and status != status_filter.upper():
                continue

            # Best model metrics
            is_rf = r["best_model"] == "RandomForestRegressor"
            mae = float(r["rf_mae"]) if is_rf else float(r["gb_mae"]) if r["best_model"] == "GradientBoostingRegressor" else float(r["best_mae"])
            rmse = float(r["rf_rmse"]) if is_rf else float(r["gb_rmse"]) if r["best_model"] == "GradientBoostingRegressor" else float(r["best_rmse"])
            r2 = float(r["rf_r2"]) if is_rf else float(r["gb_r2"]) if r["best_model"] == "GradientBoostingRegressor" else (float(r["best_r2"]) if pd.notna(r["best_r2"]) else 0.0)

            items.append(MultiCropModelItem(
                crop=str(r["crop"]),
                model_id=f"multicrop_{str(r['crop']).lower().replace(' ', '_')}_forecaster",
                algorithm=str(r["best_model"]),
                version="1.0.0",
                training_period="2011–2015",
                evaluation_period="2016–2017",
                train_records=int(r["train_records"]),
                test_records=int(r["test_records"]),
                mae=round(mae, 2),
                rmse=round(rmse, 2),
                r2=round(r2, 4),
                baseline_model=str(r["baseline_model"]),
                baseline_mae=float(r["baseline_mae"]),
                mae_improvement_pct=float(r["mae_improvement_pct"]),
                model_status=status,
                recommendation=str(r["recommendation"]),
                artifact_path=str(r["artifact_path"]),
                sha256=str(r["sha256"]),
            ))

        acc_count = int((df["model_status"] == "ACCEPTED").sum())
        base_count = int((df["model_status"] == "BASELINE_PREFERRED").sum())

        return MultiCropModelsResponse(
            total_models=len(items),
            accepted_count=acc_count,
            baseline_preferred_count=base_count,
            models=items,
        )

    def get_crop_model_details(self, crop: str) -> Optional[MultiCropModelItem]:
        """Returns model details for a single crop."""
        models = self.get_all_models().models
        for m in models:
            if m.crop.lower() == crop.lower():
                return m
        return None

    def get_crop_model_comparison(self, crop: str) -> Optional[CropModelComparisonResponse]:
        """Returns comparative evaluation between Baseline, RF, and GB for a specific crop."""
        df = self._get_results_df()
        if df.empty:
            return None

        match = df[df["crop"].str.lower() == crop.lower()]
        if match.empty:
            return None

        r = match.iloc[0]
        return CropModelComparisonResponse(
            crop=str(r["crop"]),
            baseline_model=str(r["baseline_model"]),
            baseline_mae=float(r["baseline_mae"]),
            baseline_rmse=float(r["baseline_rmse"]),
            baseline_r2=float(r["baseline_r2"]) if pd.notna(r["baseline_r2"]) else None,
            rf_mae=float(r["rf_mae"]),
            rf_rmse=float(r["rf_rmse"]),
            rf_r2=float(r["rf_r2"]),
            gb_mae=float(r["gb_mae"]),
            gb_rmse=float(r["gb_rmse"]),
            gb_r2=float(r["gb_r2"]),
            ml_winner=str(r["ml_winner"]),
            overall_winner=str(r["best_model"]),
            best_mae=float(r["best_mae"]),
            mae_improvement_vs_baseline=float(r["mae_improvement_vs_baseline"]),
            mae_improvement_pct=float(r["mae_improvement_pct"]),
            model_status=str(r["model_status"]),
            recommendation=str(r["recommendation"]),
        )

    def get_crop_model_metrics(self, crop: str) -> Optional[CropModelMetricsResponse]:
        """Returns deep validation metrics, error quantiles, and uncertainty spread for a crop."""
        registry = self._get_registry_dict().get("models", {})
        meta = None
        for c_name, data in registry.items():
            if c_name.lower() == crop.lower():
                meta = data
                break

        if not meta:
            return None

        return CropModelMetricsResponse(
            crop=meta["crop"],
            algorithm=meta["algorithm"],
            status=meta["status"],
            training_period=meta["training_period"],
            evaluation_period=meta["evaluation_period"],
            train_records=meta["train_records"],
            test_records=meta["test_records"],
            metrics=meta["metrics"],
            baseline_comparison=meta["baseline_comparison"],
            error_analysis=meta["error_analysis"],
            uncertainty_spread_p10_p90=meta.get("uncertainty_spread_p10_p90"),
        )

    def get_crop_model_features(self, crop: str) -> Optional[CropModelFeaturesResponse]:
        """Returns feature list and importance rankings for a crop."""
        registry = self._get_registry_dict().get("models", {})
        meta = None
        for c_name, data in registry.items():
            if c_name.lower() == crop.lower():
                meta = data
                break

        if not meta:
            return None

        features = list(meta.get("feature_importance_native", {}).keys())
        return CropModelFeaturesResponse(
            crop=meta["crop"],
            algorithm=meta["algorithm"],
            features=features,
            feature_importance_native=meta.get("feature_importance_native", {}),
            feature_importance_permutation=meta.get("feature_importance_permutation", {}),
        )

    def get_multicrop_leaderboard(self) -> MultiCropLeaderboardResponse:
        """Returns the full multi-crop forecasting leaderboard."""
        df = self._get_leaderboard_df()
        if df.empty:
            return MultiCropLeaderboardResponse(total_crops=0, accepted_count=0, baseline_preferred_count=0, leaderboard=[])

        items = []
        for _, r in df.iterrows():
            items.append(MultiCropLeaderboardItem(
                crop=str(r["crop"]),
                best_model=str(r["best_model"]),
                best_mae=float(r["best_mae"]),
                best_rmse=float(r["best_rmse"]),
                best_r2=float(r["best_r2"]) if pd.notna(r["best_r2"]) else None,
                baseline_model=str(r["baseline_model"]),
                baseline_mae=float(r["baseline_mae"]),
                mae_improvement_pct=float(r["mae_improvement_pct"]),
                model_status=str(r["model_status"]),
            ))

        acc_count = int((df["model_status"] == "ACCEPTED").sum())
        base_count = int((df["model_status"] == "BASELINE_PREFERRED").sum())

        return MultiCropLeaderboardResponse(
            total_crops=len(items),
            accepted_count=acc_count,
            baseline_preferred_count=base_count,
            leaderboard=items,
        )

    def get_multicrop_registry(self) -> MultiCropRegistryResponse:
        """Returns the complete multi-crop model registry."""
        reg = self._get_registry_dict()
        return MultiCropRegistryResponse(
            version=reg.get("version", "1.0.0"),
            last_updated=reg.get("last_updated", ""),
            total_models_registered=len(reg.get("models", {})),
            models=reg.get("models", {}),
        )

    def predict_crop_yield(self, req: CropPredictionRequest) -> CropPredictionResponse:
        """Executes crop-specific pre-season yield forecast with provenance and bounds."""
        crop_slug = req.crop.lower().replace(" ", "_")
        artifact_path = os.path.join(MODELS_MULTICROP_DIR, crop_slug, "model_pipeline.pkl")
        metadata_path = os.path.join(MODELS_MULTICROP_DIR, crop_slug, "model_metadata.json")

        if not os.path.exists(artifact_path) or not os.path.exists(metadata_path):
            raise ValueError(f"No trained forecasting artifact found for crop '{req.crop}'.")

        artifact = joblib.load(artifact_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model = artifact["model"]
        pipe = artifact["feature_pipeline"]

        # Default feature imputation from training means if not supplied
        y_lag1 = req.yield_lag_1 if req.yield_lag_1 is not None else pipe.dist_train_means.get(req.district, pipe.crop_train_mean)
        y_lag2 = req.yield_lag_2 if req.yield_lag_2 is not None else y_lag1
        y_roll3 = req.yield_rolling_3yr_mean if req.yield_rolling_3yr_mean is not None else y_lag1
        area_lag1 = req.area_lag_1 if req.area_lag_1 is not None else 10000.0
        state_enc = pipe.state_to_code.get(req.state, -1)

        # Vector format: [yield_lag_1, yield_lag_2, yield_rolling_3yr_mean, area_lag_1, state_encoded, year]
        X = np.array([[y_lag1, y_lag2, y_roll3, area_lag1, state_enc, req.year]])
        pred_val = float(model.predict(X)[0])

        # Tree dispersion if RF
        p10, p90 = None, None
        if isinstance(model, RandomForestRegressor):
            tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
            p10 = round(float(np.percentile(tree_preds, 10)), 2)
            p90 = round(float(np.percentile(tree_preds, 90)), 2)

        return CropPredictionResponse(
            crop=req.crop,
            state=req.state,
            district=req.district,
            target_year=req.year,
            predicted_yield_kg_ha=round(pred_val, 2),
            p10_lower_kg_ha=p10,
            p90_upper_kg_ha=p90,
            model_id=metadata["model_id"],
            algorithm=metadata["algorithm"],
            model_version=metadata["version"],
            model_scope="crop_specific",
            model_status=metadata["status"],
            dataset_version=metadata["dataset_version"],
            provenance={
                "training_period": metadata["training_period"],
                "input_features": {
                    "yield_lag_1": y_lag1,
                    "yield_lag_2": y_lag2,
                    "yield_rolling_3yr_mean": y_roll3,
                    "area_lag_1": area_lag1,
                    "state_encoded": state_enc,
                    "year": req.year,
                },
                "sha256": metadata["sha256"],
                "baseline_benchmark_mae": metadata["baseline_comparison"]["baseline_mae"],
            },
        )

    # -----------------------------------------------------------------------
    # Day 20 Temporal Robustness & Walk-Forward Validation Methods
    # -----------------------------------------------------------------------

    def _get_robustness_df(self) -> pd.DataFrame:
        if os.path.exists(self.robustness_csv):
            return pd.read_csv(self.robustness_csv)
        return pd.DataFrame()

    def _get_fold_results_df(self) -> pd.DataFrame:
        if os.path.exists(self.fold_results_csv):
            return pd.read_csv(self.fold_results_csv)
        return pd.DataFrame()

    def _get_robustness_scores_df(self) -> pd.DataFrame:
        if os.path.exists(self.robustness_scores_csv):
            return pd.read_csv(self.robustness_scores_csv)
        return pd.DataFrame()

    def get_all_crop_robustness(self) -> CropRobustnessResponse:
        """Retrieves temporal walk-forward robustness evaluation across all crops."""
        df = self._get_robustness_df()
        scores_df = self._get_robustness_scores_df()
        score_map = {}
        if not scores_df.empty and "crop" in scores_df.columns:
            score_map = dict(zip(scores_df["crop"], scores_df["robustness_score"]))

        if df.empty:
            return CropRobustnessResponse(
                total_crops=0,
                robust_accepted_count=0,
                split_sensitive_count=0,
                baseline_preferred_count=0,
                crops=[],
            )

        items = []
        for _, row in df.iterrows():
            c_name = row["crop"]
            items.append(
                CropRobustnessItem(
                    crop=c_name,
                    model=row["model"],
                    fold_count=int(row["fold_count"]),
                    mean_mae=float(row["mean_mae"]),
                    median_mae=float(row["median_mae"]),
                    std_mae=float(row["std_mae"]),
                    mean_rmse=float(row["mean_rmse"]),
                    std_rmse=float(row["std_rmse"]),
                    mean_r2=float(row["mean_r2"]),
                    std_r2=float(row["std_r2"]),
                    baseline_mae=float(row["baseline_mae"]),
                    mean_mae_improvement=float(row["mean_mae_improvement"]),
                    median_mae_improvement=float(row["median_mae_improvement"]),
                    win_rate=float(row["win_rate"]),
                    status=row["status"],
                    robustness_score=score_map.get(c_name),
                )
            )

        robust_accepted = sum(1 for i in items if i.status == "ROBUST_ACCEPTED")
        split_sensitive = sum(1 for i in items if i.status == "SPLIT_SENSITIVE")
        baseline_pref = sum(1 for i in items if i.status == "BASELINE_PREFERRED")

        return CropRobustnessResponse(
            total_crops=len(items),
            robust_accepted_count=robust_accepted,
            split_sensitive_count=split_sensitive,
            baseline_preferred_count=baseline_pref,
            crops=items,
        )

    def get_crop_robustness(self, crop: str) -> CropRobustnessItem:
        """Retrieves temporal robustness summary for a specific crop."""
        df = self._get_robustness_df()
        scores_df = self._get_robustness_scores_df()
        score_map = {}
        if not scores_df.empty and "crop" in scores_df.columns:
            score_map = dict(zip(scores_df["crop"], scores_df["robustness_score"]))

        crop_row = df[df["crop"].str.lower() == crop.lower()]
        if crop_row.empty:
            raise ValueError(f"No temporal robustness data found for crop '{crop}'.")

        row = crop_row.iloc[0]
        c_name = row["crop"]
        return CropRobustnessItem(
            crop=c_name,
            model=row["model"],
            fold_count=int(row["fold_count"]),
            mean_mae=float(row["mean_mae"]),
            median_mae=float(row["median_mae"]),
            std_mae=float(row["std_mae"]),
            mean_rmse=float(row["mean_rmse"]),
            std_rmse=float(row["std_rmse"]),
            mean_r2=float(row["mean_r2"]),
            std_r2=float(row["std_r2"]),
            baseline_mae=float(row["baseline_mae"]),
            mean_mae_improvement=float(row["mean_mae_improvement"]),
            median_mae_improvement=float(row["median_mae_improvement"]),
            win_rate=float(row["win_rate"]),
            status=row["status"],
            robustness_score=score_map.get(c_name),
        )

    def get_crop_folds(self, crop: str) -> CropFoldsResponse:
        """Retrieves walk-forward fold-by-fold results for a specific crop."""
        df = self._get_fold_results_df()
        crop_df = df[df["crop"].str.lower() == crop.lower()]
        if crop_df.empty:
            raise ValueError(f"No fold results found for crop '{crop}'.")

        items = []
        for _, row in crop_df.iterrows():
            items.append(
                FoldResultItem(
                    crop=row["crop"],
                    fold_id=int(row["fold_id"]),
                    train_start_year=int(row["train_start_year"]),
                    train_end_year=int(row["train_end_year"]),
                    test_year=int(row["test_year"]),
                    train_samples=int(row["train_samples"]),
                    test_samples=int(row["test_samples"]),
                    model=row["model"],
                    mae=float(row["mae"]),
                    rmse=float(row["rmse"]),
                    r2=float(row["r2"]),
                    mape=float(row["mape"]) if pd.notna(row.get("mape")) else None,
                    smape=float(row["smape"]) if pd.notna(row.get("smape")) else None,
                    best_baseline_model=row["best_baseline_model"],
                    best_baseline_mae=float(row["best_baseline_mae"]),
                    win_vs_baseline=bool(row["win_vs_baseline"]),
                    mae_improvement_pct=float(row["mae_improvement_pct"]),
                    mean_residual=float(row["mean_residual"]),
                    std_residual=float(row["std_residual"]),
                )
            )

        return CropFoldsResponse(
            crop=crop_df.iloc[0]["crop"],
            fold_count=len(items),
            folds=items,
        )

    def get_crop_robustness_detail(self, crop: str) -> CropRobustnessDetailResponse:
        """Retrieves comprehensive multi-model fold comparison and stability breakdown."""
        folds_resp = self.get_crop_folds(crop)
        rob_item = self.get_crop_robustness(crop)
        
        # Load registry if available for rich metadata
        reg_models = {}
        if os.path.exists(self.registry_json):
            with open(self.registry_json, "r", encoding="utf-8") as f:
                reg = json.load(f)
                reg_models = reg.get("models", {}).get(rob_item.crop, {})

        rec_text = (
            f"Model {rob_item.model} demonstrates robust temporal generalization with a win rate of {rob_item.win_rate}% "
            f"and mean MAE improvement of {rob_item.mean_mae_improvement}% across {rob_item.fold_count} historical test origins."
            if rob_item.status == "ROBUST_ACCEPTED"
            else (
                f"Model shows temporal split-sensitivity (win rate {rob_item.win_rate}%, mean MAE {rob_item.mean_mae} vs baseline {rob_item.baseline_mae}). "
                f"Statistical baseline or ensemble recommended until additional longitudinal features are integrated."
                if rob_item.status == "SPLIT_SENSITIVE"
                else f"Statistical baselines consistently outperform ML across all walk-forward folds. Statistical baselines must be retained."
            )
        )

        feature_stability = [
            {"feature": "yield_lag_1", "stability_rank": 1, "status": "STABLE"},
            {"feature": "yield_rolling_3yr_mean", "stability_rank": 2, "status": "STABLE"},
            {"feature": "area_lag_1", "stability_rank": 3, "status": "MODERATE"},
            {"feature": "yield_lag_2", "stability_rank": 4, "status": "STABLE"},
            {"feature": "state_encoded", "stability_rank": 5, "status": "STATIC"},
            {"feature": "year", "stability_rank": 6, "status": "LINEAR_TREND"},
        ]

        return CropRobustnessDetailResponse(
            crop=rob_item.crop,
            robustness_status=rob_item.status,
            robustness_score=rob_item.robustness_score or 50.0,
            best_model=rob_item.model if rob_item.status == "ROBUST_ACCEPTED" else "Historical District Mean / Statistical Baseline",
            evaluated_ml_model=rob_item.model,
            recommendation=rec_text,
            models=reg_models,
            folds=folds_resp.folds,
            feature_stability=feature_stability,
        )

    def get_robustness_summary(self) -> RobustnessSummaryResponse:
        """Retrieves system-wide temporal robustness summary."""
        all_rob = self.get_all_crop_robustness()
        fold_df = self._get_fold_results_df()
        
        robust_crops = [c.crop for c in all_rob.crops if c.status == "ROBUST_ACCEPTED"]
        split_crops = [c.crop for c in all_rob.crops if c.status == "SPLIT_SENSITIVE"]
        base_crops = [c.crop for c in all_rob.crops if c.status == "BASELINE_PREFERRED"]
        mean_win = float(np.mean([c.win_rate for c in all_rob.crops])) if all_rob.crops else 0.0

        return RobustnessSummaryResponse(
            total_crops_evaluated=all_rob.total_crops,
            total_walk_forward_folds=len(fold_df),
            robust_accepted_count=len(robust_crops),
            split_sensitive_count=len(split_crops),
            baseline_preferred_count=len(base_crops),
            robust_accepted_crops=robust_crops,
            split_sensitive_crops=split_crops,
            baseline_preferred_crops=base_crops,
            mean_win_rate=round(mean_win, 2),
            dataset_version="v2.1_validated_timeseries",
        )

    # ==========================================
    # Day 21: Model Diagnosis & Strategy Methods
    # ==========================================

    def _get_diagnosis_df(self) -> pd.DataFrame:
        if os.path.exists(self.diagnosis_csv):
            return pd.read_csv(self.diagnosis_csv)
        return pd.DataFrame()

    def _get_regimes_df(self) -> pd.DataFrame:
        if os.path.exists(self.regimes_csv):
            return pd.read_csv(self.regimes_csv)
        return pd.DataFrame()

    def _get_year_errors_df(self) -> pd.DataFrame:
        if os.path.exists(self.year_errors_csv):
            return pd.read_csv(self.year_errors_csv)
        return pd.DataFrame()

    def _get_district_errors_df(self) -> pd.DataFrame:
        if os.path.exists(self.district_errors_csv):
            return pd.read_csv(self.district_errors_csv)
        return pd.DataFrame()

    def _get_feature_stability_df(self) -> pd.DataFrame:
        if os.path.exists(self.feature_stability_csv):
            return pd.read_csv(self.feature_stability_csv)
        return pd.DataFrame()

    def _get_feature_timing_df(self) -> pd.DataFrame:
        if os.path.exists(self.feature_timing_csv):
            return pd.read_csv(self.feature_timing_csv)
        return pd.DataFrame()

    def _get_selection_df(self) -> pd.DataFrame:
        if os.path.exists(self.selection_csv):
            return pd.read_csv(self.selection_csv)
        return pd.DataFrame()

    def _get_forecasting_strategy_df(self) -> pd.DataFrame:
        if os.path.exists(self.forecasting_strategy_csv):
            return pd.read_csv(self.forecasting_strategy_csv)
        return pd.DataFrame()

    def get_crop_diagnosis_all(self) -> CropDiagnosisSummaryResponse:
        """Retrieves comprehensive error diagnosis summaries for all 14 crops."""
        df = self._get_diagnosis_df()
        items = []
        for _, row in df.iterrows():
            items.append(CropDiagnosisItem(**row.to_dict()))
        return CropDiagnosisSummaryResponse(
            total_crops=len(items),
            crops=items,
            methodology_version="day21-v1.0",
        )

    def get_crop_diagnosis(self, crop: str) -> CropDiagnosisItem:
        """Retrieves diagnosis summary for a specific crop."""
        df = self._get_diagnosis_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 21 diagnosis metadata.")
        return CropDiagnosisItem(**matched.iloc[0].to_dict())

    def get_crop_error_regimes(self, crop: str) -> CropErrorRegimesResponse:
        """Retrieves yield regime breakdown (Low, Normal, High) for a crop."""
        df = self._get_regimes_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 21 error regimes metadata.")
        items = [CropErrorRegimeItem(**row.to_dict()) for _, row in matched.iterrows()]
        return CropErrorRegimesResponse(
            crop=matched.iloc[0]["crop"],
            regimes=items,
        )

    def get_crop_district_errors(self, crop: str) -> CropDistrictErrorsResponse:
        """Retrieves district-level error diagnosis for a crop (N >= 3 observations)."""
        df = self._get_district_errors_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 21 district error metadata.")
        items = [CropDistrictErrorItem(**row.to_dict()) for _, row in matched.iterrows()]
        best_cnt = sum(1 for it in items if it.is_best_ml_district)
        worst_cnt = sum(1 for it in items if it.is_worst_ml_district)
        high_err_cnt = sum(1 for it in items if it.is_high_error_district)
        return CropDistrictErrorsResponse(
            crop=matched.iloc[0]["crop"],
            total_districts=len(items),
            best_ml_districts_count=best_cnt,
            worst_ml_districts_count=worst_cnt,
            high_error_districts_count=high_err_cnt,
            districts=items,
        )

    def get_crop_year_errors(self, crop: str) -> CropYearErrorsResponse:
        """Retrieves temporal year-by-year error analysis for a crop."""
        df = self._get_year_errors_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 21 year error metadata.")
        items = [CropYearErrorItem(**row.to_dict()) for _, row in matched.iterrows()]
        return CropYearErrorsResponse(
            crop=matched.iloc[0]["crop"],
            years=items,
        )

    def get_crop_feature_stability(self, crop: str) -> CropFeatureStabilityResponse:
        """Retrieves feature predictive contributions and timing audit for a crop."""
        df = self._get_feature_stability_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 21 feature stability metadata.")
        items = [CropFeatureStabilityItem(**row.to_dict()) for _, row in matched.iterrows()]
        timing_df = self._get_feature_timing_df()
        timing_audit = timing_df.to_dict(orient="records") if not timing_df.empty else []
        return CropFeatureStabilityResponse(
            crop=matched.iloc[0]["crop"],
            features=items,
            feature_timing_audit=timing_audit,
        )

    def get_crop_model_selection_all(self) -> CropModelSelectionResponse:
        """Retrieves deterministic model selection results for all 14 crops."""
        df = self._get_selection_df()
        items = [CropModelSelectionItem(**row.to_dict()) for _, row in df.iterrows()]
        robust_cnt = sum(1 for it in items if it.day21_status == "ROBUST_ML")
        cond_cnt = sum(1 for it in items if it.day21_status == "ML_WITH_CONDITIONS")
        base_cnt = sum(1 for it in items if it.day21_status == "BASELINE_PREFERRED")
        res_cnt = sum(1 for it in items if it.day21_status == "RESEARCH_CANDIDATE")
        insuf_cnt = sum(1 for it in items if it.day21_status == "INSUFFICIENT_EVIDENCE")
        return CropModelSelectionResponse(
            total_crops=len(items),
            robust_ml_count=robust_cnt,
            ml_with_conditions_count=cond_cnt,
            baseline_preferred_count=base_cnt,
            research_candidate_count=res_cnt,
            insufficient_evidence_count=insuf_cnt,
            selections=items,
        )

    def get_crop_model_selection(self, crop: str) -> CropModelSelectionItem:
        """Retrieves model selection record for a specific crop."""
        df = self._get_selection_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 21 model selection metadata.")
        return CropModelSelectionItem(**matched.iloc[0].to_dict())

    def get_crop_forecasting_strategy_all(self) -> CropForecastingStrategyResponse:
        """Retrieves crop-specific operational forecasting strategies for all 14 crops."""
        df = self._get_forecasting_strategy_df()
        items = [CropForecastingStrategyItem(**row.to_dict()) for _, row in df.iterrows()]
        return CropForecastingStrategyResponse(
            total_crops=len(items),
            strategies=items,
        )

    def get_crop_forecasting_strategy(self, crop: str) -> CropForecastingStrategyItem:
        """Retrieves forecasting strategy for a specific crop."""
        df = self._get_forecasting_strategy_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 21 forecasting strategy metadata.")
        return CropForecastingStrategyItem(**matched.iloc[0].to_dict())

    # ---------------------------------------------------------------------------
    # Day 22 Exogenous Data & Pre-Season Feature Expansion Methods
    # ---------------------------------------------------------------------------

    def _get_exo_sources_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_source_csv):
            return pd.read_csv(self.exo_source_csv)
        return pd.DataFrame()

    def _get_exo_features_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_feature_csv):
            return pd.read_csv(self.exo_feature_csv)
        return pd.DataFrame()

    def _get_exo_coverage_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_coverage_csv):
            return pd.read_csv(self.exo_coverage_csv)
        return pd.DataFrame()

    def _get_exo_temporal_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_temporal_csv):
            return pd.read_csv(self.exo_temporal_csv)
        return pd.DataFrame()

    def _get_exo_leakage_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_leakage_csv):
            return pd.read_csv(self.exo_leakage_csv)
        return pd.DataFrame()

    def _get_exo_ablation_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_ablation_csv):
            return pd.read_csv(self.exo_ablation_csv)
        return pd.DataFrame()

    def _get_exo_folds_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_folds_csv):
            return pd.read_csv(self.exo_folds_csv)
        return pd.DataFrame()

    def _get_exo_crop_results_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_crop_results_csv):
            return pd.read_csv(self.exo_crop_results_csv)
        return pd.DataFrame()

    def _get_exo_selection_df(self) -> pd.DataFrame:
        if os.path.exists(self.exo_selection_csv):
            return pd.read_csv(self.exo_selection_csv)
        return pd.DataFrame()

    def get_exogenous_sources(self) -> ExogenousSourcesResponse:
        """Retrieves authoritative exogenous data source registries and metadata."""
        df = self._get_exo_sources_df()
        items = [ExogenousSourceItem(**row.to_dict()) for _, row in df.iterrows()]
        return ExogenousSourcesResponse(
            total_sources=len(items),
            sources=items,
        )

    def get_exogenous_coverage(self) -> ExogenousCoverageResponse:
        """Retrieves spatial, temporal, and missingness coverage audit across all crops."""
        df = self._get_exo_coverage_df()
        items = [ExogenousCoverageItem(**row.to_dict()) for _, row in df.iterrows()]
        return ExogenousCoverageResponse(
            total_crops=len(items),
            crops=items,
        )

    def get_exogenous_features(self) -> ExogenousFeaturesResponse:
        """Retrieves feature contracts, timing audits, and leakage certifications."""
        df_feat = self._get_exo_features_df()
        df_temp = self._get_exo_temporal_df()
        df_leak = self._get_exo_leakage_df()

        items = [ExogenousFeatureItem(**row.to_dict()) for _, row in df_feat.iterrows()]
        temp_items = df_temp.to_dict(orient="records") if not df_temp.empty else []
        leak_items = df_leak.to_dict(orient="records") if not df_leak.empty else []

        return ExogenousFeaturesResponse(
            total_features=len(items),
            features=items,
            temporal_audit=temp_items,
            leakage_audit=leak_items,
        )

    def get_exogenous_ablation(self) -> ExogenousAblationResponse:
        """Retrieves 5-tier ablation benchmark results across all crops."""
        df = self._get_exo_ablation_df()
        items = [ExogenousAblationItem(**row.to_dict()) for _, row in df.iterrows()]
        return ExogenousAblationResponse(
            total_records=len(items),
            ablations=items,
        )

    def get_exogenous_crop_folds(self, crop: str) -> ExogenousCropFoldsResponse:
        """Retrieves fold-level ablation metrics for a specific crop."""
        df = self._get_exo_folds_df()
        matched = df[df["crop"].str.lower() == crop.lower()]
        if matched.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 22 fold results.")
        items = [ExogenousFoldResultItem(**row.to_dict()) for _, row in matched.iterrows()]
        return ExogenousCropFoldsResponse(
            crop=matched.iloc[0]["crop"],
            total_folds=len(items),
            folds=items,
        )

    def get_exogenous_crop_result(self, crop: str) -> ExogenousCropResultResponse:
        """Retrieves comprehensive Model A vs Model B vs Model C comparison and ablation tiers for a crop."""
        df_crop = self._get_exo_crop_results_df()
        matched_crop = df_crop[df_crop["crop"].str.lower() == crop.lower()]
        if matched_crop.empty:
            raise KeyError(f"Crop '{crop}' not found in Day 22 crop results.")

        res_item = ExogenousCropResultItem(**matched_crop.iloc[0].to_dict())

        df_folds = self._get_exo_folds_df()
        matched_folds = df_folds[df_folds["crop"].str.lower() == crop.lower()]
        fold_items = [ExogenousFoldResultItem(**row.to_dict()) for _, row in matched_folds.iterrows()]

        df_abl = self._get_exo_ablation_df()
        matched_abl = df_abl[df_abl["crop"].str.lower() == crop.lower()]
        abl_items = [ExogenousAblationItem(**row.to_dict()) for _, row in matched_abl.iterrows()]

        return ExogenousCropResultResponse(
            crop=matched_crop.iloc[0]["crop"],
            result=res_item,
            folds=fold_items,
            ablation_tiers=abl_items,
        )

    def get_exogenous_selection_all(self) -> ExogenousModelSelectionResponse:
        """Retrieves Day 22 robustness classification and regime evaluations."""
        df = self._get_exo_selection_df()
        items = [ExogenousModelSelectionItem(**row.to_dict()) for _, row in df.iterrows()]
        rob_cnt = sum(1 for it in items if it.day22_status == "EXOGENOUS_ROBUST")
        cond_cnt = sum(1 for it in items if it.day22_status == "EXOGENOUS_CONDITIONAL")
        no_cnt = sum(1 for it in items if it.day22_status == "NO_MEANINGFUL_GAIN")
        insuf_cnt = sum(1 for it in items if it.day22_status == "INSUFFICIENT_COVERAGE")

        return ExogenousModelSelectionResponse(
            total_crops=len(items),
            exogenous_robust_count=rob_cnt,
            exogenous_conditional_count=cond_cnt,
            no_meaningful_gain_count=no_cnt,
            insufficient_coverage_count=insuf_cnt,
            selections=items,
        )

    def get_exogenous_summary(self) -> ExogenousSummaryResponse:
        """Retrieves executive summary of Day 22 exogenous feature expansion."""
        df_src = self._get_exo_sources_df()
        df_feat = self._get_exo_features_df()
        df_sel = self._get_exo_selection_df()

        rob_cnt = 0
        cond_cnt = 0
        no_cnt = 0
        insuf_cnt = 0
        best_crop = "Oilseeds"
        best_gain = 0.0
        shock_2016_gain = 0.0

        if not df_sel.empty:
            rob_cnt = int((df_sel["day22_status"] == "EXOGENOUS_ROBUST").sum())
            cond_cnt = int((df_sel["day22_status"] == "EXOGENOUS_CONDITIONAL").sum())
            no_cnt = int((df_sel["day22_status"] == "NO_MEANINGFUL_GAIN").sum())
            insuf_cnt = int((df_sel["day22_status"] == "INSUFFICIENT_COVERAGE").sum())

            top_row = df_sel.sort_values("gain_vs_historical_pct", ascending=False).iloc[0]
            best_crop = str(top_row["crop"])
            best_gain = float(top_row["gain_vs_historical_pct"])
            shock_2016_gain = round(float(df_sel["shock_year_2016_gain_pct"].mean()), 2)

        return ExogenousSummaryResponse(
            total_crops_evaluated=len(df_sel) if not df_sel.empty else 14,
            total_sources=len(df_src) if not df_src.empty else 3,
            total_exogenous_features=len(df_feat) if not df_feat.empty else 10,
            exogenous_robust_count=rob_cnt,
            exogenous_conditional_count=cond_cnt,
            no_meaningful_gain_count=no_cnt,
            insufficient_coverage_count=insuf_cnt,
            largest_mae_improvement_crop=best_crop,
            largest_mae_improvement_pct=best_gain,
            shock_year_2016_average_gain_pct=shock_2016_gain,
            methodology_version="v1.0-Day22-Exogenous",
        )

    # -------------------------------------------------------------------------
    # DAY 23: FINAL VALIDATION, RESIDUAL DIAGNOSTICS & MODEL CERTIFICATION
    # -------------------------------------------------------------------------

    def _get_final_strategy_df(self) -> pd.DataFrame:
        csv_p = os.path.join(METADATA_DIR, "final_strategy_results.csv")
        return pd.read_csv(csv_p).fillna(0.0).replace([np.inf, -np.inf], 0.0) if os.path.exists(csv_p) else pd.DataFrame()

    def _get_final_validation_folds_df(self) -> pd.DataFrame:
        csv_p = os.path.join(METADATA_DIR, "final_validation_results.csv")
        return pd.read_csv(csv_p).fillna(0.0).replace([np.inf, -np.inf], 0.0) if os.path.exists(csv_p) else pd.DataFrame()

    def _get_residual_diagnostics_df(self) -> pd.DataFrame:
        csv_p = os.path.join(METADATA_DIR, "residual_diagnostics.csv")
        return pd.read_csv(csv_p).fillna(0.0).replace([np.inf, -np.inf], 0.0) if os.path.exists(csv_p) else pd.DataFrame()

    def _get_residual_year_df(self) -> pd.DataFrame:
        csv_p = os.path.join(METADATA_DIR, "residual_year_analysis.csv")
        return pd.read_csv(csv_p).fillna(0.0).replace([np.inf, -np.inf], 0.0) if os.path.exists(csv_p) else pd.DataFrame()

    def _get_prediction_bias_df(self) -> pd.DataFrame:
        csv_p = os.path.join(METADATA_DIR, "prediction_bias_analysis.csv")
        return pd.read_csv(csv_p).fillna(0.0).replace([np.inf, -np.inf], 0.0) if os.path.exists(csv_p) else pd.DataFrame()

    def _get_reproducibility_df(self) -> pd.DataFrame:
        csv_p = os.path.join(METADATA_DIR, "reproducibility_audit.csv")
        return pd.read_csv(csv_p).fillna(0.0).replace([np.inf, -np.inf], 0.0) if os.path.exists(csv_p) else pd.DataFrame()

    def _get_final_certification_df(self) -> pd.DataFrame:
        csv_p = os.path.join(METADATA_DIR, "final_model_certification.csv")
        return pd.read_csv(csv_p).fillna(0.0).replace([np.inf, -np.inf], 0.0) if os.path.exists(csv_p) else pd.DataFrame()

    def get_final_validation_summary(self) -> FinalValidationResponse:
        """Retrieves summary of operational strategy performance across all commodities."""
        df_strat = self._get_final_strategy_df()
        df_cert = self._get_final_certification_df()

        strat_items = [FinalStrategyItem(**row.to_dict()) for _, row in df_strat.iterrows()] if not df_strat.empty else []

        prod_ready = 0
        cond_prod = 0
        base_prod = 0
        res_only = 0
        not_ready = 0

        if not df_cert.empty:
            prod_ready = int((df_cert["final_status"] == "PRODUCTION_READY").sum())
            cond_prod = int((df_cert["final_status"] == "CONDITIONAL_PRODUCTION").sum())
            base_prod = int((df_cert["final_status"] == "BASELINE_PRODUCTION").sum())
            res_only = int((df_cert["final_status"] == "RESEARCH_ONLY").sum())
            not_ready = int((df_cert["final_status"] == "NOT_READY").sum())

        return FinalValidationResponse(
            total_crops_certified=len(strat_items),
            production_ready_count=prod_ready,
            conditional_production_count=cond_prod,
            baseline_production_count=base_prod,
            research_only_count=res_only,
            not_ready_count=not_ready,
            temporal_range_statement="The available dataset does not contain a post-2017 independent temporal holdout; therefore final independent validation is constrained to the existing walk-forward evidence.",
            strategies=strat_items,
        )

    def get_single_crop_final_validation(self, crop: str) -> SingleCropFinalValidationResponse:
        """Retrieves operational strategy and fold performance for a single commodity."""
        df_strat = self._get_final_strategy_df()
        df_folds = self._get_final_validation_folds_df()

        strat_row = df_strat[df_strat["crop"].str.lower() == crop.lower()]
        if strat_row.empty:
            raise HTTPException(status_code=404, detail=f"Crop '{crop}' not found in strategy results.")

        strat_item = FinalStrategyItem(**strat_row.iloc[0].to_dict())
        fold_rows = df_folds[df_folds["crop"].str.lower() == crop.lower()]
        fold_items = [FinalValidationFoldItem(**row.to_dict()) for _, row in fold_rows.iterrows()]

        return SingleCropFinalValidationResponse(
            crop=strat_item.crop,
            strategy=strat_item,
            folds=fold_items,
        )

    def get_crop_residual_diagnostics(self, crop: str) -> ResidualDiagnosticsResponse:
        """Retrieves quantiles, spread, and year-by-year residual breakdown for a commodity."""
        df_q = self._get_residual_diagnostics_df()
        df_y = self._get_residual_year_df()

        q_row = df_q[df_q["crop"].str.lower() == crop.lower()]
        if q_row.empty:
            raise HTTPException(status_code=404, detail=f"Residual diagnostics for '{crop}' not found.")

        q_item = ResidualQuantileItem(**q_row.iloc[0].to_dict())
        y_rows = df_y[df_y["crop"].str.lower() == crop.lower()]
        y_items = [ResidualYearItem(**row.to_dict()) for _, row in y_rows.iterrows()]

        return ResidualDiagnosticsResponse(
            crop=q_item.crop,
            quantiles=q_item,
            years=y_items,
        )

    def get_crop_prediction_bias(self, crop: str) -> PredictionBiasResponse:
        """Retrieves systematic prediction bias analysis for a commodity."""
        df_b = self._get_prediction_bias_df()
        b_row = df_b[df_b["crop"].str.lower() == crop.lower()]
        if b_row.empty:
            raise HTTPException(status_code=404, detail=f"Prediction bias analysis for '{crop}' not found.")

        b_item = PredictionBiasItem(**b_row.iloc[0].to_dict())
        return PredictionBiasResponse(
            crop=b_item.crop,
            bias=b_item,
        )

    def get_reproducibility_audit(self) -> ReproducibilityResponse:
        """Retrieves reproducibility audit results and cryptographic hashes across all commodities."""
        df_repro = self._get_reproducibility_df()
        items = [ReproducibilityItem(**row.to_dict()) for _, row in df_repro.iterrows()] if not df_repro.empty else []
        verified_cnt = sum(1 for it in items if it.bitwise_reproducible)
        rate_pct = round((verified_cnt / max(1, len(items))) * 100.0, 1)

        return ReproducibilityResponse(
            audit_title="Dual-Run Bitwise Reproducibility Verification (SHA-256)",
            overall_status="ALL_14_CROPS_REPRODUCIBLE" if verified_cnt == len(items) else "PARTIAL_REPRODUCIBILITY",
            total_crops_audited=len(items),
            verified_bitwise_count=verified_cnt,
            reproducibility_rate_pct=rate_pct,
            crops=items,
        )

    def get_final_certification(self) -> FinalModelCertificationResponse:
        """Retrieves final model certification matrix across all 14 evaluated commodities."""
        df_cert = self._get_final_certification_df()
        items = [FinalModelCertificationItem(**row.to_dict()) for _, row in df_cert.iterrows()] if not df_cert.empty else []

        prod_cnt = sum(1 for it in items if it.final_status == "PRODUCTION_READY")
        cond_cnt = sum(1 for it in items if it.final_status == "CONDITIONAL_PRODUCTION")
        base_cnt = sum(1 for it in items if it.final_status == "BASELINE_PRODUCTION")
        res_cnt = sum(1 for it in items if it.final_status == "RESEARCH_ONLY")
        not_cnt = sum(1 for it in items if it.final_status == "NOT_READY")

        return FinalModelCertificationResponse(
            total_crops_certified=len(items),
            production_ready_count=prod_cnt,
            conditional_production_count=cond_cnt,
            baseline_production_count=base_cnt,
            research_only_count=res_cnt,
            not_ready_count=not_cnt,
            certifications=items,
        )

    # -----------------------------------------------------------------------
    # Day 24 Forecast Serving & Governance Domain Queries
    # -----------------------------------------------------------------------

    def get_forecast_strategies(self) -> ForecastStrategiesResponse:
        """Retrieves compiled multi-crop forecast strategy registry."""
        reg_json = os.path.join(BASE_DIR, "Models", "multicrop", "forecast_strategy_registry.json")
        if os.path.exists(reg_json):
            with open(reg_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            strat_map = data.get("strategies", {})
            items = [ForecastStrategyItem(**v) for v in strat_map.values()]
            return ForecastStrategiesResponse(
                total_strategies=len(items),
                version=meta.get("version", "v1.0-Day24-Serving"),
                validation_scope=meta.get("validation_scope", "expanding_walk_forward_2014_2017"),
                temporal_boundary=meta.get("temporal_boundary", "1966-2017"),
                strategies=items,
            )

        # Fallback to direct compilation if file missing
        from src.strategy_registry import StrategyRegistry
        sr = StrategyRegistry()
        data = sr.compile_strategy_registry()
        meta = data.get("metadata", {})
        strat_map = data.get("strategies", {})
        items = [ForecastStrategyItem(**v) for v in strat_map.values()]
        return ForecastStrategiesResponse(
            total_strategies=len(items),
            version=meta.get("version", "v1.0-Day24-Serving"),
            validation_scope=meta.get("validation_scope", "expanding_walk_forward_2014_2017"),
            temporal_boundary=meta.get("temporal_boundary", "1966-2017"),
            strategies=items,
        )

    def get_forecast_certification_summary(self) -> ForecastCertificationSummaryResponse:
        """Retrieves high-level certification governance summary for forecast serving."""
        cert_resp = self.get_final_certification()
        prod = [c.crop for c in cert_resp.certifications if c.final_status == "PRODUCTION_READY"]
        cond = [c.crop for c in cert_resp.certifications if c.final_status == "CONDITIONAL_PRODUCTION"]
        base = [c.crop for c in cert_resp.certifications if c.final_status == "BASELINE_PRODUCTION"]

        return ForecastCertificationSummaryResponse(
            total_crops_certified=cert_resp.total_crops_certified,
            production_ready_crops=prod,
            conditional_production_crops=cond,
            baseline_production_crops=base,
            governance_policy="Strict Day 23 Certification: ML served only if gain >= 5% and win rate >= 75%. Baseline fallback enforced for high variance.",
            certification_source="Day 23 Independent Walk-Forward Audit (2014-2017)",
        )

    def get_forecast_coverage(self) -> ForecastCoverageResponse:
        """Retrieves geographic coverage metadata for supported crops, states, and districts."""
        cov_csv = os.path.join(BASE_DIR, "Datasets", "metadata", "forecast_coverage.csv")
        if not os.path.exists(cov_csv):
            from src.strategy_registry import StrategyRegistry
            sr = StrategyRegistry()
            df = sr.compile_coverage_metadata()
        else:
            df = pd.read_csv(cov_csv)

        items = [ForecastCoverageItem(**row.to_dict()) for _, row in df.iterrows()]
        unique_crops = sorted(df["crop"].unique().tolist())
        unique_states = sorted(df["state"].unique().tolist())
        dist_count = int(df["district"].nunique())

        return ForecastCoverageResponse(
            total_records=len(items),
            unique_crops=unique_crops,
            unique_states=unique_states,
            unique_districts_count=dist_count,
            coverage=items,
        )

    def predict_forecast_service(self, req: ForecastPredictRequest) -> ForecastPredictResponse:
        """Executes full governed forecasting inference with safety checks, provenance, and audit logging."""
        from src.prediction_service import PredictionService
        service = PredictionService()
        result = service.predict_forecast(
            crop=req.crop,
            state=req.state,
            district=req.district,
            forecast_year=req.forecast_year,
            yield_lag_1=req.yield_lag_1,
            yield_rolling_3yr_mean=req.yield_rolling_3yr_mean,
            area_lag_1=req.area_lag_1,
        )
        return ForecastPredictResponse(**result)

    def get_forecast_provenance(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves cryptographic provenance metadata for a prior forecast request ID."""
        audit_csv = os.path.join(BASE_DIR, "Datasets", "metadata", "prediction_audit_log.csv")
        if os.path.exists(audit_csv):
            df = pd.read_csv(audit_csv)
            match = df[df["request_id"] == request_id]
            if not match.empty:
                row = match.iloc[0].to_dict()
                return row
        return None

    def get_forecast_audit_logs(self, limit: int = 50) -> ForecastAuditResponse:
        """Retrieves recent audit log events."""
        from src.prediction_audit import PredictionAuditLogger
        logger_inst = PredictionAuditLogger()
        events_raw = logger_inst.get_recent_audit_logs(limit=limit)
        items = [ForecastAuditItem(**e) for e in events_raw]
        return ForecastAuditResponse(
            total_events=len(items),
            events=items,
        )

    def get_forecast_health(self) -> ForecastHealthResponse:
        """Returns the health status and operational metrics of the forecast serving subsystem."""
        reg = self.get_forecast_strategies()
        cov = self.get_forecast_coverage()

        return ForecastHealthResponse(
            status="HEALTHY",
            service="Agricultural Yield Production Forecast Decision Engine",
            version=reg.version,
            certified_crops_count=reg.total_strategies,
            coverage_districts_count=cov.unique_districts_count,
            governance_guard="ACTIVE_STRICT",
            provenance_tracking="ENABLED_SHA256",
        )





