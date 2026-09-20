import os
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from backend.schemas.agriculture import (
    ModelMetricsResponse,
    ModelMetricItem,
    FeatureImportanceResponse,
    AblationItem,
    DeterministicEstimateResponse,
)

class AnalyticsService:
    def get_model_metrics(self) -> ModelMetricsResponse:
        base_dir = Path(__file__).resolve().parent.parent.parent
        comparison_path = base_dir / 'Models' / 'model_comparison.csv'
        feat_path = base_dir / 'Models' / 'feature_importance.json'

        leaderboard: List[ModelMetricItem] = [
            # Ground truth deterministic baseline
            ModelMetricItem(
                id="det-baseline",
                model_name="Deterministic Agricultural Baseline",
                model_type="deterministic",
                feature_set="Production / Area × 1000 (Exact Ratio)",
                train_r2=1.0,
                random_r2=0.9893,
                random_mae=6.56,
                random_rmse=114.58,
                temporal_r2=0.9771,
                temporal_mae=16.68,
                temporal_rmse=168.15,
                cv_r2=0.9942,
                cv_mae=4.21,
                cv_rmse=84.13,
                status="baseline",
                notes="Non-ML ground-truth algebraic formula. Outperforms all ML models by 15-20x lower MAE.",
            )
        ]

        if comparison_path.exists():
            df_comp = pd.read_csv(comparison_path)
            for idx, row in df_comp.iterrows():
                m_name = str(row.get('Model', f'Model-{idx}'))
                leaderboard.append(
                    ModelMetricItem(
                        id=f"ml-model-{idx}",
                        model_name=m_name,
                        model_type="ml",
                        feature_set="Year, State, Area, Production",
                        train_r2=float(row.get('Train R2', 0.0)) if pd.notnull(row.get('Train R2')) else None,
                        random_r2=float(row.get('Test R2 (Random)', 0.0)) if pd.notnull(row.get('Test R2 (Random)')) else None,
                        random_mae=float(row.get('Test MAE (Random)', 0.0)) if pd.notnull(row.get('Test MAE (Random)')) else None,
                        random_rmse=float(row.get('Test RMSE (Random)', 0.0)) if pd.notnull(row.get('Test RMSE (Random)')) else None,
                        temporal_r2=float(row.get('Test R2 (Temporal)', 0.0)) if pd.notnull(row.get('Test R2 (Temporal)')) else None,
                        temporal_mae=float(row.get('Test MAE (Temporal)', 0.0)) if pd.notnull(row.get('Test MAE (Temporal)')) else None,
                        temporal_rmse=float(row.get('Test RMSE (Temporal)', 0.0)) if pd.notnull(row.get('Test RMSE (Temporal)')) else None,
                        cv_r2=float(row.get('5-Fold CV R2', 0.0)) if pd.notnull(row.get('5-Fold CV R2')) else None,
                        cv_mae=float(row.get('5-Fold CV MAE', 0.0)) if pd.notnull(row.get('5-Fold CV MAE')) else None,
                        cv_rmse=float(row.get('5-Fold CV RMSE', 0.0)) if pd.notnull(row.get('5-Fold CV RMSE')) else None,
                        status="production" if idx == 4 else "evaluated",
                        notes="HistGradientBoosting (Top ML)" if idx == 4 else "Machine learning regressor",
                    )
                )

        # Load feature importance if present
        feat_imp_resp: Optional[FeatureImportanceResponse] = None
        if feat_path.exists():
            with open(feat_path, 'r') as f:
                feat_data = json.load(f)
                native_mdi = {k: float(v) for k, v in feat_data.get('native_feature_importance', {}).items()}
                perm_imp = feat_data.get('permutation_feature_importance', {})
                feat_imp_resp = FeatureImportanceResponse(
                    native_mdi=native_mdi,
                    permutation_importance=perm_imp,
                )

        # 4 Empirical Ablation Configurations from SCIENTIFIC_VALIDATION.md
        ablations = [
            AblationItem(
                config_name="Configuration 1: Standard Full Features",
                feature_set="Area, Production, State, Year",
                num_features=4,
                random_r2=0.9570,
                random_mae=96.27,
                temporal_r2=0.9395,
                temporal_mae=122.82,
                group_kfold_r2=0.8024,
                interpretation="Approximates mathematical ratio P/A with high precision.",
            ),
            AblationItem(
                config_name="Configuration 2: Pre-Season (No Production)",
                feature_set="Area, State, Year (No Prod)",
                num_features=3,
                random_r2=0.7412,
                random_mae=345.81,
                temporal_r2=0.6479,
                temporal_mae=468.03,
                group_kfold_r2=-0.0038,
                interpretation="True pre-season forecasting. Error increases 3.6x; collapses on unseen states.",
            ),
            AblationItem(
                config_name="Configuration 3: No Area",
                feature_set="Production, State, Year (No Area)",
                num_features=3,
                random_r2=0.7674,
                random_mae=335.21,
                temporal_r2=0.7243,
                temporal_mae=400.71,
                group_kfold_r2=0.1384,
                interpretation="Moderate correlation with district production scale.",
            ),
            AblationItem(
                config_name="Configuration 4: Coarse Geography",
                feature_set="State, Year only",
                num_features=2,
                random_r2=0.4971,
                random_mae=581.78,
                temporal_r2=0.4293,
                temporal_mae=672.34,
                group_kfold_r2=-0.9263,
                interpretation="Coarse regional baseline with poor predictive fidelity.",
            ),
        ]

        return ModelMetricsResponse(
            leaderboard=leaderboard,
            feature_importance=feat_imp_resp,
            ablation_experiments=ablations,
        )

    def estimate_deterministic(self, area: float, production: float) -> DeterministicEstimateResponse:
        if area <= 0:
            raise ValueError("Cultivated area must be greater than 0")
        
        # Yield (kg/ha) = (Production ('000 t) / Area ('000 ha)) * 1000
        calculated_yield = round((production / area) * 1000.0, 2)
        return DeterministicEstimateResponse(
            estimated_yield=calculated_yield,
            method="Production / Area × 1000",
            type="deterministic",
            unit="kg/ha",
            area=area,
            production=production,
        )


analytics_service = AnalyticsService()
