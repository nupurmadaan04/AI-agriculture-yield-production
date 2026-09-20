"""
Day 24 Forecast Strategy Router & Inference Dispatcher.
Routes requests to certified ML models, conditional pipelines, or statistical baselines.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("forecast_router")


class ForecastRouter:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.models_dir = self.base_dir / "Models" / "multicrop"
        self.processed_dir = self.base_dir / "Datasets" / "processed"
        self._panel_df: Optional[pd.DataFrame] = None
        self._loaded_models: Dict[str, Any] = {}

    def _get_panel(self) -> pd.DataFrame:
        if self._panel_df is None:
            panel_p = self.processed_dir / "agricultural_panel.csv"
            if panel_p.exists():
                self._panel_df = pd.read_csv(panel_p)
            else:
                self._panel_df = pd.DataFrame()
        return self._panel_df

    def _get_model(self, artifact_name: str) -> Optional[Tuple[Any, Optional[Any]]]:
        if artifact_name not in self._loaded_models:
            p = self.models_dir / artifact_name
            if p.exists():
                try:
                    import joblib
                    loaded = joblib.load(p)
                    if isinstance(loaded, dict):
                        estimator = loaded.get("model") or loaded.get("feature_pipeline")
                        pipeline_helper = loaded.get("feature_pipeline")
                        self._loaded_models[artifact_name] = (estimator, pipeline_helper)
                    else:
                        self._loaded_models[artifact_name] = (loaded, None)
                except Exception as e:
                    logger.warning("Error loading artifact %s: %s", artifact_name, e)
                    return None
            else:
                return None
        return self._loaded_models.get(artifact_name)

    def route_and_predict(
        self,
        crop: str,
        state: str,
        district: str,
        forecast_year: int,
        strategy_meta: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes inference based on certified strategy routing.
        Returns a dictionary with prediction, unit, fallback_used, evidence_type, and route_log.
        """
        cert_status = strategy_meta.get("certification_status", "BASELINE_PRODUCTION")
        panel = self._get_panel()

        # Extract district historical subset
        dist_df = pd.DataFrame()
        yield_col = "yield_kg_ha" if not panel.empty and "yield_kg_ha" in panel.columns else "yield"
        area_col = "area_ha" if not panel.empty and "area_ha" in panel.columns else "area"

        if not panel.empty:
            dist_df = panel[
                (panel["crop"].str.lower() == crop.lower())
                & (panel["state"].str.lower() == state.lower())
                & (panel["district"].str.lower() == district.lower())
                & (panel["year"] < (forecast_year or 2018))
            ]

        # Calculate historical stats for fallback & safety checks
        dist_mean = float(dist_df[yield_col].mean()) if not dist_df.empty and not pd.isna(dist_df[yield_col].mean()) else None
        dist_std = float(dist_df[yield_col].std()) if len(dist_df) > 1 and not pd.isna(dist_df[yield_col].std()) else 0.0
        n_obs = len(dist_df)

        features = features or {}

        # -------------------------------------------------------------
        # 1. PRODUCTION_READY ROUTE (Oilseeds)
        # -------------------------------------------------------------
        if cert_status == "PRODUCTION_READY":
            artifact_name = strategy_meta.get("model_artifact")
            model_info = self._get_model(artifact_name) if artifact_name else None
            estimator, pipe_helper = model_info if model_info else (None, None)

            # Fallback check: if district history is too sparse (< 3) or model unavailable
            if estimator is None or n_obs < 3:
                fallback_pred = dist_mean if dist_mean is not None else 500.0
                return {
                    "prediction": round(fallback_pred, 2),
                    "unit": "kg/ha",
                    "fallback_used": True,
                    "fallback_reason": "Sparse district history (< 3 observations) or model unavailable",
                    "evidence_type": "HISTORICAL_BASELINE_FALLBACK",
                    "selected_route": "SPARSE_DISTRICT_FALLBACK",
                }

            # Prepare 6-feature vector for ML
            lag1 = float(features.get("yield_lag_1", dist_df[yield_col].iloc[-1] if not dist_df.empty else dist_mean))
            lag2 = float(features.get("yield_lag_2", dist_df[yield_col].iloc[-2] if len(dist_df) > 1 else lag1))
            roll3 = float(features.get("yield_rolling_3yr_mean", dist_df[yield_col].tail(3).mean() if not dist_df.empty else dist_mean))
            area_lag1 = float(features.get("area_lag_1", dist_df[area_col].iloc[-1] if not dist_df.empty and area_col in dist_df.columns else 10000.0))
            
            state_code = 0
            if pipe_helper and hasattr(pipe_helper, "state_to_code"):
                state_code = pipe_helper.state_to_code.get(state, 0)

            feat_vector = np.array([[lag1, lag2, roll3, area_lag1, state_code, forecast_year or 2017]])
            try:
                pred_raw = float(estimator.predict(feat_vector)[0])
            except Exception as e:
                logger.warning("Estimator prediction error: %s, using fallback", e)
                pred_raw = dist_mean if dist_mean is not None else 500.0

            pred_clamped = max(10.0, pred_raw)

            return {
                "prediction": round(pred_clamped, 2),
                "unit": "kg/ha",
                "fallback_used": False,
                "fallback_reason": None,
                "evidence_type": "PREDICTED_ML",
                "selected_route": "CERTIFIED_MACHINE_LEARNING",
            }

        # -------------------------------------------------------------
        # 2. CONDITIONAL_PRODUCTION ROUTE (Sugarcane)
        # -------------------------------------------------------------
        elif cert_status == "CONDITIONAL_PRODUCTION":
            artifact_name = strategy_meta.get("model_artifact")
            model_info = self._get_model(artifact_name) if artifact_name else None
            estimator, pipe_helper = model_info if model_info else (None, None)

            if estimator is None or n_obs < 3:
                fallback_pred = dist_mean if dist_mean is not None else 1400.0
                return {
                    "prediction": round(fallback_pred, 2),
                    "unit": "kg/ha",
                    "fallback_used": True,
                    "fallback_reason": "Sparse district history (< 3 observations) or model unavailable",
                    "evidence_type": "HISTORICAL_BASELINE_FALLBACK",
                    "selected_route": "SPARSE_DISTRICT_FALLBACK",
                }

            lag1 = float(features.get("yield_lag_1", dist_df[yield_col].iloc[-1] if not dist_df.empty else dist_mean))
            lag2 = float(features.get("yield_lag_2", dist_df[yield_col].iloc[-2] if len(dist_df) > 1 else lag1))
            roll3 = float(features.get("yield_rolling_3yr_mean", dist_df[yield_col].tail(3).mean() if not dist_df.empty else dist_mean))
            area_lag1 = float(features.get("area_lag_1", dist_df[area_col].iloc[-1] if not dist_df.empty and area_col in dist_df.columns else 50000.0))

            state_code = 0
            if pipe_helper and hasattr(pipe_helper, "state_to_code"):
                state_code = pipe_helper.state_to_code.get(state, 0)

            feat_vector = np.array([[lag1, lag2, roll3, area_lag1, state_code, forecast_year or 2017]])
            try:
                pred_raw = float(estimator.predict(feat_vector)[0])
            except Exception as e:
                logger.warning("Estimator prediction error: %s, using fallback", e)
                pred_raw = dist_mean if dist_mean is not None else 1400.0

            # Apply 3-sigma variance clipping
            fallback_applied = False
            pred_final = pred_raw
            if dist_mean is not None and dist_std > 0:
                lower_bound = max(10.0, dist_mean - 3.0 * dist_std)
                upper_bound = dist_mean + 3.0 * dist_std
                if pred_raw < lower_bound or pred_raw > upper_bound:
                    pred_final = np.clip(pred_raw, lower_bound, upper_bound)
                    fallback_applied = True

            return {
                "prediction": round(float(pred_final), 2),
                "unit": "kg/ha",
                "fallback_used": fallback_applied,
                "fallback_reason": "3-sigma variance clipping applied to bounds" if fallback_applied else None,
                "evidence_type": "PREDICTED_ML_CLIPPED" if fallback_applied else "PREDICTED_ML",
                "selected_route": "CONDITIONAL_VARIANCE_CLIPPED_ML",
            }

        # -------------------------------------------------------------
        # 3. BASELINE_PRODUCTION ROUTE (12 Commodities: Chickpea, Rice, etc.)
        # -------------------------------------------------------------
        else:
            if dist_mean is not None:
                final_val = dist_mean
                fallback_used = False
                fallback_reason = None
            elif not panel.empty:
                state_df = panel[
                    (panel["crop"].str.lower() == crop.lower())
                    & (panel["state"].str.lower() == state.lower())
                ]
                final_val = float(state_df[yield_col].mean()) if not state_df.empty else 300.0
                fallback_used = True
                fallback_reason = "District history missing; state mean baseline used."
            else:
                final_val = 300.0
                fallback_used = True
                fallback_reason = "Panel missing; default global baseline applied."

            return {
                "prediction": round(final_val, 2),
                "unit": "kg/ha",
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "evidence_type": "HISTORICAL_BASELINE",
                "selected_route": "CERTIFIED_STATISTICAL_BASELINE",
            }
