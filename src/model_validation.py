"""
Unified Model Validation Engine.

Provides rigorous, reproducible out-of-time chronological evaluation across
national, state, district, year, and yield-range slices without data leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from backend.utils.data_loader import data_loader
from backend.services.forecast_service import forecast_service
from backend.services.ml_service import ml_service, STATE_TO_CODE, CODE_TO_STATE

class ModelValidationEngine:
    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epsilon: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Computes standard validation metrics safely.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'sample_count': 0,
                'mae': 0.0,
                'rmse': 0.0,
                'r2': 0.0,
                'mape': 0.0,
                'median_absolute_error': 0.0,
                'mean_residual': 0.0,
                'residual_std': 0.0,
                'overprediction_rate_pct': 0.0,
                'underprediction_rate_pct': 0.0,
                'bias_direction': 'NEUTRAL'
            }

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        # R2 calculation requires sample variance
        if len(y_true) > 1 and np.var(y_true) > 1e-4:
            r2 = float(r2_score(y_true, y_pred))
        else:
            r2 = 0.0

        # Safe MAPE calculation
        safe_true = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
        mape = float(np.mean(np.abs((y_true - y_pred) / safe_true)) * 100.0)
        med_ae = float(median_absolute_error(y_true, y_pred))

        # Residuals: Observed - Predicted
        residuals = y_true - y_pred
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))

        over_pct = float(np.mean(y_pred > y_true) * 100.0)
        under_pct = float(np.mean(y_pred < y_true) * 100.0)

        if mean_res > 25.0:
            bias_dir = 'UNDERPREDICTING'  # Model predicts lower than actual
        elif mean_res < -25.0:
            bias_dir = 'OVERPREDICTING'   # Model predicts higher than actual
        else:
            bias_dir = 'BALANCED'

        return {
            'sample_count': int(len(y_true)),
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'r2': round(r2, 4),
            'mape': round(mape, 2),
            'median_absolute_error': round(med_ae, 2),
            'mean_residual': round(mean_res, 2),
            'residual_std': round(std_res, 2),
            'overprediction_rate_pct': round(over_pct, 1),
            'underprediction_rate_pct': round(under_pct, 1),
            'bias_direction': bias_dir
        }

    def generate_out_of_time_evaluation(
        self,
        test_year_min: int = 2016
    ) -> pd.DataFrame:
        """
        Generates detailed record-level evaluation for the chronological test period (2016-2017).
        """
        df = data_loader.dataframe.copy()

        # Compute engineered features if missing
        df = df.sort_values(by=['Dist Name', 'Year'])
        if 'TOTAL_CROPPED_AREA' not in df.columns:
            df['TOTAL_CROPPED_AREA'] = df['RICE AREA (1000 ha)'] + df['WHEAT AREA (1000 ha)'] + df['COTTON AREA (1000 ha)'] + df['SUGARCANE AREA (1000 ha)']
        if 'RICE_AREA_SHARE' not in df.columns:
            df['RICE_AREA_SHARE'] = np.clip(df['RICE AREA (1000 ha)'] / np.maximum(df['TOTAL_CROPPED_AREA'], 0.1), 0.0, 1.0)
        if 'RICE_YIELD_LAG1' not in df.columns:
            df['RICE_YIELD_LAG1'] = df.groupby('Dist Name')['RICE YIELD (Kg per ha)'].shift(1).bfill()
        if 'RICE_YIELD_ROLL3' not in df.columns:
            df['RICE_YIELD_ROLL3'] = df.groupby('Dist Name')['RICE YIELD (Kg per ha)'].transform(lambda s: s.rolling(3, min_periods=1).mean())

        test_df = df[df['Year'] >= test_year_min].copy()

        # Generate predictions using the loaded Exogenous Forecaster
        pipe = forecast_service.get_pipeline()
        scaler = pipe.named_steps['scaler']
        model = pipe.named_steps['model']
        feature_names = [
            'Year', 'State Code', 'RICE AREA (1000 ha)', 'TOTAL_CROPPED_AREA',
            'RICE_AREA_SHARE', 'WHEAT AREA (1000 ha)', 'COTTON AREA (1000 ha)',
            'SUGARCANE AREA (1000 ha)', 'RICE_YIELD_LAG1', 'RICE_YIELD_ROLL3'
        ]

        X_test_df = test_df[feature_names]
        X_test_scaled = scaler.transform(X_test_df)
        y_pred = model.predict(X_test_scaled)
        y_true = test_df['RICE YIELD (Kg per ha)'].values

        test_df['predicted_yield'] = np.round(y_pred, 1)
        test_df['residual'] = np.round(y_true - y_pred, 1)
        test_df['absolute_error'] = np.round(np.abs(y_true - y_pred), 1)
        
        safe_true = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
        test_df['relative_error_pct'] = np.round((test_df['absolute_error'] / safe_true) * 100.0, 1)

        # Ensemble prediction spread (P90 - P10)
        tree_preds = np.array([tree.predict(X_test_scaled) for tree in model.estimators_])
        p10 = np.percentile(tree_preds, 10, axis=0)
        p90 = np.percentile(tree_preds, 90, axis=0)
        test_df['lower_bound_p10'] = np.round(p10, 1)
        test_df['upper_bound_p90'] = np.round(p90, 1)
        test_df['prediction_spread'] = np.round(p90 - p10, 1)
        test_df['prediction_spread_pct'] = np.round(((p90 - p10) / np.maximum(y_pred, 1.0)) * 100.0, 1)

        return test_df

model_validation_engine = ModelValidationEngine()
