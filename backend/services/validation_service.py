"""
Validation & Regional Performance Service.

Exposes national, state, and slice-level validation metrics on out-of-time datasets.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.model_validation import model_validation_engine
from backend.services.model_registry_service import model_registry_service

class ValidationService:
    _instance: Optional['ValidationService'] = None
    _cached_eval_df: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ValidationService, cls).__new__(cls)
        return cls._instance

    def get_eval_df(self) -> pd.DataFrame:
        if self._cached_eval_df is None:
            self._cached_eval_df = model_validation_engine.generate_out_of_time_evaluation()
        return self._cached_eval_df

    def get_validation_overview(self) -> Dict[str, Any]:
        """Provides executive out-of-time validation metrics."""
        eval_df = self.get_eval_df()
        metrics = model_validation_engine.evaluate_predictions(
            eval_df['RICE YIELD (Kg per ha)'].values,
            eval_df['predicted_yield'].values
        )

        return {
            'primary_model': 'Exogenous Random Forest Forecaster',
            'evaluation_type': 'Chronological Out-of-Time Test Set',
            'evaluation_period': '2016–2017 (618 district observations)',
            'training_period': '2010–2015 (1,851 district observations)',
            'metrics': metrics,
            'benchmark_comparison': [
                {'model': 'Naive Last Observation (t-1)', 'mae': 377.57, 'rmse': 575.09, 'r2': 0.7319, 'mape': 20.48},
                {'model': 'Historical District Mean', 'mae': 360.58, 'rmse': 530.57, 'r2': 0.7718, 'mape': 17.02},
                {'model': 'Linear Trend Regression', 'mae': 371.61, 'rmse': 538.23, 'r2': 0.7652, 'mape': 19.08},
                {'model': 'HistGradientBoosting', 'mae': 366.78, 'rmse': 524.63, 'r2': 0.7769, 'mape': 18.53},
                {'model': 'Random Forest Forecaster (Selected)', 'mae': 353.01, 'rmse': 513.11, 'r2': 0.7866, 'mape': 18.04}
            ]
        }

    def get_state_performances(self) -> List[Dict[str, Any]]:
        """Computes performance slices across all 20 states."""
        eval_df = self.get_eval_df()
        results = []

        for state_name, s_group in eval_df.groupby('State Name'):
            metrics = model_validation_engine.evaluate_predictions(
                s_group['RICE YIELD (Kg per ha)'].values,
                s_group['predicted_yield'].values
            )
            results.append({
                'state': state_name,
                'sample_count': metrics['sample_count'],
                'mean_observed_yield': round(float(s_group['RICE YIELD (Kg per ha)'].mean()), 1),
                'mean_predicted_yield': round(float(s_group['predicted_yield'].mean()), 1),
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'r2': metrics['r2'],
                'mape': metrics['mape'],
                'mean_residual': metrics['mean_residual'],
                'bias_direction': metrics['bias_direction']
            })

        results.sort(key=lambda x: x['mae'])
        return results

    def get_predictions_scatter(self, limit: int = 150) -> List[Dict[str, Any]]:
        """Returns sample points for Observed vs Predicted scatter chart."""
        eval_df = self.get_eval_df()
        sample = eval_df[['Year', 'State Name', 'Dist Name', 'RICE YIELD (Kg per ha)', 'predicted_yield', 'residual', 'absolute_error']].head(limit)
        
        points = []
        for _, r in sample.iterrows():
            points.append({
                'year': int(r['Year']),
                'state': str(r['State Name']),
                'district': str(r['Dist Name']),
                'observed': float(r['RICE YIELD (Kg per ha)']),
                'predicted': float(r['predicted_yield']),
                'residual': float(r['residual']),
                'absolute_error': float(r['absolute_error'])
            })
        return points

validation_service = ValidationService()
