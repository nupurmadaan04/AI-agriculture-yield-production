"""
Agricultural Error & Residual Intelligence Engine.

Performs deep diagnostics on prediction errors, residual distributions,
systemic bias, and regional error concentrations.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.model_validation import model_validation_engine

# Explicit deterministic error severity thresholds
ERROR_THRESHOLD_LOW = 250.0       # kg/ha
ERROR_THRESHOLD_MODERATE = 600.0  # kg/ha

class ErrorAnalysisEngine:
    def analyze_errors(self, eval_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Computes structured error distributions and regional rankings.
        """
        if eval_df is None:
            eval_df = model_validation_engine.generate_out_of_time_evaluation()

        residuals = eval_df['residual'].values
        abs_errors = eval_df['absolute_error'].values

        # 1. Residual Distribution Bins
        hist_counts, bin_edges = np.histogram(residuals, bins=10)
        residual_bins = []
        for i in range(len(hist_counts)):
            residual_bins.append({
                'bin_min': round(float(bin_edges[i]), 1),
                'bin_max': round(float(bin_edges[i+1]), 1),
                'bin_label': f"{int(bin_edges[i])} to {int(bin_edges[i+1])}",
                'count': int(hist_counts[i]),
                'percentage': round(float(hist_counts[i] / len(residuals) * 100.0), 1)
            })

        # 2. Error Severity Tiers
        low_count = int(np.sum(abs_errors < ERROR_THRESHOLD_LOW))
        mod_count = int(np.sum((abs_errors >= ERROR_THRESHOLD_LOW) & (abs_errors < ERROR_THRESHOLD_MODERATE)))
        high_count = int(np.sum(abs_errors >= ERROR_THRESHOLD_MODERATE))

        # 3. Error Percentiles
        percentiles = {
            'p25': round(float(np.percentile(abs_errors, 25)), 1),
            'p50': round(float(np.percentile(abs_errors, 50)), 1),
            'p75': round(float(np.percentile(abs_errors, 75)), 1),
            'p90': round(float(np.percentile(abs_errors, 90)), 1),
            'p95': round(float(np.percentile(abs_errors, 95)), 1),
            'p99': round(float(np.percentile(abs_errors, 99)), 1),
        }

        # 4. Top Largest Absolute Error Outliers
        top_errors = eval_df.sort_values(by='absolute_error', ascending=False).head(10)
        largest_errors = []
        for _, row in top_errors.iterrows():
            largest_errors.append({
                'year': int(row['Year']),
                'state': str(row['State Name']),
                'district': str(row['Dist Name']),
                'observed_yield': float(row['RICE YIELD (Kg per ha)']),
                'predicted_yield': float(row['predicted_yield']),
                'residual': float(row['residual']),
                'absolute_error': float(row['absolute_error']),
                'relative_error_pct': float(row['relative_error_pct'])
            })

        # 5. State Error Rankings
        state_ranks = []
        for s_name, s_group in eval_df.groupby('State Name'):
            s_res = model_validation_engine.evaluate_predictions(
                s_group['RICE YIELD (Kg per ha)'].values,
                s_group['predicted_yield'].values
            )
            state_ranks.append({
                'state': s_name,
                'sample_count': s_res['sample_count'],
                'mean_observed_yield': round(float(s_group['RICE YIELD (Kg per ha)'].mean()), 1),
                'mean_predicted_yield': round(float(s_group['predicted_yield'].mean()), 1),
                'mae': s_res['mae'],
                'rmse': s_res['rmse'],
                'r2': s_res['r2'],
                'mape': s_res['mape'],
                'mean_residual': s_res['mean_residual'],
                'bias_direction': s_res['bias_direction']
            })

        state_ranks.sort(key=lambda x: x['mae'], reverse=True)

        return {
            'total_test_samples': len(eval_df),
            'mean_absolute_error': round(float(np.mean(abs_errors)), 2),
            'median_absolute_error': round(float(np.median(abs_errors)), 2),
            'residual_bins': residual_bins,
            'percentiles': percentiles,
            'severity_breakdown': {
                'low_error_count': low_count,
                'low_error_pct': round(low_count / len(eval_df) * 100.0, 1),
                'moderate_error_count': mod_count,
                'moderate_error_pct': round(mod_count / len(eval_df) * 100.0, 1),
                'high_error_count': high_count,
                'high_error_pct': round(high_count / len(eval_df) * 100.0, 1),
                'thresholds': {
                    'low_threshold': ERROR_THRESHOLD_LOW,
                    'moderate_threshold': ERROR_THRESHOLD_MODERATE
                }
            },
            'largest_errors': largest_errors,
            'state_error_rankings': state_ranks
        }

error_analysis_engine = ErrorAnalysisEngine()
