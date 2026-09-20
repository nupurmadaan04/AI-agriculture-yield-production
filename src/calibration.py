"""
Prediction Spread & Calibration Analysis Engine.

Assesses the empirical relationship between Random Forest ensemble prediction dispersion
(P10–P90 spread) and observed absolute errors across risk/spread decile buckets.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.model_validation import model_validation_engine

class CalibrationEngine:
    def analyze_calibration(self, eval_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Groups predictions into spread buckets to assess calibration behavior.
        """
        if eval_df is None:
            eval_df = model_validation_engine.generate_out_of_time_evaluation()

        spread_pct = eval_df['prediction_spread_pct'].values
        abs_errors = eval_df['absolute_error'].values
        y_pred = eval_df['predicted_yield'].values

        # Group into 5 spread buckets
        bucket_defs = [
            (0.0, 15.0, "Narrow (<15%)"),
            (15.0, 25.0, "Moderate (15–25%)"),
            (25.0, 35.0, "Elevated (25–35%)"),
            (35.0, 50.0, "High (35–50%)"),
            (50.0, 200.0, "Extreme (>50%)")
        ]

        buckets = []
        for b_min, b_max, label in bucket_defs:
            mask = (spread_pct >= b_min) & (spread_pct < b_max)
            sub_err = abs_errors[mask]
            sub_spread = eval_df.loc[mask, 'prediction_spread'].values

            if len(sub_err) > 0:
                mean_err = float(np.mean(sub_err))
                med_err = float(np.median(sub_err))
                mean_sp = float(np.mean(sub_spread))
                mean_y = float(np.mean(y_pred[mask]))
                error_rate = (mean_err / max(mean_y, 1.0)) * 100.0
            else:
                mean_err = 0.0
                med_err = 0.0
                mean_sp = 0.0
                error_rate = 0.0

            buckets.append({
                'bucket_label': label,
                'spread_min_pct': b_min,
                'spread_max_pct': b_max,
                'sample_count': int(np.sum(mask)),
                'percentage_of_test_set': round(float(np.sum(mask) / len(eval_df) * 100.0), 1),
                'mean_spread_kg_ha': round(mean_sp, 1),
                'mean_absolute_error_kg_ha': round(mean_err, 1),
                'median_absolute_error_kg_ha': round(med_err, 1),
                'observed_error_rate_pct': round(error_rate, 1)
            })

        # Calculate Pearson correlation between spread and error
        valid_mask = ~np.isnan(spread_pct) & ~np.isnan(abs_errors)
        if np.sum(valid_mask) > 2:
            corr = float(np.corrcoef(spread_pct[valid_mask], abs_errors[valid_mask])[0, 1])
        else:
            corr = 0.0

        return {
            'total_evaluated_samples': len(eval_df),
            'spread_error_correlation': round(corr, 3),
            'calibration_buckets': buckets,
            'scientific_disclaimer': (
                "P10–P90 represents Random Forest ensemble prediction dispersion across 150 individual decision trees "
                "and is not a formal frequentist confidence interval. A positive correlation indicates that higher tree "
                "disagreement empirically associates with higher absolute prediction errors."
            )
        }

calibration_engine = CalibrationEngine()
