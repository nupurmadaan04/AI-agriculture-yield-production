"""
Agricultural Feature Drift & Distribution Shift Detection Engine.

Quantifies distribution shifts between the reference training distribution (2010–2015)
and the out-of-time evaluation distribution (2016–2017) using Population Stability Index (PSI)
and Kolmogorov-Smirnov (KS) statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from backend.utils.data_loader import data_loader

MONITORED_DRIFT_FEATURES = [
    'RICE AREA (1000 ha)',
    'TOTAL_CROPPED_AREA',
    'RICE_AREA_SHARE',
    'RICE_YIELD_LAG1',
    'RICE_YIELD_ROLL3',
    'WHEAT AREA (1000 ha)',
    'COTTON AREA (1000 ha)',
    'SUGARCANE AREA (1000 ha)',
    'RICE YIELD (Kg per ha)'
]

class ModelDriftEngine:
    @staticmethod
    def calculate_psi(
        reference: np.ndarray,
        target: np.ndarray,
        num_bins: int = 10,
        epsilon: float = 1e-4
    ) -> float:
        """
        Calculates Population Stability Index (PSI) between two continuous distributions.
        """
        if len(reference) == 0 or len(target) == 0:
            return 0.0

        # Create quantile bins from reference
        quantiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(reference, quantiles)
        bins = np.unique(bins)

        if len(bins) < 2:
            return 0.0

        # Calculate proportions in each bin
        ref_counts, _ = np.histogram(reference, bins=bins)
        tgt_counts, _ = np.histogram(target, bins=bins)

        ref_pct = (ref_counts + epsilon) / (len(reference) + epsilon * len(ref_counts))
        tgt_pct = (tgt_counts + epsilon) / (len(target) + epsilon * len(tgt_counts))

        psi = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
        return float(max(0.0, psi))

    def detect_drift(self) -> Dict[str, Any]:
        """
        Analyzes feature drift across the training and evaluation splits.
        """
        df = data_loader.dataframe.copy()

        # Compute lag and area share features if not present in raw df
        df = df.sort_values(by=['Dist Name', 'Year'])
        if 'TOTAL_CROPPED_AREA' not in df.columns:
            df['TOTAL_CROPPED_AREA'] = df['RICE AREA (1000 ha)'] + df['WHEAT AREA (1000 ha)'] + df['COTTON AREA (1000 ha)'] + df['SUGARCANE AREA (1000 ha)']
        if 'RICE_AREA_SHARE' not in df.columns:
            df['RICE_AREA_SHARE'] = np.clip(df['RICE AREA (1000 ha)'] / np.maximum(df['TOTAL_CROPPED_AREA'], 0.1), 0.0, 1.0)
        if 'RICE_YIELD_LAG1' not in df.columns:
            df['RICE_YIELD_LAG1'] = df.groupby('Dist Name')['RICE YIELD (Kg per ha)'].shift(1).bfill()
        if 'RICE_YIELD_ROLL3' not in df.columns:
            df['RICE_YIELD_ROLL3'] = df.groupby('Dist Name')['RICE YIELD (Kg per ha)'].transform(lambda s: s.rolling(3, min_periods=1).mean())

        ref_df = df[df['Year'] <= 2015]
        eval_df = df[df['Year'] >= 2016]

        feature_diagnostics = []
        drift_counts = {'NORMAL': 0, 'WATCH': 0, 'DRIFT_DETECTED': 0}

        for feat in MONITORED_DRIFT_FEATURES:
            if feat not in df.columns:
                continue

            ref_vals = ref_df[feat].dropna().values
            eval_vals = eval_df[feat].dropna().values

            if len(ref_vals) == 0 or len(eval_vals) == 0:
                continue

            psi = self.calculate_psi(ref_vals, eval_vals, num_bins=10)
            ks_res = ks_2samp(ref_vals, eval_vals)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)

            ref_mean = float(np.mean(ref_vals))
            eval_mean = float(np.mean(eval_vals))
            mean_shift_pct = float(((eval_mean - ref_mean) / max(abs(ref_mean), 1e-4)) * 100.0)

            ref_std = float(np.std(ref_vals))
            eval_std = float(np.std(eval_vals))

            if psi < 0.10:
                status = 'NORMAL'
                drift_counts['NORMAL'] += 1
            elif psi < 0.25:
                status = 'WATCH'
                drift_counts['WATCH'] += 1
            else:
                status = 'DRIFT_DETECTED'
                drift_counts['DRIFT_DETECTED'] += 1

            feature_diagnostics.append({
                'feature_name': feat,
                'psi_score': round(psi, 4),
                'ks_statistic': round(ks_stat, 4),
                'ks_pvalue': round(ks_pval, 4),
                'reference_mean': round(ref_mean, 2),
                'evaluation_mean': round(eval_mean, 2),
                'mean_shift_pct': round(mean_shift_pct, 1),
                'reference_std': round(ref_std, 2),
                'evaluation_std': round(eval_std, 2),
                'status': status
            })

        overall_status = 'DRIFT_DETECTED' if drift_counts['DRIFT_DETECTED'] > 0 else 'WATCH' if drift_counts['WATCH'] > 1 else 'NORMAL'

        return {
            'reference_period': '2010–2015 (Training Baseline)',
            'evaluation_period': '2016–2017 (Out-of-Time Test Set)',
            'reference_samples': len(ref_df),
            'evaluation_samples': len(eval_df),
            'overall_status': overall_status,
            'summary_counts': drift_counts,
            'features': feature_diagnostics,
            'interpretation': (
                "PSI < 0.10 denotes stable distribution (NORMAL). "
                "0.10 <= PSI < 0.25 indicates moderate shift (WATCH). "
                "PSI >= 0.25 denotes significant distribution departure (DRIFT DETECTED). "
                "Drift detection identifies feature distribution divergence and does not automatically imply physical failure."
            )
        }

model_drift_engine = ModelDriftEngine()
