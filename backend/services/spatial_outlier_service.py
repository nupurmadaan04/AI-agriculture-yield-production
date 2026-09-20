"""
Spatial Outlier Detection Service.

Identifies agricultural districts that depart significantly from their parent
state's empirical distribution using robust within-state z-scores and MAD metrics.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from src.spatial_features import spatial_feature_engine

class SpatialOutlierService:
    _instance: Optional['SpatialOutlierService'] = None
    _cached_features: Optional[pd.DataFrame] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpatialOutlierService, cls).__new__(cls)
        return cls._instance

    def get_district_features(self) -> pd.DataFrame:
        if self._cached_features is None:
            df = data_loader.dataframe
            self._cached_features = spatial_feature_engine.compute_district_spatial_features(df)
        return self._cached_features

    def get_spatial_outliers(
        self,
        state: Optional[str] = None,
        z_threshold: float = 1.8
    ) -> List[Dict[str, Any]]:
        """
        Identifies districts with significant within-state statistical deviation.
        """
        feat_df = self.get_district_features().copy()

        if state and state.strip() and state.lower() != 'all':
            feat_df = feat_df[feat_df['state_name'].str.lower() == state.strip().lower()]

        outliers = []
        for _, row in feat_df.iterrows():
            z_state = abs(row['district_yield_zscore_state'])
            rel_risk = row['state_relative_risk_ratio']
            anom_rate = row['anomaly_rate_pct']

            is_outlier = False
            reasons = []

            if z_state >= z_threshold:
                is_outlier = True
                direction = "above" if row['district_yield_zscore_state'] > 0 else "below"
                reasons.append(f"Yield is {z_state:.2f} std deviations {direction} state average ({row['state_relative_yield_ratio']:.2f}x ratio).")

            if rel_risk >= 2.0:
                is_outlier = True
                reasons.append(f"Yield volatility is {rel_risk:.1f}x higher than state baseline.")

            if anom_rate >= 25.0:
                is_outlier = True
                reasons.append(f"High historical outlier frequency ({anom_rate:.1f}% of surveyed years).")

            if is_outlier:
                outliers.append({
                    'state': row['state_name'],
                    'district': row['district_name'],
                    'yield_kg_ha': row['average_yield_kg_ha'],
                    'state_mean_yield': row['state_mean_yield'],
                    'within_state_zscore': row['district_yield_zscore_state'],
                    'relative_yield_ratio': row['state_relative_yield_ratio'],
                    'relative_risk_ratio': row['state_relative_risk_ratio'],
                    'anomaly_rate_pct': row['anomaly_rate_pct'],
                    'theil_sen_slope': row['trend_theil_sen_slope'],
                    'reasons': reasons,
                    'severity': 'HIGH' if z_state >= 2.5 or rel_risk >= 3.0 else 'MODERATE'
                })

        # Sort descending by absolute z-score
        outliers.sort(key=lambda x: abs(x['within_state_zscore']), reverse=True)
        return outliers

    detect_spatial_outliers = get_spatial_outliers

spatial_outlier_service = SpatialOutlierService()

