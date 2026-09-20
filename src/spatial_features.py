import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.trend_analysis import trend_analysis_engine

class SpatialFeatureEngine:
    @staticmethod
    def compute_district_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes spatial intelligence features for all 311 districts.
        """
        district_records = []

        # Global dataset statistics
        national_mean_yield = float(df['RICE YIELD (Kg per ha)'].mean())
        national_std_yield = float(df['RICE YIELD (Kg per ha)'].std()) or 1.0

        # Group by State & District
        for (state_name, dist_name), dist_df in df.groupby(['State Name', 'Dist Name']):
            state_df = df[df['State Name'] == state_name]
            state_mean_yield = float(state_df['RICE YIELD (Kg per ha)'].mean())
            state_std_yield = float(state_df['RICE YIELD (Kg per ha)'].std()) or 1.0

            dist_mean_yield = float(dist_df['RICE YIELD (Kg per ha)'].mean())
            dist_std_yield = float(dist_df['RICE YIELD (Kg per ha)'].std()) if len(dist_df) > 1 else 0.0
            dist_mean_area = float(dist_df['RICE AREA (1000 ha)'].mean())

            # District z-scores
            z_score_state = (dist_mean_yield - state_mean_yield) / state_std_yield
            z_score_national = (dist_mean_yield - national_mean_yield) / national_std_yield

            # State relative ratios
            rel_yield_ratio = dist_mean_yield / state_mean_yield if state_mean_yield > 0 else 1.0
            state_volatility = state_std_yield / state_mean_yield if state_mean_yield > 0 else 0.1
            dist_volatility = dist_std_yield / dist_mean_yield if dist_mean_yield > 0 else 0.1
            rel_risk_ratio = dist_volatility / state_volatility if state_volatility > 0 else 1.0

            # Trend slope
            yearly = dist_df.groupby('Year')['RICE YIELD (Kg per ha)'].mean().sort_index()
            trend_res = trend_analysis_engine.analyze_series(
                [int(y) for y in yearly.index],
                [float(v) for v in yearly.values]
            )

            # Anomaly rate heuristic (years with >35% drop vs district mean)
            anom_count = sum(1 for y in dist_df['RICE YIELD (Kg per ha)'] if y < dist_mean_yield * 0.65 or y > dist_mean_yield * 1.5)
            anomaly_rate = float(anom_count / len(dist_df)) if len(dist_df) > 0 else 0.0

            district_records.append({
                'state_name': state_name,
                'district_name': dist_name,
                'state_code': int(dist_df['State Code'].iloc[0]),
                'dist_code': int(dist_df['Dist Code'].iloc[0]),
                'observations_count': len(dist_df),
                'average_yield_kg_ha': round(dist_mean_yield, 1),
                'yield_std_dev': round(dist_std_yield, 1),
                'yield_volatility_pct': round(dist_volatility * 100.0, 1),
                'average_area_k_ha': round(dist_mean_area, 1),
                'state_mean_yield': round(state_mean_yield, 1),
                'state_relative_yield_ratio': round(rel_yield_ratio, 3),
                'state_relative_risk_ratio': round(rel_risk_ratio, 3),
                'district_yield_zscore_state': round(z_score_state, 2),
                'district_yield_zscore_national': round(z_score_national, 2),
                'trend_theil_sen_slope': round(trend_res['theil_sen_slope'], 2),
                'trend_direction': trend_res['direction'],
                'anomaly_rate_pct': round(anomaly_rate * 100.0, 1)
            })

        feat_df = pd.DataFrame(district_records)
        return feat_df

spatial_feature_engine = SpatialFeatureEngine()
