"""
Agricultural Anomaly Service.

Loads the unsupervised Isolation Forest pipeline, performs multi-dimensional anomaly detection,
computes statistical deviation metrics (z-scores, percentage deviation from historical district baselines),
and generates human-interpretable explanations grounded in empirical data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service, STATE_TO_CODE, CODE_TO_STATE

ANOMALY_FEATURE_COLUMNS = [
    'RICE AREA (1000 ha)',
    'RICE PRODUCTION (1000 tons)',
    'RICE YIELD (Kg per ha)',
    'TOTAL_CROPPED_AREA',
    'RICE_AREA_SHARE',
    'RICE_YIELD_LAG1',
    'RICE_YIELD_ROLL3'
]

class AnomalyService:
    _instance: Optional['AnomalyService'] = None
    _anomaly_model: Any = None
    _cached_dataset_anomalies: Optional[List[Dict[str, Any]]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnomalyService, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """Loads and caches the Isolation Forest pipeline."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        model_path = base_dir / 'Models' / 'agricultural_anomaly_pipeline.pkl'
        if model_path.exists():
            try:
                self._anomaly_model = joblib.load(model_path)
                print(f"[AnomalyService] Loaded anomaly detection pipeline from {model_path}")
            except Exception as e:
                print(f"[AnomalyService Error] Failed to load anomaly detector: {e}")

    def compute_statistical_deviation(
        self,
        value: float,
        history: pd.Series,
        metric_name: str
    ) -> Tuple[Optional[float], Optional[float], List[str]]:
        """
        Computes z-score and percentage deviation against historical distributions.
        Returns: (z_score, pct_deviation, reasons)
        """
        reasons = []
        if len(history) < 2 or history.std() == 0:
            return None, None, ["Insufficient historical observations for reliable deviation analysis."]

        hist_mean = float(history.mean())
        hist_std = float(history.std())
        z_score = float((value - hist_mean) / hist_std)
        pct_dev = float(((value - hist_mean) / hist_mean) * 100) if hist_mean != 0 else 0.0

        if abs(z_score) >= 3.0:
            reasons.append(f"{metric_name} is extreme outlier ({z_score:+.2f} std dev from historical mean: {hist_mean:.1f}).")
        elif abs(z_score) >= 2.0:
            reasons.append(f"{metric_name} deviates substantially ({pct_dev:+.1f}% vs district historical mean of {hist_mean:.1f}).")
        elif abs(pct_dev) >= 30.0:
            reasons.append(f"{metric_name} shows notable departure ({pct_dev:+.1f}%) from historical baseline.")

        return round(z_score, 2), round(pct_dev, 2), reasons

    def detect_anomaly(
        self,
        year: int,
        state_val: Any,
        area: float,
        yield_val: Optional[float] = None,
        production: Optional[float] = None,
        district: Optional[str] = None,
        total_cropped_area: Optional[float] = None,
        rice_area_share: Optional[float] = None,
        rice_yield_lag1: Optional[float] = None,
        rice_yield_roll3: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates an observation for agricultural anomaly status using Isolation Forest
        and statistical distribution comparisons.
        """
        if self._anomaly_model is None:
            self.load_model()

        state_code, state_name = ml_service.resolve_state(state_val)
        df = data_loader.dataframe

        # Resolve district history
        d_matches = df[df['State Code'] == state_code]
        if district and district.strip() and district.lower() != 'unknown':
            sub = d_matches[d_matches['Dist Name'].str.lower() == district.strip().lower()]
            if not sub.empty:
                d_matches = sub

        # Defaults for missing features
        defaults = ml_service.get_district_agronomic_defaults(state_code, district, rice_area=area)

        # Inferred production & yield if only one is passed
        y_val = yield_val if yield_val is not None and yield_val > 0 else (
            (production / area * 1000.0) if production is not None and area > 0 else defaults['rice_yield_lag1']
        )
        p_val = production if production is not None and production >= 0 else (
            (y_val * area / 1000.0) if area > 0 else (defaults['rice_yield_lag1'] * area / 1000.0)
        )

        t_area = float(total_cropped_area) if total_cropped_area is not None and total_cropped_area > 0 else defaults['total_cropped_area']
        t_area = max(t_area, area)
        r_share = float(rice_area_share) if rice_area_share is not None and 0 <= rice_area_share <= 1 else (area / t_area)
        lag1 = float(rice_yield_lag1) if rice_yield_lag1 is not None and rice_yield_lag1 > 0 else defaults['rice_yield_lag1']
        roll3 = float(rice_yield_roll3) if rice_yield_roll3 is not None and rice_yield_roll3 > 0 else defaults['rice_yield_roll3']

        input_df = pd.DataFrame([{
            'RICE AREA (1000 ha)': float(area),
            'RICE PRODUCTION (1000 tons)': float(p_val),
            'RICE YIELD (Kg per ha)': float(y_val),
            'TOTAL_CROPPED_AREA': float(t_area),
            'RICE_AREA_SHARE': float(r_share),
            'RICE_YIELD_LAG1': float(lag1),
            'RICE_YIELD_ROLL3': float(roll3)
        }])

        is_anomaly = False
        raw_score = 0.0
        normalized_score = 0.0

        if self._anomaly_model is not None:
            pred = self._anomaly_model.predict(input_df)[0] # -1 for anomaly, 1 for normal
            is_anomaly = bool(pred == -1)
            raw_score = float(self._anomaly_model.decision_function(input_df)[0])
            # Normalize decision score to [0, 100] where higher = more abnormal
            # Typical raw_score is in range [-0.2, 0.2]
            normalized_score = float(round(np.clip((0.15 - raw_score) / 0.30 * 100.0, 0.0, 100.0), 1))
        else:
            normalized_score = 25.0

        # Detailed Statistical Deviation Analysis
        reasons: List[str] = []
        z_yield, pct_yield, y_reasons = self.compute_statistical_deviation(
            y_val, d_matches['RICE YIELD (Kg per ha)'], "Observed Yield"
        )
        z_area, pct_area, a_reasons = self.compute_statistical_deviation(
            area, d_matches['RICE AREA (1000 ha)'], "Cultivated Rice Area"
        )

        reasons.extend(y_reasons)
        reasons.extend(a_reasons)

        if area < 2.0:
            reasons.append("Small acreage edge case (<2,000 ha): sensitive to survey reporting fluctuations.")

        if is_anomaly and not reasons:
            reasons.append("Multi-dimensional land allocation and productivity combination deviates from typical regional patterns.")
        elif not is_anomaly and not reasons:
            reasons.append("All agricultural variables align within expected regional historical baselines.")

        # Severity Mapping
        if normalized_score >= 75.0 or (z_yield is not None and abs(z_yield) >= 3.0):
            severity = "EXTREME"
        elif normalized_score >= 55.0 or (z_yield is not None and abs(z_yield) >= 2.0):
            severity = "HIGH"
        elif normalized_score >= 35.0 or is_anomaly:
            severity = "MODERATE"
        else:
            severity = "LOW"

        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': normalized_score,
            'raw_decision_score': round(raw_score, 4),
            'severity': severity,
            'yield_z_score': z_yield,
            'yield_deviation_pct': pct_yield,
            'area_z_score': z_area,
            'area_deviation_pct': pct_area,
            'reasons': reasons,
            'state': state_name,
            'district': district or "Regional Average",
            'year': year
        }

    def get_dataset_anomalies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Extracts top agricultural anomalies across the ICRISAT dataset for the intelligence feed.
        """
        if self._cached_dataset_anomalies is not None:
            return self._cached_dataset_anomalies[:limit]

        if self._anomaly_model is None:
            self.load_model()

        df = data_loader.dataframe
        if df.empty:
            return []

        # Prepare dataset features
        area_cols = [c for c in df.columns if 'AREA' in c]
        tot_area = df[area_cols].sum(axis=1)
        rice_share = np.where(tot_area > 0, df['RICE AREA (1000 ha)'] / tot_area, 0.0)

        lag1 = df.groupby('Dist Code')['RICE YIELD (Kg per ha)'].shift(1)
        roll3 = df.groupby('Dist Code')['RICE YIELD (Kg per ha)'].shift(1).rolling(3, min_periods=1).mean()
        state_med = df.groupby('State Code')['RICE YIELD (Kg per ha)'].transform('median')
        lag1 = lag1.fillna(state_med)
        roll3 = roll3.fillna(lag1)

        feat_df = pd.DataFrame({
            'RICE AREA (1000 ha)': df['RICE AREA (1000 ha)'],
            'RICE PRODUCTION (1000 tons)': df['RICE PRODUCTION (1000 tons)'],
            'RICE YIELD (Kg per ha)': df['RICE YIELD (Kg per ha)'],
            'TOTAL_CROPPED_AREA': tot_area,
            'RICE_AREA_SHARE': rice_share,
            'RICE_YIELD_LAG1': lag1,
            'RICE_YIELD_ROLL3': roll3
        })

        if self._anomaly_model is not None:
            preds = self._anomaly_model.predict(feat_df)
            raw_scores = self._anomaly_model.decision_function(feat_df)
        else:
            preds = np.ones(len(df))
            raw_scores = np.zeros(len(df))

        anomalies = []
        for idx in np.where(preds == -1)[0]:
            row = df.iloc[idx]
            r_score = float(raw_scores[idx])
            norm_score = float(round(np.clip((0.15 - r_score) / 0.30 * 100.0, 0.0, 100.0), 1))

            hist_dist = df[df['Dist Code'] == row['Dist Code']]['RICE YIELD (Kg per ha)']
            h_mean = float(hist_dist.mean()) if len(hist_dist) > 0 else float(row['RICE YIELD (Kg per ha)'])
            pct_dev = ((row['RICE YIELD (Kg per ha)'] - h_mean) / h_mean * 100) if h_mean > 0 else 0.0

            if abs(pct_dev) >= 50.0:
                reason = f"Severe yield anomaly: {pct_dev:+.1f}% deviation vs district average ({h_mean:.0f} kg/ha)"
            elif row['RICE AREA (1000 ha)'] < 2.0:
                reason = f"Small acreage volatility: {row['RICE AREA (1000 ha)']:.1f}k ha with volatile yield reporting"
            else:
                reason = f"Multi-variable structural anomaly across production & land allocation (score: {norm_score:.1f})"

            anomalies.append({
                'id': f"anom_{row.get('Dist Code', idx)}_{row.get('Year', 2017)}",
                'state': str(row['State Name']),
                'district': str(row['Dist Name']),
                'year': int(row['Year']),
                'area': float(round(row['RICE AREA (1000 ha)'], 2)),
                'production': float(round(row['RICE PRODUCTION (1000 tons)'], 2)),
                'yield_val': float(round(row['RICE YIELD (Kg per ha)'], 2)),
                'anomaly_score': norm_score,
                'severity': "EXTREME" if norm_score >= 75.0 else ("HIGH" if norm_score >= 55.0 else "MODERATE"),
                'reason': reason,
                'yield_deviation_pct': round(pct_dev, 1)
            })

        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        self._cached_dataset_anomalies = anomalies
        return anomalies[:limit]

    def get_anomaly_feed(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Alias for get_dataset_anomalies."""
        return self.get_dataset_anomalies(limit=limit)

anomaly_service = AnomalyService()
