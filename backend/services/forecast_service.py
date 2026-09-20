"""
Temporal Forecasting Service.

Loads the trained forecasting model pipeline and generates multi-horizon predictions
(1, 2, and 3-year lookaheads) at national, state, and district granularities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service, STATE_TO_CODE, CODE_TO_STATE

class ForecastService:
    _instance: Optional['ForecastService'] = None
    _pipeline: Any = None
    _metadata: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ForecastService, cls).__new__(cls)
        return cls._instance

    def load_model(self) -> None:
        """Loads and caches the forecasting model pipeline and benchmark metadata."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        pipe_path = base_dir / 'Models' / 'forecasting_pipeline.pkl'
        meta_path = base_dir / 'Models' / 'forecasting_model_metadata.json'

        if pipe_path.exists():
            try:
                self._pipeline = joblib.load(pipe_path)
                print(f"[ForecastService] Loaded forecasting pipeline from {pipe_path}")
            except Exception as e:
                print(f"[ForecastService Error] Failed to load forecasting pipeline: {e}")

        if meta_path.exists():
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
            except Exception as e:
                print(f"[ForecastService Error] Failed to load metadata: {e}")

    def get_pipeline(self) -> Any:
        """Returns the loaded forecasting pipeline dictionary."""
        if self._pipeline is None:
            self.load_model()
        return self._pipeline

    def get_benchmark_metadata(self) -> Dict[str, Any]:
        """Returns the out-of-time chronological benchmark metrics."""
        if self._metadata is None:
            self.load_model()
        return self._metadata or {}

    def forecast_region(
        self,
        state_val: Optional[Any] = "Punjab",
        district: Optional[str] = None,
        horizons: List[int] = [1, 2, 3]
    ) -> Dict[str, Any]:
        """
        Generates 1, 2, and 3-year forward forecasts (2018, 2019, 2020) for a state or district.
        """
        if self._pipeline is None:
            self.load_model()

        state_code, state_name = ml_service.resolve_state(state_val)
        df = data_loader.dataframe

        # Resolve district history
        d_matches = df[df['State Code'] == state_code]
        if district and district.strip() and district.lower() != 'all':
            sub = d_matches[d_matches['Dist Name'].str.lower() == district.strip().lower()]
            if not sub.empty:
                d_matches = sub

        if d_matches.empty:
            d_matches = df[df['State Code'] == state_code]

        # Extract latest observed baseline (Year 2017)
        latest_year = int(d_matches['Year'].max())
        latest_data = d_matches[d_matches['Year'] == latest_year]
        if latest_data.empty:
            latest_data = d_matches

        base_area = float(latest_data['RICE AREA (1000 ha)'].median())
        defaults = ml_service.get_district_agronomic_defaults(state_code, district, rice_area=base_area)

        # Historical time-series for reference
        hist_yearly = d_matches.groupby('Year')['RICE YIELD (Kg per ha)'].mean().sort_index()
        historical_series = [{'year': int(y), 'yield': round(float(v), 1)} for y, v in hist_yearly.items()]

        # Multi-horizon autoregressive forecasting
        forecasts = []
        curr_lag1 = defaults['rice_yield_lag1']
        recent_yields = [float(y['yield']) for y in historical_series[-3:]] if len(historical_series) >= 3 else [curr_lag1, curr_lag1, curr_lag1]

        for h in horizons:
            target_year = latest_year + h
            curr_roll3 = float(np.mean(recent_yields[-3:]))

            feat_df = pd.DataFrame([{
                'Year': target_year,
                'State Code': state_code,
                'RICE AREA (1000 ha)': base_area,
                'TOTAL_CROPPED_AREA': defaults['total_cropped_area'],
                'RICE_AREA_SHARE': defaults['rice_area_share'],
                'WHEAT AREA (1000 ha)': defaults['wheat_area'],
                'COTTON AREA (1000 ha)': defaults['cotton_area'],
                'SUGARCANE AREA (1000 ha)': defaults['sugarcane_area'],
                'RICE_YIELD_LAG1': curr_lag1,
                'RICE_YIELD_ROLL3': curr_roll3
            }])

            if self._pipeline is not None:
                point_pred = float(self._pipeline.predict(feat_df)[0])
                # Compute tree ensemble prediction interval
                model = self._pipeline.named_steps['model']
                scaler = self._pipeline.named_steps['scaler']
                scaled_x = scaler.transform(feat_df)
                tree_preds = [tree.predict(scaled_x)[0] for tree in model.estimators_]
                p10 = float(np.percentile(tree_preds, 10))
                p90 = float(np.percentile(tree_preds, 90))
                spread = float(p90 - p10)
            else:
                point_pred = curr_lag1
                p10 = point_pred * 0.85
                p90 = point_pred * 1.15
                spread = p90 - p10

            forecasts.append({
                'horizon_years': h,
                'forecast_year': target_year,
                'predicted_yield': round(point_pred, 1),
                'lower_bound_p10': round(p10, 1),
                'upper_bound_p90': round(p90, 1),
                'prediction_spread': round(spread, 1),
                'uncertainty_pct': round((spread / point_pred * 100.0) if point_pred > 0 else 20.0, 1)
            })

            # Update autoregressive memory
            curr_lag1 = point_pred
            recent_yields.append(point_pred)

        return {
            'state': state_name,
            'district': district or "Regional Average",
            'latest_observed_year': latest_year,
            'latest_observed_yield': round(float(historical_series[-1]['yield']) if historical_series else 3000.0, 1),
            'historical_series': historical_series,
            'forecasts': forecasts,
            'model_name': 'Temporal Exogenous Random Forest Forecaster',
            'disclaimer': 'Forecasts represent model-estimated future values and should not be interpreted as causal predictions or guarantees.'
        }

forecast_service = ForecastService()
