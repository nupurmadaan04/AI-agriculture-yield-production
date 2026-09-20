"""
Temporal Trend Service.

Provides state and district-level temporal trend calculations, slope evaluations,
significance tests, and regional trend distribution matrices.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service, STATE_TO_CODE
from src.trend_analysis import trend_analysis_engine

class TrendService:
    _instance: Optional['TrendService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TrendService, cls).__new__(cls)
        return cls._instance

    def analyze_region_trend(
        self,
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyzes historical yield trajectory for a specific state or district.
        """
        df = data_loader.dataframe
        sub = df.copy()

        state_name = state or "National Panel"
        dist_name = district

        if state and state.strip() and state.lower() != 'all':
            state_code, resolved_name = ml_service.resolve_state(state)
            sub = sub[sub['State Code'] == state_code]
            state_name = resolved_name

        if district and district.strip() and district.lower() != 'all':
            d_sub = sub[sub['Dist Name'].str.lower() == district.strip().lower()]
            if not d_sub.empty:
                sub = d_sub
                dist_name = district

        if sub.empty:
            sub = df

        yearly = sub.groupby('Year')['RICE YIELD (Kg per ha)'].mean().sort_index()
        years = [int(y) for y in yearly.index]
        yields = [float(v) for v in yearly.values]

        analysis = trend_analysis_engine.analyze_series(years, yields)
        analysis['state'] = state_name
        analysis['district'] = dist_name or "All Districts"
        analysis['yearly_series'] = [{'year': y, 'yield': round(val, 1)} for y, val in zip(years, yields)]

        return analysis

    def get_all_states_trends(self) -> List[Dict[str, Any]]:
        """
        Computes trend metrics across all 20 states.
        """
        df = data_loader.dataframe
        results = []

        for s_name, s_code in sorted(STATE_TO_CODE.items()):
            s_data = df[df['State Code'] == s_code]
            if s_data.empty:
                continue

            yearly = s_data.groupby('Year')['RICE YIELD (Kg per ha)'].mean().sort_index()
            years = [int(y) for y in yearly.index]
            yields = [float(v) for v in yearly.values]

            res = trend_analysis_engine.analyze_series(years, yields)
            res['state'] = s_name
            res['state_code'] = s_code
            res['district_count'] = int(s_data['Dist Name'].nunique())
            res['avg_yield'] = float(round(s_data['RICE YIELD (Kg per ha)'].mean(), 1))
            results.append(res)

        # Sort descending by Theil-Sen slope
        results.sort(key=lambda x: x['theil_sen_slope'], reverse=True)
        return results

trend_service = TrendService()
