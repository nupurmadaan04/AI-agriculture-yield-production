"""
Change Detection Service.

Coordinates CUSUM shifts, trend-breaks, and volatility transitions across states and districts.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import pandas as pd
from backend.utils.data_loader import data_loader
from backend.services.ml_service import ml_service
from src.change_detection import change_detection_engine


class ChangeDetectionService:
    _instance: Optional['ChangeDetectionService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChangeDetectionService, cls).__new__(cls)
        return cls._instance

    def analyze_region_change(
        self,
        state: Optional[str] = "Punjab",
        district: Optional[str] = None,
        metric: str = "yield"
    ) -> Dict[str, Any]:
        """
        Executes CUSUM and trend-break analysis for a region.
        """
        df = data_loader.dataframe
        target_state = state or "Punjab"
        state_code, state_name = ml_service.resolve_state(target_state)

        sub = df[df["State Code"] == state_code]
        if district and district.strip() and district.lower() != "all":
            d_sub = sub[sub["Dist Name"].str.lower() == district.strip().lower()]
            if not d_sub.empty:
                sub = d_sub

        if sub.empty:
            sub = df[df["State Code"] == 1]

        col_map = {
            "yield": "RICE YIELD (Kg per ha)",
            "area": "RICE AREA (1000 ha)",
            "production": "RICE PRODUCTION (1000 tons)"
        }
        target_col = col_map.get(metric.lower(), "RICE YIELD (Kg per ha)")

        grouped = sub.groupby("Year")[target_col].mean().reset_index()
        years = [int(y) for y in grouped["Year"].tolist()]
        values = [float(v) for v in grouped[target_col].tolist()]

        res = change_detection_engine.analyze_series_changes(
            years=years,
            values=values,
            metric_name=metric
        )
        res["location"] = f"{state_name}" + (f" - {district}" if district else "")
        res["state"] = state_name
        res["district"] = district or "State Average"

        return res


change_detection_service = ChangeDetectionService()
