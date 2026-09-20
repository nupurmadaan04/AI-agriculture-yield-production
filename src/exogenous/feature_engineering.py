"""
Agronomic Pre-Season Exogenous Feature Engineering Engine (Day 22).

Constructs agronomically meaningful, pre-season available environmental indicators
(Rainfall anomalies, thermal extremes, moisture/aridity indices) and exports feature registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

FEATURE_REGISTRY_ENTRIES: List[Dict[str, Any]] = [
    {
        "feature": "preseason_rainfall_total",
        "source": "IMD_DISTRICT_MET_SERIES",
        "spatial_level": "district",
        "temporal_resolution": "seasonal_sum (Jan–May)",
        "observation_period": "Jan–May of year t",
        "availability_date": "May 31 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "mm",
        "transformation": "seasonal_sum",
        "leakage_status": "SAFE"
    },
    {
        "feature": "preseason_rainfall_anomaly",
        "source": "IMD_DISTRICT_MET_SERIES",
        "spatial_level": "district",
        "temporal_resolution": "seasonal_pct_departure",
        "observation_period": "Jan–May of year t",
        "availability_date": "May 31 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "%",
        "transformation": "pct_departure_from_30yr_normal",
        "leakage_status": "SAFE"
    },
    {
        "feature": "preseason_temp_mean",
        "source": "IMD_DISTRICT_MET_SERIES",
        "spatial_level": "district",
        "temporal_resolution": "seasonal_mean (Mar–May)",
        "observation_period": "Mar–May of year t",
        "availability_date": "May 31 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "°C",
        "transformation": "seasonal_mean",
        "leakage_status": "SAFE"
    },
    {
        "feature": "preseason_temp_max",
        "source": "IMD_DISTRICT_MET_SERIES",
        "spatial_level": "district",
        "temporal_resolution": "seasonal_max (Mar–May)",
        "observation_period": "Mar–May of year t",
        "availability_date": "May 31 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "°C",
        "transformation": "seasonal_maximum",
        "leakage_status": "SAFE"
    },
    {
        "feature": "preseason_temp_anomaly",
        "source": "IMD_DISTRICT_MET_SERIES",
        "spatial_level": "district",
        "temporal_resolution": "seasonal_departure",
        "observation_period": "Mar–May of year t",
        "availability_date": "May 31 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "°C",
        "transformation": "mean_temp_minus_historical_baseline",
        "leakage_status": "SAFE"
    },
    {
        "feature": "preseason_soil_moisture",
        "source": "NASA_POWER_ERA5_AGROCLIM",
        "spatial_level": "district",
        "temporal_resolution": "monthly_mean (May)",
        "observation_period": "May 1–25 of year t",
        "availability_date": "May 25 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "index [0-1]",
        "transformation": "near_real_time_satellite_saturation_index",
        "leakage_status": "SAFE"
    },
    {
        "feature": "preseason_dry_spell_days",
        "source": "IMD_DISTRICT_MET_SERIES",
        "spatial_level": "district",
        "temporal_resolution": "seasonal_count",
        "observation_period": "Jan–May of year t",
        "availability_date": "May 31 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "days",
        "transformation": "consecutive_dry_days_count",
        "leakage_status": "SAFE"
    },
    {
        "feature": "preseason_aridity_index",
        "source": "NASA_POWER_ERA5_AGROCLIM",
        "spatial_level": "district",
        "temporal_resolution": "seasonal_spei_proxy",
        "observation_period": "Jan–May of year t",
        "availability_date": "May 31 of year t",
        "forecast_origin": "pre-season",
        "lag": 0,
        "unit": "z-score",
        "transformation": "standardized_precipitation_minus_pet",
        "leakage_status": "SAFE"
    },
    {
        "feature": "rainfall_lag1_total",
        "source": "ICRISAT_AGROCLIMATIC_MESONET",
        "spatial_level": "district",
        "temporal_resolution": "annual_sum (t-1)",
        "observation_period": "Full year t-1",
        "availability_date": "Dec 31 of year t-1",
        "forecast_origin": "pre-season",
        "lag": 1,
        "unit": "mm",
        "transformation": "previous_year_annual_precipitation",
        "leakage_status": "SAFE"
    },
    {
        "feature": "irrigation_ratio_lag1",
        "source": "ICRISAT_AGROCLIMATIC_MESONET",
        "spatial_level": "district",
        "temporal_resolution": "annual_ratio (t-1)",
        "observation_period": "Full year t-1",
        "availability_date": "Dec 31 of year t-1",
        "forecast_origin": "pre-season",
        "lag": 1,
        "unit": "ratio [0-1]",
        "transformation": "irrigated_area_divided_by_total_cropped_area",
        "leakage_status": "SAFE"
    }
]


class ExogenousFeatureEngineeringEngine:
    """Engineers agronomically meaningful pre-season features and manages the feature contract."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"

    def export_feature_registry(self) -> Path:
        """Exports the official feature contract registry."""
        df = pd.DataFrame(FEATURE_REGISTRY_ENTRIES)
        out_csv = self.metadata_dir / "exogenous_feature_registry.csv"
        df.to_csv(out_csv, index=False)
        return out_csv

    def engineer_exogenous_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms merged raw weather variables into standardized, modeling-ready features.
        """
        df = merged_df.copy()

        # 1. Rainfall features
        df["preseason_rainfall_total"] = df["preseason_rainfall_mm"].fillna(150.0)
        df["preseason_rainfall_anomaly"] = df["preseason_rainfall_anomaly_pct"].fillna(0.0)

        # 2. Temperature features
        df["preseason_temp_mean"] = df["preseason_temp_mean_c"].fillna(29.0)
        df["preseason_temp_max"] = df["preseason_temp_max_c"].fillna(38.0)
        df["preseason_temp_anomaly"] = df["preseason_temp_anomaly_c"].fillna(0.0)

        # 3. Moisture and Aridity features
        df["preseason_soil_moisture"] = df["preseason_soil_moisture_index"].fillna(0.40)
        df["preseason_dry_spell_days"] = df["preseason_dry_spell_days"].fillna(25)

        # Pre-season aridity index (SPEI proxy = normalized (Rain - Potential Evapo-transpiration proxy))
        pet_proxy = df["preseason_temp_mean"] * 12.0  # Simple Thornthwaite/Hargreaves proxy
        spei_proxy = (df["preseason_rainfall_total"] - pet_proxy) / (pet_proxy + 1e-5)
        df["preseason_aridity_index"] = np.clip(spei_proxy, -3.0, 3.0).round(4)

        # 4. Lagged climate & inputs
        df["rainfall_lag1_total"] = df["rainfall_lag1_annual_mm"].fillna(850.0)
        df["irrigation_ratio_lag1"] = df["irrigation_ratio_lag1"].fillna(0.35)

        return df


if __name__ == "__main__":
    engine = ExogenousFeatureEngineeringEngine()
    reg_p = engine.export_feature_registry()
    print(f"Exported feature registry to {reg_p}")
