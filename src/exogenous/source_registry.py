"""
Authoritative Source Registry for Exogenous Agricultural & Meteorological Data (Day 22).

Registers metadata, license terms, spatial resolution, temporal resolution,
publication schedules, and access protocols for all external environmental datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd

SOURCES = [
    {
        "source_id": "IMD_DISTRICT_MET_SERIES",
        "source_name": "IMD District Meteorological Series",
        "provider": "India Meteorological Department (IMD), Ministry of Earth Sciences, Govt. of India",
        "source_url": "https://mausam.imd.gov.in/",
        "tier": "Tier 1: Weather / Climate",
        "spatial_level": "DISTRICT",
        "spatial_resolution": "District Aggregate from 0.25° x 0.25° Gridded Product",
        "temporal_resolution": "MONTHLY_AND_SEASONAL",
        "temporal_coverage": "1966–2017",
        "variables_provided": [
            "preseason_rainfall_mm",
            "preseason_rainfall_anomaly_pct",
            "preseason_temp_mean_c",
            "preseason_temp_max_c",
            "preseason_temp_min_c",
            "preseason_dry_spell_days",
            "rainfall_lag1_annual_mm"
        ],
        "units": {
            "rainfall": "mm",
            "temperature": "Degrees Celsius (°C)",
            "anomaly": "Percentage deviation from 30-year normal (%)",
            "dry_spells": "Consecutive days with precipitation < 1.0mm"
        },
        "publication_schedule": "Monthly Bulletin (Published by 10th of succeeding month)",
        "preseason_availability_date": "May 31 of harvest year t (prior to Kharif planting)",
        "license_terms": "Open Government Data License India (GODL) / Open Research Access",
        "status": "VERIFIED_PRIMARY",
        "missingness_handling": "State-level agro-climatic zone mean imputation where station density is sparse."
    },
    {
        "source_id": "NASA_POWER_ERA5_AGROCLIM",
        "source_name": "NASA POWER / ECMWF ERA5-Land Agro-Climatology",
        "provider": "NASA Langley Research Center & European Centre for Medium-Range Weather Forecasts",
        "source_url": "https://power.larc.nasa.gov/",
        "tier": "Tier 2: Environmental / Moisture",
        "spatial_level": "DISTRICT",
        "spatial_resolution": "District centroid aggregation from 0.1° x 0.1° reanalysis grid",
        "temporal_resolution": "MONTHLY",
        "temporal_coverage": "1981–2017",
        "variables_provided": [
            "preseason_soil_moisture_index",
            "preseason_spei_aridity_index",
            "preseason_heat_stress_days"
        ],
        "units": {
            "soil_moisture": "Normalized Saturation Index [0.0, 1.0]",
            "spei_aridity": "Standardized Precipitation Evapotranspiration Index [-3.0, +3.0]",
            "heat_stress": "Count of days where Tmax > 38.0°C in Mar–May"
        },
        "publication_schedule": "Near-Real-Time (5-day latency)",
        "preseason_availability_date": "May 25 of harvest year t",
        "license_terms": "NASA Open Data Policy / Copernicus Open Access",
        "status": "VERIFIED_SECONDARY",
        "missingness_handling": "Bilinear spatial interpolation from adjacent grid nodes."
    },
    {
        "source_id": "ICRISAT_AGROCLIMATIC_MESONET",
        "source_name": "ICRISAT Semi-Arid Tropics Climate Network",
        "provider": "International Crops Research Institute for the Semi-Arid Tropics (ICRISAT)",
        "source_url": "http://data.icrisat.org/dld/",
        "tier": "Tier 1: Weather / Climate & Irrigation",
        "spatial_level": "DISTRICT",
        "spatial_resolution": "Standard 1966 District Panel (311 Districts)",
        "temporal_resolution": "ANNUAL_SEASONAL",
        "temporal_coverage": "1966–2017",
        "variables_provided": [
            "irrigation_ratio_lag1",
            "monsoon_onset_anomaly_days"
        ],
        "units": {
            "irrigation_ratio": "Gross Irrigated Area / Gross Cropped Area [0.0, 1.0]",
            "onset_anomaly": "Days deviation from historical district median onset date"
        },
        "publication_schedule": "Annual ICRISAT DLD Research Release",
        "preseason_availability_date": "Historical record (t-1 observed)",
        "license_terms": "Open Access for Academic & Research Use",
        "status": "VERIFIED_TERTIARY",
        "missingness_handling": "Forward fill with state mean boundary fallback."
    }
]


class ExogenousSourceRegistry:
    """Manages authoritative metadata, licenses, and provenance for all exogenous datasets."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def get_sources(self) -> List[Dict[str, Any]]:
        return SOURCES

    def get_source(self, source_id: str) -> Dict[str, Any] | None:
        for s in SOURCES:
            if s["source_id"] == source_id:
                return s
        return None

    def export_registry(self) -> Tuple[Path, Path]:
        csv_path = self.metadata_dir / "exogenous_source_registry.csv"
        json_path = self.metadata_dir / "exogenous_source_registry.json"

        # Export CSV
        df = pd.DataFrame([
            {
                "source_id": s["source_id"],
                "source_name": s["source_name"],
                "provider": s["provider"],
                "tier": s["tier"],
                "spatial_level": s["spatial_level"],
                "temporal_resolution": s["temporal_resolution"],
                "temporal_coverage": s["temporal_coverage"],
                "status": s["status"],
                "preseason_availability_date": s["preseason_availability_date"],
                "license_terms": s["license_terms"]
            }
            for s in SOURCES
        ])
        df.to_csv(csv_path, index=False)

        # Export JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "1.0.0",
                "generated_at": "2026-09-03T18:00:00Z",
                "total_sources": len(SOURCES),
                "sources": SOURCES
            }, f, indent=2)

        return csv_path, json_path


if __name__ == "__main__":
    registry = ExogenousSourceRegistry()
    csv_p, json_p = registry.export_registry()
    print(f"Exported source registry to {csv_p} and {json_p}")
