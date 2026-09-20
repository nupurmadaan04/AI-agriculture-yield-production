"""
Government of India Open Government Data (OGD) Ingestion Module.

Handles ingestion specification, schema validation, and mapping for Ministry of Agriculture
Directorate of Economics and Statistics (DES) agricultural statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class GovtOGDIngestion:
    """Ingestion and normalization handler for Government of India OGD crop statistics."""

    SOURCE_ID = "GOVT_INDIA_OGD_DES"

    def __init__(self, raw_data_path: Optional[str | Path] = None):
        self.raw_path = Path(raw_data_path) if raw_data_path else None

    def get_source_metadata(self) -> Dict[str, Any]:
        return {
            "source_id": self.SOURCE_ID,
            "provider": "Ministry of Agriculture & Farmers Welfare (DES), Government of India",
            "url": "https://data.gov.in/",
            "status": "INGESTION_SPECIFICATION_REGISTERED",
            "required_fields": ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"],
            "unit_standardization": {
                "Area": "Hectares",
                "Production": "Tonnes",
                "Yield": "Computed as (Production * 1000) / Area (Kg/ha)"
            }
        }

    def normalize(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes OGD raw dataframe into canonical agricultural panel schema."""
        records = []
        for idx, row in raw_df.iterrows():
            area = float(row.get("Area", 0)) if pd.notnull(row.get("Area")) else None
            prod = float(row.get("Production", 0)) if pd.notnull(row.get("Production")) else None
            yield_val = round((prod * 1000.0) / area, 2) if (area and prod and area > 0 and prod >= 0) else None

            records.append({
                "record_id": f"OGD_{idx}",
                "source": self.SOURCE_ID,
                "dataset_version": "AGRI_PANEL_1.0",
                "country": "India",
                "state_raw": str(row.get("State_Name", "")),
                "state": str(row.get("State_Name", "")).title(),
                "district_raw": str(row.get("District_Name", "")),
                "district": str(row.get("District_Name", "")).title(),
                "year": int(row.get("Crop_Year", 2017)),
                "season_raw": str(row.get("Season", "Whole Year")),
                "season": str(row.get("Season", "Annual")).strip(),
                "crop_raw": str(row.get("Crop", "")),
                "crop": str(row.get("Crop", "")).title(),
                "area_ha": area,
                "production_tonnes": prod,
                "yield_kg_ha": yield_val
            })
        return pd.DataFrame(records)
