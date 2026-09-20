"""
FAOSTAT Crop Production Ingestion Module.

Handles ingestion specification, schema mapping, and unit conversions
for Food and Agriculture Organization (FAO) international crop statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class FAOSTATIngestion:
    """Ingestion and normalization handler for FAOSTAT agricultural statistics."""

    SOURCE_ID = "FAOSTAT_CROP_PRODUCTION"

    def __init__(self, raw_data_path: Optional[str | Path] = None):
        self.raw_path = Path(raw_data_path) if raw_data_path else None

    def get_source_metadata(self) -> Dict[str, Any]:
        return {
            "source_id": self.SOURCE_ID,
            "provider": "Food and Agriculture Organization of the United Nations (FAO)",
            "url": "https://www.fao.org/faostat/en/#data/QCL",
            "status": "INGESTION_SPECIFICATION_REGISTERED",
            "required_fields": ["Area", "Item", "Element", "Year", "Unit", "Value"],
            "unit_standardization": {
                "Area harvested": "ha",
                "Production": "tonnes",
                "Yield": "hg/ha -> converted to kg/ha (Value / 10)"
            }
        }
