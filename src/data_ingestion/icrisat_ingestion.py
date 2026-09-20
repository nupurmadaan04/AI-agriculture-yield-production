"""
ICRISAT District Level Database Ingestion Module.

Extracts, cleans, and standardizes ICRISAT multi-crop district panel data (1966–2017).
Normalizes wide format (CROP AREA, CROP PRODUCTION, CROP YIELD) into standardized long format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from src.data_ingestion.ingestion_utils import compute_sha256, profile_dataframe


# Mapping from wide column prefix to canonical standardized crop name
CROP_PREFIX_MAPPING: Dict[str, str] = {
    "RICE": "Rice",
    "WHEAT": "Wheat",
    "KHARIF SORGHUM": "Kharif Sorghum",
    "RABI SORGHUM": "Rabi Sorghum",
    "SORGHUM": "Sorghum",
    "PEARL MILLET": "Pearl Millet",
    "MAIZE": "Maize",
    "FINGER MILLET": "Finger Millet",
    "BARLEY": "Barley",
    "CHICKPEA": "Chickpea",
    "PIGEONPEA": "Pigeonpea",
    "MINOR PULSES": "Minor Pulses",
    "GROUNDNUT": "Groundnut",
    "SESAMUM": "Sesamum",
    "RAPESEED AND MUSTARD": "Rapeseed and Mustard",
    "SAFFLOWER": "Safflower",
    "CASTOR": "Castor",
    "LINSEED": "Linseed",
    "SUNFLOWER": "Sunflower",
    "SOYABEAN": "Soyabean",
    "OILSEEDS": "Oilseeds",
    "SUGARCANE": "Sugarcane",
    "COTTON": "Cotton",
    "FRUITS": "Fruits",
    "VEGETABLES": "Vegetables",
    "FRUITS AND VEGETABLES": "Fruits and Vegetables",
    "POTATOES": "Potatoes",
    "ONION": "Onion",
    "FODDER": "Fodder"
}


class ICRISATIngestion:
    """Ingests and transforms ICRISAT District-Level multi-crop panel dataset."""

    SOURCE_ID = "ICRISAT_DLD_1966_2017"

    def __init__(self, raw_csv_path: str | Path):
        self.raw_path = Path(raw_csv_path)

    def ingest_raw(self) -> pd.DataFrame:
        """Reads the raw CSV file."""
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw ICRISAT dataset not found at: {self.raw_path}")
        return pd.read_csv(self.raw_path)

    def extract_long_panel(self, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transforms wide-format ICRISAT panel (80 columns) into canonical standardized long panel.
        Standardizes units:
        - Area: 1000 ha -> hectares (* 1000)
        - Production: 1000 tons -> metric tonnes (* 1000)
        - Yield: kg/ha
        """
        df = raw_df if raw_df is not None else self.ingest_raw()

        records = []
        base_cols = ["Dist Code", "Year", "State Code", "State Name", "Dist Name"]

        for idx, row in df.iterrows():
            dist_code = row["Dist Code"]
            year = int(row["Year"])
            state_code = row["State Code"]
            state_name = str(row["State Name"]).strip()
            dist_name = str(row["Dist Name"]).strip()

            for prefix, std_crop in CROP_PREFIX_MAPPING.items():
                area_col = f"{prefix} AREA (1000 ha)"
                prod_col = f"{prefix} PRODUCTION (1000 tons)"
                yield_col = f"{prefix} YIELD (Kg per ha)"

                has_area = area_col in df.columns
                has_prod = prod_col in df.columns
                has_yield = yield_col in df.columns

                area_val = row[area_col] if has_area and pd.notnull(row[area_col]) else None
                prod_val = row[prod_col] if has_prod and pd.notnull(row[prod_col]) else None
                yield_val = row[yield_col] if has_yield and pd.notnull(row[yield_col]) else None

                # Only include record if at least one metric is present and positive/non-null
                if area_val is not None or prod_val is not None or yield_val is not None:
                    # Clean -1.0 or -999.0 placeholder missing values commonly in survey data
                    a_clean = float(area_val) * 1000.0 if (area_val is not None and float(area_val) >= 0) else None
                    p_clean = float(prod_val) * 1000.0 if (prod_val is not None and float(prod_val) >= 0) else None
                    y_clean = float(yield_val) if (yield_val is not None and float(yield_val) >= 0) else None

                    # If area is 0 and production > 0, area was rounded down below survey threshold
                    if a_clean == 0.0 and p_clean is not None and p_clean > 0:
                        # Retain production but flag area as unmeasured/unrounded
                        a_clean = None
                        y_clean = None
                    elif y_clean is None and a_clean is not None and p_clean is not None and a_clean > 0:
                        y_clean = round((p_clean * 1000.0) / a_clean, 2)

                    season = "Kharif" if "KHARIF" in prefix else ("Rabi" if "RABI" in prefix else "Annual")

                    rec = {
                        "record_id": f"ICR_{dist_code}_{year}_{std_crop[:3].upper()}_{idx}",
                        "source": self.SOURCE_ID,
                        "dataset_version": "AGRI_PANEL_1.0",
                        "country": "India",
                        "state_raw": state_name,
                        "state": state_name.title(),
                        "district_raw": dist_name,
                        "district": dist_name.title(),
                        "year": year,
                        "season_raw": season,
                        "season": season,
                        "crop_raw": prefix,
                        "crop": std_crop,
                        "area_ha": a_clean,
                        "production_tonnes": p_clean,
                        "yield_kg_ha": y_clean
                    }
                    records.append(rec)

        long_df = pd.DataFrame(records)
        return long_df
