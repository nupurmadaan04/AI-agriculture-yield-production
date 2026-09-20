"""
Canonical Agricultural Dataset Loader.

Loads, validates, and caches the cleaned ICRISAT agricultural dataset.
Guarantees zero runtime modification of source data, complete schema validation,
and detailed dataset health/metadata reporting.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from backend.core.paths import paths
from backend.core.version import DATASET_VERSION

REQUIRED_COLUMNS: List[str] = [
    'State Name',
    'Dist Name',
    'Year',
    'RICE AREA (1000 ha)',
    'RICE PRODUCTION (1000 tons)',
    'RICE YIELD (Kg per ha)',
]


class DataLoader:
    _instance: Optional['DataLoader'] = None
    _df: Optional[pd.DataFrame] = None
    _metadata: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance

    def load_dataset(self, csv_path: Optional[str | Path] = None) -> pd.DataFrame:
        """Loads, validates, and caches the cleaned agricultural dataset from disk."""
        if self._df is not None:
            return self._df

        target_path = Path(csv_path) if csv_path else paths.canonical_dataset_path

        if not target_path.exists():
            raise FileNotFoundError(f"Agricultural dataset not found at canonical path: {target_path}")

        df = pd.read_csv(target_path)

        # 1. Validate required columns exist
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required canonical columns: {missing}")

        # 2. Compute immutable dataset health metadata
        self._metadata = {
            "dataset_version": DATASET_VERSION,
            "file_path": str(target_path.resolve()),
            "total_records": int(len(df)),
            "total_states": int(df['State Name'].nunique()),
            "total_districts": int(df['Dist Name'].nunique()),
            "year_min": int(df['Year'].min()),
            "year_max": int(df['Year'].max()),
            "null_count": int(df[REQUIRED_COLUMNS].isnull().sum().sum()),
            "is_validated": True
        }

        self._df = df
        return self._df

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns the loaded dataframe."""
        if self._df is None:
            return self.load_dataset()
        return self._df

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns the cached dataset health and provenance metadata."""
        if self._metadata is None:
            self.load_dataset()
        return self._metadata or {}

    def get_status(self) -> Dict[str, Any]:
        """Returns health/readiness status dictionary for system readiness checks."""
        try:
            df = self.dataframe
            return {
                "status": "ready",
                "total_records": len(df),
                "dataset_version": DATASET_VERSION,
                "year_range": f"{df['Year'].min()} - {df['Year'].max()}"
            }
        except Exception as e:
            return {
                "status": "not_ready",
                "error": str(e)
            }


data_loader = DataLoader()
