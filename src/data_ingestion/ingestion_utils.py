"""
Data Ingestion Utilities.

Provides file hashing (SHA-256), schema profiling, and validation functions
for multi-source agricultural dataset ingestion.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def compute_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def profile_dataframe(df: pd.DataFrame, source_id: str, file_path: str) -> Dict[str, Any]:
    """Generates standard profile metrics for an ingested dataframe."""
    return {
        "source_id": source_id,
        "file_path": str(file_path),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "null_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
