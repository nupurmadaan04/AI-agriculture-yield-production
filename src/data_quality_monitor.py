"""
Agricultural Data Quality & Integrity Audit Monitor.

Evaluates dataset completeness, numerical validity, relational consistency,
and temporal continuity, producing a reproducible 0–100 Data Quality Score.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from backend.utils.data_loader import data_loader

class DataQualityMonitor:
    def audit_dataset(self) -> Dict[str, Any]:
        """
        Executes comprehensive 4-pillar data quality audit on the ICRISAT panel.
        """
        df = data_loader.dataframe

        total_rows = len(df)
        total_cells = df.size

        # 1. Completeness Score (Weight: 0.30)
        missing_count = int(df.isnull().sum().sum())
        completeness_pct = float(max(0.0, (1.0 - (missing_count / max(total_cells, 1))) * 100.0))

        # 2. Validity Score (Weight: 0.30)
        # Checks: Area >= 0, Production >= 0, Yield between 400 and 7000 kg/ha
        invalid_area = int(np.sum(df['RICE AREA (1000 ha)'] < 0))
        invalid_prod = int(np.sum(df['RICE PRODUCTION (1000 tons)'] < 0))
        invalid_yield = int(np.sum((df['RICE YIELD (Kg per ha)'] < 400) | (df['RICE YIELD (Kg per ha)'] > 7000)))
        total_invalid = invalid_area + invalid_prod + invalid_yield
        validity_pct = float(max(0.0, (1.0 - (total_invalid / max(total_rows * 3, 1))) * 100.0))

        # 3. Consistency Score (Weight: 0.20)
        # Checks: Duplicate year-district rows, unmapped state codes
        duplicates = int(df.duplicated(subset=['Dist Name', 'Year']).sum())
        consistency_pct = float(max(0.0, (1.0 - (duplicates / max(total_rows, 1))) * 100.0))

        # 4. Temporal Integrity Score (Weight: 0.20)
        # Checks: Full 8-year span (2010–2017) presence across districts
        year_counts = df.groupby('Dist Name')['Year'].nunique()
        expected_years = 8
        full_history_districts = int(np.sum(year_counts == expected_years))
        temporal_integrity_pct = float((full_history_districts / max(len(year_counts), 1)) * 100.0)

        # Composite Data Quality Score (0–100)
        overall_score = (
            0.30 * completeness_pct +
            0.30 * validity_pct +
            0.20 * consistency_pct +
            0.20 * temporal_integrity_pct
        )

        return {
            'overall_quality_score': round(overall_score, 1),
            'status': 'EXCELLENT' if overall_score >= 90.0 else 'GOOD' if overall_score >= 80.0 else 'NEEDS_REVIEW',
            'records_evaluated': total_rows,
            'total_features': df.shape[1],
            'sub_scores': {
                'completeness': {
                    'score': round(completeness_pct, 1),
                    'missing_cells': missing_count,
                    'total_cells': total_cells,
                    'weight': 0.30
                },
                'validity': {
                    'score': round(validity_pct, 1),
                    'invalid_area_count': invalid_area,
                    'invalid_prod_count': invalid_prod,
                    'out_of_range_yield_count': invalid_yield,
                    'weight': 0.30
                },
                'consistency': {
                    'score': round(consistency_pct, 1),
                    'duplicate_keys_count': duplicates,
                    'weight': 0.20
                },
                'temporal_integrity': {
                    'score': round(temporal_integrity_pct, 1),
                    'monitored_districts': len(year_counts),
                    'districts_with_full_8yr_history': full_history_districts,
                    'weight': 0.20
                }
            },
            'dataset_temporal_bounds': {
                'start_year': int(df['Year'].min()),
                'end_year': int(df['Year'].max()),
                'total_years': int(df['Year'].nunique())
            }
        }

data_quality_monitor = DataQualityMonitor()
