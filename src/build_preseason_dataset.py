"""
Genuine Pre-Season Dataset Builder and Feature Pipeline.

Constructs an exogenous, leak-free pre-season feature matrix using only:
1. Pre-season land allocation features (Total Cropped Area, Rice Area Share, Major Crop Areas)
2. Historical lagged performance (t-1 yield, 3-year rolling average yield)
3. Spatio-temporal identifiers (Year, State Code, Dist Code)

Strictly excludes all concurrent harvest production and concurrent yield variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Core Pre-Season Features
EXOGENOUS_FEATURE_COLUMNS = [
    'Year',
    'State Code',
    'RICE AREA (1000 ha)',
    'TOTAL_CROPPED_AREA',
    'RICE_AREA_SHARE',
    'WHEAT AREA (1000 ha)',
    'COTTON AREA (1000 ha)',
    'SUGARCANE AREA (1000 ha)',
    'RICE_YIELD_LAG1',
    'RICE_YIELD_ROLL3'
]

TARGET_COLUMN = 'RICE YIELD (Kg per ha)'

def load_and_engineer_preseason_features(
    dataset_path: str = "Datasets/rice_data_outlier_removed.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads dataset and engineers strictly pre-season exogenous features.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_file = base_dir / dataset_path
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found at {data_file}")
        
    df = pd.read_csv(data_file)
    df = df.sort_values(['Dist Code', 'Year']).reset_index(drop=True)
    
    # 1. Compute Pre-Season Land Allocation Features
    area_cols = [c for c in df.columns if 'AREA' in c]
    df['TOTAL_CROPPED_AREA'] = df[area_cols].sum(axis=1)
    df['RICE_AREA_SHARE'] = np.where(
        df['TOTAL_CROPPED_AREA'] > 0,
        df['RICE AREA (1000 ha)'] / df['TOTAL_CROPPED_AREA'],
        0.0
    )
    
    # 2. Compute Historical Lagged Yields (strictly shift(1) from past years)
    df['RICE_YIELD_LAG1'] = df.groupby('Dist Code')[TARGET_COLUMN].shift(1)
    df['RICE_YIELD_LAG2'] = df.groupby('Dist Code')[TARGET_COLUMN].shift(2)
    df['RICE_YIELD_ROLL3'] = (
        df.groupby('Dist Code')[TARGET_COLUMN]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )
    
    # Handle initial-year missing lags using state-level historical median (no future leak)
    state_medians = df.groupby('State Code')[TARGET_COLUMN].transform('median')
    df['RICE_YIELD_LAG1'] = df['RICE_YIELD_LAG1'].fillna(state_medians)
    df['RICE_YIELD_LAG2'] = df['RICE_YIELD_LAG2'].fillna(df['RICE_YIELD_LAG1'])
    df['RICE_YIELD_ROLL3'] = df['RICE_YIELD_ROLL3'].fillna(df['RICE_YIELD_LAG1'])
    
    # Manifest metadata
    manifest = {
        "dataset_name": "ICRISAT District-Level Rice Panel",
        "total_records": int(len(df)),
        "temporal_range": [int(df['Year'].min()), int(df['Year'].max())],
        "states_count": int(df['State Code'].nunique()),
        "districts_count": int(df['Dist Code'].nunique()),
        "target_variable": TARGET_COLUMN,
        "preseason_features": EXOGENOUS_FEATURE_COLUMNS,
        "feature_descriptions": {
            "Year": "Agricultural survey year (multi-year technological trend)",
            "State Code": "State categorical identifier (1-20)",
            "RICE AREA (1000 ha)": "District rice cultivated area in thousand hectares",
            "TOTAL_CROPPED_AREA": "Total cultivated area across all crops in district",
            "RICE_AREA_SHARE": "Proportion of district cropland allocated to rice cultivation",
            "WHEAT AREA (1000 ha)": "District wheat cropped area (Rabi cropping intensity indicator)",
            "COTTON AREA (1000 ha)": "District cotton cropped area (cash crop competition indicator)",
            "SUGARCANE AREA (1000 ha)": "District sugarcane cropped area (high irrigation access indicator)",
            "RICE_YIELD_LAG1": "Reported rice yield in previous agricultural year (t-1)",
            "RICE_YIELD_ROLL3": "3-year historical rolling average rice yield up to t-1"
        },
        "excluded_leakage_features": [
            "RICE PRODUCTION (1000 tons)",
            "All concurrent non-rice production variables (23 columns)",
            "All concurrent non-rice yield variables (22 columns)"
        ]
    }
    
    return df, manifest

def main():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / 'Models'
    models_dir.mkdir(exist_ok=True, parents=True)
    
    df, manifest = load_and_engineer_preseason_features()
    
    manifest_path = models_dir / 'preseason_feature_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Pre-season dataset engineered: {df.shape[0]} rows, {len(EXOGENOUS_FEATURE_COLUMNS)} pre-season features.")
    print(f"Feature manifest saved to {manifest_path}")

if __name__ == '__main__':
    main()
