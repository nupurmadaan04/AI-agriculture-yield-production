"""
Agricultural Anomaly Detector Training Engine.

Trains an unsupervised IsolationForest model on numerical agricultural indicators
(land allocation, production, yield, and historical productivity lags) to detect
anomalous observations, survey errors, and severe agricultural distress patterns.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.build_preseason_dataset import load_and_engineer_preseason_features

ANOMALY_FEATURE_COLUMNS = [
    'RICE AREA (1000 ha)',
    'RICE PRODUCTION (1000 tons)',
    'RICE YIELD (Kg per ha)',
    'TOTAL_CROPPED_AREA',
    'RICE_AREA_SHARE',
    'RICE_YIELD_LAG1',
    'RICE_YIELD_ROLL3'
]

def main():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / 'Models'
    models_dir.mkdir(exist_ok=True, parents=True)

    df, manifest = load_and_engineer_preseason_features()

    X = df[ANOMALY_FEATURE_COLUMNS].copy()

    # Isolation Forest Pipeline with standard scaling
    anomaly_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('detector', IsolationForest(
            n_estimators=150,
            contamination=0.05, # Expecting ~5% structural or reporting anomalies
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("Fitting IsolationForest Anomaly Detector...")
    anomaly_pipeline.fit(X)

    # Evaluate anomaly predictions and decision scores
    preds = anomaly_pipeline.predict(X) # -1 for anomalies, 1 for normal
    scores = anomaly_pipeline.decision_function(X) # lower score = more abnormal

    anomaly_count = int(np.sum(preds == -1))
    normal_count = int(np.sum(preds == 1))
    anomaly_pct = float(round((anomaly_count / len(df)) * 100, 2))

    print(f"Total Observations: {len(df)}")
    print(f"Detected Anomalies: {anomaly_count} ({anomaly_pct}%)")
    print(f"Normal Records: {normal_count}")

    # Save Pipeline
    model_path = models_dir / 'agricultural_anomaly_pipeline.pkl'
    joblib.dump(anomaly_pipeline, model_path)
    print(f"Saved anomaly detection pipeline to {model_path}")

    # Metadata
    metadata = {
        "model_name": "IsolationForest (Agricultural Anomaly Detector)",
        "model_id": "iforest-agri-anomaly",
        "model_type": "unsupervised_anomaly_detection",
        "algorithm": "IsolationForest",
        "n_estimators": 150,
        "contamination": 0.05,
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "features": ANOMALY_FEATURE_COLUMNS,
        "features_count": len(ANOMALY_FEATURE_COLUMNS),
        "excluded_identifiers": ["Dist Code", "State Code", "Year", "State Name", "Dist Name"],
        "dataset_statistics": {
            "total_records": len(df),
            "anomalies_detected": anomaly_count,
            "anomaly_percentage": anomaly_pct,
            "score_min": float(round(scores.min(), 4)),
            "score_max": float(round(scores.max(), 4)),
            "score_mean": float(round(scores.mean(), 4)),
            "score_std": float(round(scores.std(), 4))
        },
        "scientific_interpretation": (
            "Anomalies represent multi-dimensional statistical outliers across crop area, production volume, "
            "yield levels, land allocation shares, and historical performance lags. They highlight severe local "
            "distress, data recording anomalies, or sudden regional land reallocation."
        )
    }

    metadata_path = models_dir / 'anomaly_model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved anomaly detector metadata to {metadata_path}")

if __name__ == '__main__':
    main()
