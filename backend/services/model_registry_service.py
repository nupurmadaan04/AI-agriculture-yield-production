"""
Agricultural Model Registry & Governance Service.

Tracks and registers all trained machine learning pipelines, versioning metadata,
training/evaluation boundaries, feature lists, and operational deployment statuses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

class ModelRegistryService:
    _instance: Optional['ModelRegistryService'] = None
    _models: Optional[List[Dict[str, Any]]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistryService, cls).__new__(cls)
        return cls._instance

    def get_registered_models(self) -> List[Dict[str, Any]]:
        """
        Returns structured registry entries for all trained pipelines in Models/.
        """
        if self._models is not None:
            return self._models

        base_dir = Path(__file__).resolve().parent.parent.parent
        models_dir = base_dir / 'Models'

        registry = [
            {
                'model_id': 'exogenous_rf_forecaster',
                'model_name': 'Exogenous Random Forest Forecaster',
                'version': '2.1.0',
                'model_type': 'RandomForestRegressor (n_estimators=150, max_depth=14)',
                'task': 'Multi-Horizon Yield Forecasting & Scenario Simulation',
                'target': 'RICE YIELD (Kg per ha)',
                'training_period': '2010–2015 (1,851 records)',
                'evaluation_period': '2016–2017 (618 out-of-time records)',
                'test_metrics': {
                    'mae': 353.01,
                    'rmse': 513.11,
                    'r2': 0.7866,
                    'mape': 18.04
                },
                'features': [
                    'Year', 'State Code', 'RICE AREA (1000 ha)', 'TOTAL_CROPPED_AREA',
                    'RICE_AREA_SHARE', 'WHEAT AREA (1000 ha)', 'COTTON AREA (1000 ha)',
                    'SUGARCANE AREA (1000 ha)', 'RICE_YIELD_LAG1', 'RICE_YIELD_ROLL3'
                ],
                'artifact_path': 'Models/forecasting_pipeline.pkl',
                'status': 'ACTIVE_PRODUCTION',
                'is_primary': True
            },
            {
                'model_id': 'pre_season_exogenous_pipeline',
                'model_name': 'Pre-Season Exogenous Pipeline',
                'version': '2.0.0',
                'model_type': 'GradientBoostingRegressor / RandomForest',
                'task': 'Single-Season Pre-Harvest Yield Estimation',
                'target': 'RICE YIELD (Kg per ha)',
                'training_period': '2010–2015',
                'evaluation_period': '2016–2017',
                'test_metrics': {
                    'mae': 366.78,
                    'rmse': 524.63,
                    'r2': 0.7769,
                    'mape': 18.53
                },
                'features': [
                    'Year', 'State Code', 'RICE AREA (1000 ha)', 'TOTAL_CROPPED_AREA',
                    'RICE_AREA_SHARE', 'RICE_YIELD_LAG1', 'RICE_YIELD_ROLL3'
                ],
                'artifact_path': 'Models/pre_season_exogenous_pipeline.pkl',
                'status': 'ACTIVE',
                'is_primary': False
            },
            {
                'model_id': 'isolation_forest_anomaly_detector',
                'model_name': 'Isolation Forest Anomaly Detector',
                'version': '1.0.0',
                'model_type': 'IsolationForest (contamination=0.04)',
                'task': 'Multi-Dimensional Agricultural Shock & Anomaly Detection',
                'target': 'Anomaly Score & Departure Flags',
                'training_period': '2010–2017 (Unsupervised)',
                'evaluation_period': 'Cross-Temporal Validation',
                'test_metrics': {
                    'contamination_rate': 0.04,
                    'anomalies_detected': 99
                },
                'features': ['RICE YIELD (Kg per ha)', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)'],
                'artifact_path': 'Models/agricultural_anomaly_pipeline.pkl',
                'status': 'ACTIVE_PRODUCTION',
                'is_primary': False
            },
            {
                'model_id': 'kmeans_spatial_clusterer',
                'model_name': 'KMeans Spatial Clustering Pipeline',
                'version': '1.0.0',
                'model_type': 'KMeans (n_clusters=4, StandardScaler)',
                'task': 'Unsupervised Agro-Climatic Regional Clustering',
                'target': 'Cluster ID (0–3)',
                'training_period': '2010–2017 Panel Aggregates',
                'evaluation_period': 'Silhouette & DB Index Optimization',
                'test_metrics': {
                    'silhouette_score': 0.2712,
                    'calinski_harabasz': 120.78,
                    'davies_bouldin': 1.2358
                },
                'features': ['average_yield_kg_ha', 'yield_volatility_pct', 'trend_theil_sen_slope', 'anomaly_rate_pct'],
                'artifact_path': 'Models/spatial_cluster_pipeline.pkl',
                'status': 'ACTIVE_PRODUCTION',
                'is_primary': False
            }
        ]

        self._models = registry
        return self._models

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Finds a specific model by ID or name."""
        models = self.get_registered_models()
        for m in models:
            if m['model_id'].lower() == model_id.lower() or m['model_name'].lower() == model_id.lower():
                return m
        return None

model_registry_service = ModelRegistryService()
