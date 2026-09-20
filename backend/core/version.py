"""
Platform Versioning Constants.

Distinguishes Application Version, Dataset Version, Model Version,
Methodology Version, and API Version across the entire platform.
"""

APPLICATION_VERSION: str = "1.0.0"
DATASET_VERSION: str = "ICRISAT 1966-2017"
MODEL_VERSION: str = "exogenous_rf_forecaster v2.1.0"
METHODOLOGY_VERSION: str = "Day 15 Production Integrated"
API_VERSION: str = "v1"

# Registered Pipeline Artifact Versions
REGISTERED_MODEL_VERSIONS = {
    "exogenous_rf_forecaster": "v2.1.0",
    "pre_season_baseline_rf": "v1.0.0",
    "post_harvest_baseline_rf": "v1.0.0",
    "isolation_forest_anomaly_detector": "v1.0.0",
    "kmeans_spatial_clusterer": "v1.0.0",
    "multi_horizon_forecaster": "v1.0.0"
}
