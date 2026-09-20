from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

class StateSpatialItem(BaseModel):
    state: str
    state_code: int
    lat: float
    lon: float
    region: str
    agro_zone: str
    district_count: int
    average_yield_kg_ha: float
    average_area_k_ha: float
    risk_score: float
    risk_level: str
    trend_direction: str
    theil_sen_slope: float
    forecast_1yr_kg_ha: float
    anomaly_count: int
    cluster_id: int

class GeospatialOverviewResponse(BaseModel):
    total_states_monitored: int
    total_districts_monitored: int
    total_observations: int
    national_average_yield_kg_ha: float
    national_average_risk_score: float
    high_risk_states_count: int
    anomalous_states_count: int
    spatial_outliers_count: int
    clusters_count: int

class SpatialClusterItem(BaseModel):
    cluster_id: int
    cluster_name: str
    archetype: str
    risk_profile: str
    district_count: int
    avg_yield_kg_ha: float
    avg_volatility_pct: float
    avg_theil_sen_slope: float
    avg_anomaly_rate_pct: float
    dominant_states: Dict[str, int]

class SpatialQueryRequest(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    metric: Optional[str] = "yield"
    risk_threshold: Optional[float] = None
    anomaly_only: Optional[bool] = False
    cluster_id: Optional[int] = None

class SimilarRegionItem(BaseModel):
    state: str
    district: str
    similarity_score: float
    average_yield_kg_ha: float
    yield_volatility_pct: float
    trend_theil_sen_slope: float
