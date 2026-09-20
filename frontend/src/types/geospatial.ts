export interface StateSpatialItem {
  state: string;
  state_code: number;
  lat: number;
  lon: number;
  region: string;
  agro_zone: string;
  district_count: number;
  average_yield_kg_ha: number;
  average_area_k_ha: number;
  risk_score: number;
  risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  trend_direction: string;
  theil_sen_slope: number;
  forecast_1yr_kg_ha: number;
  anomaly_count: number;
  cluster_id: number;
}

export interface GeospatialOverviewResponse {
  total_states_monitored: number;
  total_districts_monitored: number;
  total_observations: number;
  national_average_yield_kg_ha: number;
  national_average_risk_score: number;
  high_risk_states_count: number;
  anomalous_states_count: number;
  spatial_outliers_count: number;
  clusters_count: number;
}

export interface SpatialClusterItem {
  cluster_id: number;
  cluster_name: string;
  archetype: string;
  risk_profile: string;
  district_count: number;
  avg_yield_kg_ha: number;
  avg_volatility_pct: number;
  avg_theil_sen_slope: number;
  avg_anomaly_rate_pct: number;
  dominant_states: Record<string, number>;
}

export interface SpatialOutlierItem {
  state: string;
  district: string;
  yield_kg_ha: number;
  state_mean_yield: number;
  within_state_zscore: number;
  relative_yield_ratio: number;
  relative_risk_ratio: number;
  anomaly_rate_pct: number;
  theil_sen_slope: number;
  reasons: string[];
  severity: 'MODERATE' | 'HIGH';
}

export interface StateSpatialProfileResponse {
  state: string;
  state_code: number;
  lat: number;
  lon: number;
  region: string;
  agro_zone: string;
  district_count: number;
  average_yield_kg_ha: number;
  risk_score: number;
  risk_level: string;
  trend_direction: string;
  theil_sen_slope: number;
  forecasts: Array<{
    forecast_year: number;
    horizon_years: number;
    predicted_yield: number;
    lower_bound_p10: number;
    upper_bound_p90: number;
    uncertainty_pct: number;
  }>;
  spatial_outliers: SpatialOutlierItem[];
  districts: Array<{
    district_name: string;
    average_yield_kg_ha: number;
    yield_volatility_pct: number;
    state_relative_yield_ratio: number;
    district_yield_zscore_state: number;
    trend_direction: string;
    trend_theil_sen_slope: number;
    anomaly_rate_pct: number;
  }>;
}

export interface SimilarRegionItem {
  state: string;
  district: string;
  similarity_score: number;
  average_yield_kg_ha: number;
  yield_volatility_pct: number;
  trend_theil_sen_slope: number;
}
