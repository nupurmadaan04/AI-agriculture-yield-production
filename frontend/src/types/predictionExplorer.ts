/**
 * Day 29 Prediction Explorer & Forecast Explainability Types
 */

export interface HistoricalObservationItem {
  year: number
  yield_kg_ha: number
  area_ha?: number | null
  production_tonnes?: number | null
  observation_type: 'OBSERVED' | 'DERIVED' | string
}

export interface ForecastContextResponse {
  crop: string
  state: string
  district: string
  forecast_year: number
  historical_observations_count: number
  district_historical_mean?: number | null
  previous_year_yield?: number | null
  rolling_3yr_mean?: number | null
  historical_min_yield?: number | null
  historical_max_yield?: number | null
  historical_yield_std?: number | null
  recent_observations: HistoricalObservationItem[]
  has_sufficient_history: boolean
  context_notes: string
}

export interface ModelFeatureImportanceItem {
  feature_name: string
  importance_pct: number
  contribution_direction?: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' | string | null
  description?: string | null
}

export interface ForecastEvidenceResponse {
  crop: string
  strategy_name: string
  certification_status: 'PRODUCTION_READY' | 'CONDITIONAL_PRODUCTION' | 'BASELINE_PRODUCTION' | 'UNSUPPORTED' | string
  model_family?: string | null
  model_version?: string | null
  validation_protocol: string
  mean_mae?: number | null
  baseline_mae?: number | null
  mean_improvement_pct?: number | null
  fold_win_rate_pct?: number | null
  is_ml_strategy: boolean
  empirical_p10_p90_spread?: number | null
  feature_importance: ModelFeatureImportanceItem[]
  operating_rule: string
  fallback_strategy: string
  explanation_notice: string
}
