export interface HorizonForecastItem {
  horizon_years: number
  forecast_year: number
  predicted_yield: number
  lower_bound_p10: number
  upper_bound_p90: number
  prediction_spread: number
  uncertainty_pct: number
}

export interface HistoricalSeriesPoint {
  year: number
  yield: number
}

export interface ForecastYieldResponse {
  state: string
  district: string
  latest_observed_year: number
  latest_observed_yield: number
  historical_series: HistoricalSeriesPoint[]
  forecasts: HorizonForecastItem[]
  model_name: string
  disclaimer: string
}

export interface ForecastYieldRequest {
  state?: string
  district?: string
  horizons?: number[]
}

export interface TrendAnalyzeResponse {
  state: string
  district: string
  linear_slope: number
  theil_sen_slope: number
  mann_kendall_s: number
  p_value: number
  significance: string
  direction: 'STRONG INCREASING' | 'INCREASING' | 'STABLE' | 'DECREASING' | 'STRONG DECREASING'
  observations: number
  first_year: number
  last_year: number
  first_yield: number
  last_yield: number
  total_change_pct: number
  yearly_series: HistoricalSeriesPoint[]
}

export interface StateTrendItem {
  state: string
  state_code: number
  district_count: number
  avg_yield: number
  linear_slope: number
  theil_sen_slope: number
  p_value: number
  significance: string
  direction: string
  total_change_pct: number
}

export interface EarlyWarningAssessResponse {
  state: string
  district: string
  warning_score: number
  severity: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
  trend_direction: string
  trend_slope_kg_ha_yr: number
  trend_significance: string
  forecast_1yr_kg_ha: number
  forecast_change_pct: number
  latest_observed_yield: number
  prediction_spread_pct: number
  is_anomaly: boolean
  anomaly_score: number
  trigger_signals: string[]
  components: {
    trend_signal_score: number
    forecast_signal_score: number
    historical_deviation_score: number
    anomaly_signal_score: number
    prediction_spread_score: number
  }
  disclaimer: string
}

export interface EarlyWarningDashboardResponse {
  total_states_monitored: number
  critical_states_count: number
  high_states_count: number
  moderate_states_count: number
  low_states_count: number
  declining_states_count: number
  declining_states: string[]
  average_forecast_spread_pct: number
  top_priority_warnings: EarlyWarningAssessResponse[]
  state_matrix: EarlyWarningAssessResponse[]
}
