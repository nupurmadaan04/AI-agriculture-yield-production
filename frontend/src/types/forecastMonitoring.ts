/**
 * Day 30: Forecast Monitoring, Drift Detection & Outcome Intelligence TypeScript Definitions
 */

export interface MonitoringSummaryResponse {
  monitoring_status: 'HEALTHY' | 'WATCH' | 'DRIFT_DETECTED' | 'EVALUATION_UNAVAILABLE' | 'INSUFFICIENT_EVIDENCE' | string
  status_reason: string
  total_forecast_requests: number
  successful_forecasts: number
  rejected_requests: number
  evaluated_outcomes_count: number
  active_alerts_count: number
  monitored_crops_count: number
  dataset_version: string
  timestamp: string
}

export interface OperationalTimeSeriesPoint {
  date: string
  total_requests: number
  successful_requests: number
  rejected_requests: number
}

export interface CropUsageItem {
  crop: string
  request_count: number
  percentage: number
}

export interface StrategyUsageItem {
  strategy: string
  certification_status: string
  request_count: number
  percentage: number
}

export interface ForecastOperationsResponse {
  total_requests: number
  successful_requests: number
  rejected_requests: number
  failed_requests: number
  success_rate_pct: number
  time_series: OperationalTimeSeriesPoint[]
  crop_breakdown: CropUsageItem[]
  strategy_breakdown: StrategyUsageItem[]
  semantic_classification: string
  notes: string
}

export interface StatisticalMoments {
  count: number
  mean: number
  median: number
  std: number
  min_val: number
  max_val: number
  p10?: number
  p25?: number
  p75?: number
  p90?: number
}

export interface DistributionHistogramBin {
  bin_start: number
  bin_end: number
  count: number
  density: number
}

export interface PredictionDistributionItem {
  crop: string
  strategy: string
  unit: string
  current_predictions: StatisticalMoments
  historical_reference: StatisticalMoments
  histogram_bins: DistributionHistogramBin[]
  distribution_shift_detected: boolean
  shift_metric?: string
  shift_value?: number
  semantic_classification: string
}

export interface PredictionDistributionResponse {
  total_monitored_crops: number
  distributions: PredictionDistributionItem[]
  semantic_classification: string
  evaluation_window: string
  reference_window: string
}

export interface FeatureDriftItem {
  feature_name: string
  metric: string
  observed_value: number
  p_value?: number
  threshold: number
  status: 'NO_DRIFT' | 'MODERATE_DRIFT' | 'SIGNIFICANT_DRIFT' | 'MONITORING_ONLY' | string
  reference_window: string
  evaluation_window: string
  reference_samples: number
  evaluation_samples: number
  evidence: string
  semantic_classification: string
}

export interface CoverageDriftItem {
  dimension: string
  reference_count: number
  current_count: number
  coverage_ratio: number
  status: string
  notes: string
}

export interface DriftMonitoringResponse {
  overall_drift_status: string
  features: FeatureDriftItem[]
  coverage_drift: CoverageDriftItem[]
  missingness_drift_pct: number
  threshold_source: string
  semantic_classification: string
  notes: string
}

export interface OutcomeEvaluationItem {
  crop: string
  state: string
  district: string
  forecast_year: number
  forecast_origin: number
  predicted_yield: number
  observed_yield: number
  signed_error: number
  absolute_error: number
  relative_error_pct?: number
  strategy: string
  model_version: string
  unit: string
  evaluation_status: string
  semantic_classification: string
}

export interface OutcomeEvaluationSummary {
  crop: string
  evaluated_samples: number
  mae: number
  rmse: number
  median_absolute_error: number
  mean_signed_bias: number
  mape?: number
  evaluation_years: number[]
  status: string
}

export interface OutcomeEvaluationResponse {
  status: 'EVALUATED' | 'EVALUATION_UNAVAILABLE' | 'INSUFFICIENT_EVIDENCE' | string
  reason: string
  summary?: OutcomeEvaluationSummary
  records: OutcomeEvaluationItem[]
  total_records: number
  temporal_boundary_rule: string
  semantic_classification: string
}

export interface TemporalErrorItem {
  year: number
  evaluated_forecasts: number
  mae: number
  rmse: number
  bias: number
  p25_error: number
  p75_error: number
}

export interface GeographicErrorItem {
  state: string
  district: string
  evaluated_forecasts: number
  mae: number
  rmse: number
  bias: number
}

export interface RegimeErrorItem {
  regime: string
  sample_count: number
  mae: number
  rmse: number
  mean_signed_bias: number
}

export interface ErrorDecompositionResponse {
  crop: string
  temporal_breakdown: TemporalErrorItem[]
  geographic_breakdown: GeographicErrorItem[]
  regime_breakdown: RegimeErrorItem[]
  semantic_classification: string
  notes: string
}

export interface CropBiasItem {
  crop: string
  mean_actual_yield: number
  mean_residual: number
  median_residual: number
  normalized_mean_error_pct: number
  bias_status: 'OVER_PREDICTION_BIAS' | 'UNDER_PREDICTION_BIAS' | 'NO_CLEAR_BIAS' | string
  bias_description: string
  bias_threshold_rule: string
  sample_count: number
  semantic_classification: string
}

export interface BiasAnalysisResponse {
  crops: CropBiasItem[]
  methodology: string
  semantic_classification: string
  notes: string
}

export interface MonitoringAlertItem {
  alert_id: string
  timestamp: string
  severity: 'INFO' | 'WATCH' | 'WARNING' | 'CRITICAL' | string
  category: 'DRIFT' | 'BIAS' | 'OPERATIONAL' | 'INTEGRITY' | 'DATASET' | string
  signal: string
  metric: string
  observed_value: string
  threshold?: string
  reference_window?: string
  evaluation_window?: string
  sample_size?: number
  crop?: string
  evidence: string
  recommended_action: string
}

export interface MonitoringAlertsResponse {
  active_alerts: MonitoringAlertItem[]
  total_alerts: number
  has_critical_alerts: boolean
  timestamp: string
  semantic_classification: string
}

export interface MonitoringHealthResponse {
  status: string
  subsystem: string
  version: string
  audit_records_available: number
  telemetry_records_available: number
  outcomes_dataset_available: boolean
  timestamp: string
}
