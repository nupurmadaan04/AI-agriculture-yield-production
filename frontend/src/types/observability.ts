export interface SystemHealthItem {
  api_status: string
  readiness_status: string
  backend_status: string
  uptime_seconds: number
  process_id: number
  active_threads: number
  cpu_percent: number
  system_cpu_percent: number
  memory_rss_mb: number
  system_memory_percent: number
  environment: string
  timestamp: string
}

export interface RuntimeMetricsResponse {
  total_requests: number
  successful_requests: number
  error_requests: number
  error_rate_pct: number
  rps: number
  min_latency_ms: number
  p50_latency_ms: number
  p90_latency_ms: number
  p95_latency_ms: number
  p99_latency_ms: number
  max_latency_ms: number
  sample_count: number
  active_window_seconds: number
  has_runtime_data: boolean
}

export interface ForecastOperationsMetrics {
  total_forecasts: number
  successful_forecasts: number
  rejected_forecasts: number
  failed_forecasts: number
  success_rate_pct: number
  rejection_rate_pct: number
  forecasts_by_crop: Record<string, number>
  forecasts_by_strategy: Record<string, number>
  forecasts_by_status: Record<string, number>
  recent_forecast_count: number
  has_runtime_data: boolean
}

export interface StrategyUsageItem {
  crop: string
  strategy: string
  certification_status: string
  algorithm: string
  runtime_invocations_count: number
  runtime_percentage: number | null
  fallback_invocations_count: number
  last_used_timestamp: string | null
}

export interface StrategyMonitoringResponse {
  total_strategies_monitored: number
  total_runtime_observations: number
  has_runtime_data: boolean
  strategies: StrategyUsageItem[]
}

export interface TraceStageItem {
  stage_name: string
  status: string
  duration_ms: number | null
  timestamp?: string | null
  details?: Record<string, any> | null
}

export interface PredictionTraceResponse {
  request_id: string
  timestamp: string
  crop: string
  state: string
  district: string
  forecast_year: number
  strategy?: string | null
  model_name?: string | null
  model_version?: string | null
  prediction?: number | null
  unit: string
  status: string
  error_code?: string | null
  error_message?: string | null
  total_duration_ms?: number | null
  stages: TraceStageItem[]
  provenance_hash?: string | null
  audit_status: string
  model_hash_verified: boolean
  dataset_verified: boolean
}

export interface ModelIntegrityItem {
  model_id: string
  crop: string
  algorithm: string
  version: string
  artifact_path: string
  file_exists: boolean
  registered_sha256: string
  actual_sha256: string
  integrity_status: string
  last_checked_timestamp: string
}

export interface ModelIntegrityResponse {
  total_models_registered: number
  verified_models_count: number
  failed_models_count: number
  overall_integrity_status: string
  models: ModelIntegrityItem[]
}

export interface DatasetIntegrityResponse {
  dataset_name: string
  dataset_version: string
  file_path: string
  file_exists: boolean
  total_records: number
  total_columns: number
  file_size_bytes: number
  sha256_checksum: string
  date_coverage: string
  state_count: number
  district_count: number
  crop_count: number
  schema_status: string
  last_verified_timestamp: string
}

export interface StrategyRegistryHealthResponse {
  registry_path: string
  registry_available: boolean
  compiled_at: string
  total_strategies_registered: number
  production_ready_count: number
  conditional_production_count: number
  baseline_production_count: number
  coverage_records_count: number
  certification_guard_status: string
  overall_status: string
}

export interface OperationalEventItem {
  timestamp: string
  severity: string
  event_type: string
  request_id?: string | null
  endpoint?: string | null
  status_code?: number | null
  message: string
  details?: Record<string, any> | null
}

export interface OperationalErrorsResponse {
  total_events_logged: number
  total_errors_count: number
  recent_events: OperationalEventItem[]
  errors_by_category: Record<string, number>
}

export interface AlertConditionItem {
  alert_id: string
  alert_name: string
  severity: string
  metric_name: string
  current_value: any
  threshold_value: any
  condition: string
  is_active: boolean
  message: string
  timestamp: string
}

export interface AlertsResponse {
  active_alerts_count: number
  active_alerts: AlertConditionItem[]
  resolved_alerts: AlertConditionItem[]
  configured_thresholds: Record<string, any>
}

export interface DriftMonitoringResponse {
  monitoring_notice: string
  overall_drift_status: string
  total_features_evaluated: number
  stable_features_count: number
  drifted_features_count: number
  features: Array<Record<string, any>>
}

export interface ObservabilitySummaryResponse {
  system_health: SystemHealthItem
  runtime_metrics: RuntimeMetricsResponse
  forecast_operations: ForecastOperationsMetrics
  model_integrity_status: string
  dataset_integrity_status: string
  strategy_registry_status: string
  active_alerts_count: number
  timestamp: string
}
