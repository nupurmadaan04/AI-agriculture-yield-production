export interface TemporalSignal {
  metric_name: string;
  record_count: number;
  latest_year: number | null;
  latest_value: number;
  yoy_change_pct: number;
  yoy_change_absolute: number;
  rolling_3yr_mean: number;
  rolling_3yr_std: number;
  rolling_3yr_zscore: number;
  rolling_5yr_mean: number;
  rolling_5yr_std: number;
  rolling_5yr_zscore: number;
  rolling_8yr_mean: number;
  rolling_8yr_std: number;
  trend_slope: number;
  acceleration: number;
  volatility_cv: number;
  historical_mean: number;
  deviation_from_historical: number;
  trajectory: Array<{
    year: number;
    value: number;
    historical_mean: number;
    rolling_3yr: number;
    yoy_change_pct: number;
  }>;
  location?: string;
  state?: string;
  district?: string;
}

export interface EarlyWarning {
  signal_id: string;
  signal_type: string;
  state: string;
  district: string;
  year: number;
  severity: 'INFO' | 'WATCH' | 'ELEVATED' | 'HIGH' | 'CRITICAL';
  trigger_value: number;
  threshold: number;
  unit: string;
  evidence: string[];
  recommended_action: string;
  is_active: boolean;
}

export interface AlertValidationContext {
  r2: number;
  mae: number;
  drift_status: string;
  data_quality_score: number;
}

export interface Alert {
  alert_id: string;
  location: string;
  state: string;
  district: string;
  year: number;
  severity: 'INFO' | 'WATCH' | 'ELEVATED' | 'HIGH' | 'CRITICAL';
  signal_count: number;
  dominant_signal: string;
  supporting_signals: string[];
  evidence_strength: 'LOW' | 'MODERATE' | 'STRONG' | 'VERY_STRONG';
  composite_risk_score: number;
  evidence_chain: string[];
  recommended_action: string;
  model_validation: AlertValidationContext;
  priority_score?: number;
  priority_rank?: number;
  priority_reason?: string;
}

export interface AlertSummary {
  total_alerts: number;
  critical_alerts_count: number;
  high_alerts_count: number;
  elevated_alerts_count: number;
  watch_alerts_count: number;
  info_alerts_count: number;
  states_under_watch_count: number;
  districts_under_watch_count: number;
}

export interface MonitoringOverview {
  active_alerts_count: number;
  high_critical_count: number;
  states_under_watch: number;
  districts_under_watch: number;
  persistent_signals_count: number;
  model_monitoring_status: string;
  latest_observation_year: number;
  summary: AlertSummary;
  scientific_disclaimer: string;
}

export interface StateWarningMapItem {
  state: string;
  state_code: number;
  severity: 'INFO' | 'WATCH' | 'ELEVATED' | 'HIGH' | 'CRITICAL';
  active_signals_count: number;
  critical_count: number;
  high_count: number;
  elevated_count: number;
  watch_count: number;
  average_yield_kg_ha: number;
  total_districts_monitored: number;
  dominant_concern: string;
}

export interface ChangeDetectionResult {
  metric_name: string;
  cusum_analysis: {
    change_detected: boolean;
    change_type: string;
    max_cusum_statistic: number;
    threshold: number;
    inflection_index: number | null;
    cusum_positive: number[];
    cusum_negative: number[];
    scientific_note: string;
  };
  trend_break_analysis: {
    trend_break_detected: boolean;
    inflection_year: number | null;
    pre_break_slope: number;
    post_break_slope: number;
    slope_delta: number;
    scientific_note: string;
  };
  volatility_shift_cv_delta: number;
  overall_change_flag: boolean;
  scientific_disclaimer: string;
  location?: string;
  state?: string;
  district?: string;
}

export interface WarningBacktest {
  total_evaluations: number;
  true_positives: number;
  false_positives: number;
  false_negatives: number;
  true_negatives: number;
  precision: number;
  recall: number;
  f1_score: number;
  false_positive_rate: number;
  false_negative_rate: number;
  alert_frequency_pct: number;
  mean_lead_time_years: number;
  evaluation_years_range: string;
  parameters: {
    yield_drop_threshold_pct: number;
    warning_zscore_threshold: number;
    lead_time_years: number;
  };
  is_chronologically_valid: boolean;
  scientific_disclaimer: string;
}

export interface MonitoringHealth {
  status: string;
  overall_health_score: number;
  data_quality_score: number;
  drift_status: string;
  prediction_mae: number;
  prediction_r2: number;
  calibration_quality: string;
  data_freshness_label: string;
  metrics_breakdown: {
    population_stability_index: number;
    data_completeness_pct: number;
    error_within_10pct_share: number;
    high_error_record_count: number;
    calibration_slope: number;
  };
  scientific_note: string;
}

export interface BacktestRequest {
  yield_drop_threshold_pct: number;
  warning_zscore_threshold: number;
  lead_time_years: number;
}
