export interface CropReadinessItem {
  crop: string
  readiness_score: number
  readiness_status: 'MODEL_READY' | 'ANALYTICS_READY' | 'INSUFFICIENT_DATA' | string
  data_volume_score: number
  temporal_score: number
  geographic_score: number
  target_quality_score: number
  validation_score: number
  feature_score: number
  blocking_reasons: string
  recommendation: string
  total_records?: number
  active_districts?: number
  zero_yield_pct?: number
  best_baseline_model?: string
  best_baseline_mae?: number
  best_baseline_r2?: number
}

export interface CropReadinessResponse {
  total_crops: number
  crops: CropReadinessItem[]
}

export interface CropBaselineItem {
  crop: string
  model: string
  train_period: string
  test_period: string
  train_records: number
  test_records: number
  mae: number | null
  rmse: number | null
  r2: number | null
  mape: number | null
  smape: number | null
  valid_predictions: number
  invalid_predictions: number
  notes: string
  status: string
}

export interface CropBaselinesResponse {
  crop: string
  baselines: CropBaselineItem[]
  best_model_by_mae: string | null
  best_mae: number | null
}

export interface ReadinessSummaryResponse {
  total_crops: number
  model_ready_count: number
  analytics_ready_count: number
  insufficient_data_count: number
  model_ready_crops: string[]
  analytics_ready_crops: string[]
  insufficient_data_crops: string[]
  total_records_evaluated: number
  active_dataset_version: string
}

export interface FeatureCompatibilityItem {
  feature_name: string
  source_type: string
  timing: string
  pre_season_valid: boolean
  leakage_risk: string
  classification: string
  recommendation: string
}

export interface FeatureCompatibilityResponse {
  total_features_audited: number
  features: FeatureCompatibilityItem[]
}

export interface ArchitectureDecisionResponse {
  decision: string
  recommended_architecture: string
  global_model_justified: boolean
  crop_specific_justified: boolean
  hierarchical_justified: boolean
  summary: string
  empirical_justification: string[]
}

// ---------------------------------------------------------------------------
// Day 19 Multi-Crop Forecasting Types
// ---------------------------------------------------------------------------

export interface MultiCropModelItem {
  crop: string
  model_id: string
  algorithm: string
  version: string
  training_period: string
  evaluation_period: string
  train_records: number
  test_records: number
  mae: number
  rmse: number
  r2: number
  mape?: number | null
  smape?: number | null
  baseline_model: string
  baseline_mae: number
  mae_improvement_pct: number
  model_status: 'ACCEPTED' | 'BASELINE_PREFERRED' | 'EXPERIMENTAL_ONLY' | string
  recommendation: string
  artifact_path: string
  sha256: string
}

export interface MultiCropModelsResponse {
  total_models: number
  accepted_count: number
  baseline_preferred_count: number
  models: MultiCropModelItem[]
}

export interface CropModelComparisonResponse {
  crop: string
  baseline_model: string
  baseline_mae: number
  baseline_rmse: number
  baseline_r2?: number | null
  rf_mae: number
  rf_rmse: number
  rf_r2: number
  gb_mae: number
  gb_rmse: number
  gb_r2: number
  ml_winner: string
  overall_winner: string
  best_mae: number
  mae_improvement_vs_baseline: number
  mae_improvement_pct: number
  model_status: string
  recommendation: string
}

export interface CropModelMetricsResponse {
  crop: string
  algorithm: string
  status: string
  training_period: string
  evaluation_period: string
  train_records: number
  test_records: number
  metrics: {
    mae: number
    rmse: number
    r2: number
    mape?: number | null
    smape?: number | null
  }
  baseline_comparison: {
    baseline_model: string
    baseline_mae: number
    baseline_rmse: number
    baseline_r2?: number | null
    mae_improvement_abs: number
    mae_improvement_pct: number
  }
  error_analysis: {
    p25: number
    p50_median: number
    p75: number
    p90: number
    low_error_pct: number
    moderate_error_pct: number
    high_error_pct: number
  }
  uncertainty_spread_p10_p90?: number | null
}

export interface CropModelFeaturesResponse {
  crop: string
  algorithm: string
  features: string[]
  feature_importance_native: Record<string, number>
  feature_importance_permutation: Record<string, number>
}

export interface MultiCropLeaderboardItem {
  crop: string
  best_model: string
  best_mae: number
  best_rmse: number
  best_r2?: number | null
  baseline_model: string
  baseline_mae: number
  mae_improvement_pct: number
  model_status: string
}

export interface MultiCropLeaderboardResponse {
  total_crops: number
  accepted_count: number
  baseline_preferred_count: number
  leaderboard: MultiCropLeaderboardItem[]
}

export interface CropPredictionRequest {
  crop: string
  state: string
  district: string
  year?: number
  yield_lag_1?: number | null
  yield_lag_2?: number | null
  yield_rolling_3yr_mean?: number | null
  area_lag_1?: number | null
}

export interface CropPredictionResponse {
  crop: string
  state: string
  district: string
  target_year: number
  predicted_yield_kg_ha: number
  p10_lower_kg_ha?: number | null
  p90_upper_kg_ha?: number | null
  model_id: string
  algorithm: string
  model_version: string
  model_scope: string
  model_status: string
  dataset_version: string
  provenance: {
    training_period: string
    input_features: Record<string, any>
    sha256: string
    baseline_benchmark_mae: number
  }
}

// ---------------------------------------------------------------------------
// Day 20 Temporal Robustness Types
// ---------------------------------------------------------------------------

export interface FoldResultItem {
  crop: string
  fold_id: number
  train_start_year: number
  train_end_year: number
  test_year: number
  train_samples: number
  test_samples: number
  model: string
  mae: number
  rmse: number
  r2: number
  mape?: number | null
  smape?: number | null
  best_baseline_model: string
  best_baseline_mae: number
  win_vs_baseline: boolean
  mae_improvement_pct: number
  mean_residual: number
  std_residual: number
}

export interface CropRobustnessItem {
  crop: string
  model: string
  fold_count: number
  mean_mae: number
  median_mae: number
  std_mae: number
  mean_rmse: number
  std_rmse: number
  mean_r2: number
  std_r2: number
  baseline_mae: number
  mean_mae_improvement: number
  median_mae_improvement: number
  win_rate: number
  status: 'ROBUST_ACCEPTED' | 'SPLIT_SENSITIVE' | 'BASELINE_PREFERRED' | string
  robustness_score?: number | null
}

export interface CropRobustnessResponse {
  total_crops: number
  robust_accepted_count: number
  split_sensitive_count: number
  baseline_preferred_count: number
  crops: CropRobustnessItem[]
}

export interface CropFoldsResponse {
  crop: string
  fold_count: number
  folds: FoldResultItem[]
}

export interface CropRobustnessDetailResponse {
  crop: string
  robustness_status: string
  robustness_score: number
  best_model: string
  evaluated_ml_model: string
  recommendation: string
  models: Record<string, any>
  folds: FoldResultItem[]
  feature_stability: Array<{
    feature: string
    stability_rank: number
    status: string
  }>
}

export interface RobustnessSummaryResponse {
  total_crops_evaluated: number
  total_walk_forward_folds: number
  robust_accepted_count: number
  split_sensitive_count: number
  baseline_preferred_count: number
  robust_accepted_crops: string[]
  split_sensitive_crops: string[]
  baseline_preferred_crops: string[]
  mean_win_rate: number
  dataset_version: string
}

// ==========================================
// Day 21: Model Diagnosis & Strategy Types
// ==========================================

export interface CropDiagnosisItem {
  crop: string
  selected_ml_model: string
  total_folds: number
  total_test_observations: number
  historical_median_yield: number
  ml_wins: number
  ml_losses: number
  win_rate: number
  mean_mae_improvement_pct: number
  median_mae_improvement_pct: number
  worst_fold_degradation_pct: number
  best_fold_improvement_pct: number
  ml_mae_mean: number
  ml_mae_median: number
  ml_mae_std: number
  ml_mae_min: number
  ml_mae_max: number
  ml_mae_cv: number
  base_mae_mean: number
  base_mae_median: number
  base_mae_std: number
  base_mae_min: number
  base_mae_max: number
  base_mae_cv: number
  p25_error_ml: number
  p50_error_ml: number
  p75_error_ml: number
  p90_error_ml: number
  p95_error_ml: number
  p25_error_base: number
  p50_error_base: number
  p75_error_base: number
  p90_error_base: number
  p95_error_base: number
  pct_errors_lt_100: number
  pct_errors_lt_250: number
  pct_errors_lt_500: number
  pct_errors_gt_1000: number
  normalized_error_p50: number
  normalized_error_p90: number
}

export interface CropDiagnosisSummaryResponse {
  total_crops: number
  crops: CropDiagnosisItem[]
  methodology_version: string
}

export interface CropErrorRegimeItem {
  crop: string
  regime: string
  n_observations: number
  yield_min_kg_ha: number
  yield_max_kg_ha: number
  ml_mae: number
  baseline_mae: number
  ml_improvement_pct: number
  obs_win_rate: number
  regime_status: 'ML_ADVANTAGE' | 'BASELINE_ADVANTAGE' | string
}

export interface CropErrorRegimesResponse {
  crop: string
  regimes: CropErrorRegimeItem[]
}

export interface CropDistrictErrorItem {
  crop: string
  state: string
  district: string
  observations_count: number
  ml_mae: number
  baseline_mae: number
  ml_improvement_pct: number
  ml_win_rate: number
  is_best_ml_district: boolean
  is_worst_ml_district: boolean
  is_high_error_district: boolean
}

export interface CropDistrictErrorsResponse {
  crop: string
  total_districts: number
  best_ml_districts_count: number
  worst_ml_districts_count: number
  high_error_districts_count: number
  districts: CropDistrictErrorItem[]
}

export interface CropYearErrorItem {
  crop: string
  year: number
  fold_id: number
  ml_model: string
  ml_mae: number
  ml_rmse: number
  baseline_model: string
  baseline_mae: number
  baseline_rmse: number
  ml_improvement_pct: number
  ml_win: boolean
  temporal_regime: string
}

export interface CropYearErrorsResponse {
  crop: string
  years: CropYearErrorItem[]
}

export interface CropFeatureStabilityItem {
  crop: string
  feature: string
  model: string
  mean_importance: number
  std_importance: number
  mean_rank: number
  rank_variance: number
  feature_stability_score: number
  interpretative_role: string
}

export interface CropFeatureStabilityResponse {
  crop: string
  features: CropFeatureStabilityItem[]
  feature_timing_audit: Array<{
    feature: string
    observation_time: string
    available_before_forecast: boolean
    fold_safe: boolean
    timing_status: 'SAFE' | 'CONDITIONALLY_SAFE' | 'UNSAFE' | 'UNKNOWN' | string
    timing_notes: string
  }>
}

export interface CropModelSelectionItem {
  crop: string
  day19_status: string
  day20_status: string
  day21_status: 'ROBUST_ML' | 'ML_WITH_CONDITIONS' | 'BASELINE_PREFERRED' | 'RESEARCH_CANDIDATE' | 'INSUFFICIENT_EVIDENCE' | string
  win_rate: number
  mean_mae_improvement_pct: number
  median_mae_improvement_pct: number
  worst_fold_degradation_pct: number
  ml_mae_cv: number
  feature_timing_status: string
  total_test_observations: number
  decision_basis: string
  methodology_version: string
}

export interface CropModelSelectionResponse {
  total_crops: number
  robust_ml_count: number
  ml_with_conditions_count: number
  baseline_preferred_count: number
  research_candidate_count: number
  insufficient_evidence_count: number
  selections: CropModelSelectionItem[]
}

export interface CropForecastingStrategyItem {
  crop: string
  day21_status: string
  primary_forecasting_model: string
  fallback_model: string
  operating_conditions: string
  diagnostic_notes: string
  missing_information_gaps: string
  evidence_required: string
  evidence_strength_score: number
  evidence_interpretation: string
}

export interface CropForecastingStrategyResponse {
  total_crops: number
  strategies: CropForecastingStrategyItem[]
}

// ---------------------------------------------------------------------------
// Day 22 Exogenous Data & Pre-Season Feature Expansion Types
// ---------------------------------------------------------------------------

export interface ExogenousSourceItem {
  source_id: string
  source_name: string
  provider: string
  tier: string
  spatial_level: string
  temporal_resolution: string
  temporal_coverage: string
  status: string
  preseason_availability_date: string
  license_terms: string
}

export interface ExogenousSourcesResponse {
  total_sources: number
  sources: ExogenousSourceItem[]
}

export interface ExogenousCoverageItem {
  crop: string
  records: number
  weather_coverage_pct: number
  soil_coverage_pct: number
  overall_exogenous_coverage_pct: number
  missing_pct: number
  district_coverage: number
  year_coverage: number
  coverage_status: 'EXCELLENT_COVERAGE' | 'ADEQUATE_COVERAGE' | 'INSUFFICIENT_COVERAGE' | string
}

export interface ExogenousCoverageResponse {
  total_crops: number
  crops: ExogenousCoverageItem[]
}

export interface ExogenousFeatureItem {
  feature: string
  source: string
  spatial_level: string
  temporal_resolution: string
  observation_period: string
  availability_date: string
  forecast_origin: string
  lag: number
  unit: string
  transformation: string
  leakage_status: 'SAFE' | 'CONDITIONALLY_SAFE' | 'UNSAFE' | 'UNKNOWN' | string
}

export interface ExogenousFeaturesResponse {
  total_features: number
  features: ExogenousFeatureItem[]
  temporal_audit: Array<{
    feature: string
    observation_period: string
    availability_date: string
    forecast_origin: string
    lag: number
    is_preseason_available: boolean
    is_fold_safe: boolean
    timing_status: string
    rationale: string
  }>
  leakage_audit: Array<{
    check_id: string
    description: string
    scope: string
    tested_condition: string
    leakage_risk: string
    audit_result: string
    status: string
  }>
}

export interface ExogenousAblationItem {
  crop: string
  experiment_id: string
  experiment_name: string
  feature_group: string
  algorithm: string
  n_features: number
  mean_mae: number
  median_mae: number
  mean_rmse: number
  mean_r2: number
  mean_mape: number
  mean_smape: number
  baseline_mean_mae: number
  historical_mean_mae: number
  mean_improvement_vs_baseline_pct: number
  median_improvement_vs_baseline_pct: number
  mean_improvement_vs_historical_pct: number
  win_rate_vs_baseline: number
  win_rate_vs_historical: number
}

export interface ExogenousAblationResponse {
  total_records: number
  ablations: ExogenousAblationItem[]
}

export interface ExogenousFoldResultItem {
  crop: string
  fold_id: number
  test_year: number
  experiment_id: string
  experiment_name: string
  feature_group: string
  n_features: number
  algorithm: string
  test_records: number
  mae: number
  rmse: number
  r2: number
  mape: number
  smape: number
  baseline_mae: number
  historical_mae: number
  improvement_vs_baseline_pct: number
  improvement_vs_historical_pct: number
  win_vs_baseline: boolean
  win_vs_historical: boolean
}

export interface ExogenousCropFoldsResponse {
  crop: string
  total_folds: number
  folds: ExogenousFoldResultItem[]
}

export interface ExogenousCropResultItem {
  crop: string
  algorithm: string
  model_a_hist_mae: number
  model_a_hist_rmse: number
  model_a_hist_r2: number
  model_b_exo_mae: number
  model_b_exo_rmse: number
  model_b_exo_r2: number
  model_c_base_mae: number
  model_c_base_rmse: number
  exogenous_gain_vs_historical_pct: number
  exogenous_gain_vs_baseline_pct: number
  artifact_path: string
}

export interface ExogenousCropResultResponse {
  crop: string
  result: ExogenousCropResultItem
  folds: ExogenousFoldResultItem[]
  ablation_tiers: ExogenousAblationItem[]
}

export interface ExogenousModelSelectionItem {
  crop: string
  day21_status: string
  day22_status: 'EXOGENOUS_ROBUST' | 'EXOGENOUS_CONDITIONAL' | 'NO_MEANINGFUL_GAIN' | 'INSUFFICIENT_COVERAGE' | string
  model_a_hist_mae: number
  model_b_exo_mae: number
  model_c_base_mae: number
  gain_vs_historical_pct: number
  gain_vs_baseline_pct: number
  win_rate_vs_historical: number
  win_rate_vs_baseline: number
  shock_year_2016_gain_pct: number
  best_ablation_tier: string
  decision_basis: string
}

export interface ExogenousModelSelectionResponse {
  total_crops: number
  exogenous_robust_count: number
  exogenous_conditional_count: number
  no_meaningful_gain_count: number
  insufficient_coverage_count: number
  selections: ExogenousModelSelectionItem[]
}

export interface ExogenousSummaryResponse {
  total_crops_evaluated: number
  total_sources: number
  total_exogenous_features: number
  exogenous_robust_count: number
  exogenous_conditional_count: number
  no_meaningful_gain_count: number
  insufficient_coverage_count: number
  largest_mae_improvement_crop: string
  largest_mae_improvement_pct: number
  shock_year_2016_average_gain_pct: number
  methodology_version: string
}

// ============================================================================
// DAY 23: FINAL VALIDATION, RESIDUAL DIAGNOSTICS & MODEL CERTIFICATION
// ============================================================================

export interface FinalStrategyItem {
  crop: string
  primary_model: string
  fallback_model: string
  policy_type: string
  strategy_mean_mae: number
  ml_mean_mae: number
  baseline_mean_mae: number
  strategy_gain_vs_baseline_pct: number
  strategy_gain_vs_ml_pct: number
  operating_rule: string
}

export interface FinalValidationFoldItem {
  crop: string
  fold_id: number
  test_year: number
  test_records: number
  policy_type: string
  strategy_mae: number
  strategy_rmse: number
  strategy_r2: number
  strategy_mape: number
  strategy_smape: number
  ml_mae: number
  ml_rmse: number
  ml_r2: number
  ml_mape: number
  baseline_mae: number
  baseline_rmse: number
  baseline_r2: number
  baseline_mape: number
  strategy_improvement_vs_baseline_pct: number
  strategy_improvement_vs_ml_pct: number
  strategy_win_vs_baseline: boolean
  strategy_win_vs_ml: boolean
}

export interface FinalValidationResponse {
  total_crops_certified: number
  production_ready_count: number
  conditional_production_count: number
  baseline_production_count: number
  research_only_count: number
  not_ready_count: number
  temporal_range_statement: string
  strategies: FinalStrategyItem[]
}

export interface SingleCropFinalValidationResponse {
  crop: string
  strategy: FinalStrategyItem
  folds: FinalValidationFoldItem[]
}

export interface ResidualQuantileItem {
  crop: string
  total_eval_samples: number
  mean_residual: number
  median_residual: number
  std_residual: number
  mae: number
  rmse: number
  p25_abs_error: number
  p50_abs_error: number
  p75_abs_error: number
  p90_abs_error: number
  p95_abs_error: number
  mae_q1_lowest_yield: number
  mae_q2_lower_mid_yield: number
  mae_q3_upper_mid_yield: number
  mae_q4_highest_yield: number
}

export interface ResidualYearItem {
  crop: string
  year: number
  fold_id: number
  records: number
  mean_residual: number
  median_residual: number
  mae: number
  rmse: number
  p90_abs_error: number
  temporal_regime: string
}

export interface ResidualDiagnosticsResponse {
  crop: string
  quantiles: ResidualQuantileItem
  years: ResidualYearItem[]
}

export interface PredictionBiasItem {
  crop: string
  mean_actual_yield: number
  mean_residual: number
  median_residual: number
  normalized_mean_error_pct: number
  bias_status: 'OVER_PREDICTION_BIAS' | 'UNDER_PREDICTION_BIAS' | 'NO_CLEAR_BIAS' | 'INSUFFICIENT_EVIDENCE' | string
  bias_description: string
  bias_threshold_rule: string
}

export interface PredictionBiasResponse {
  crop: string
  bias: PredictionBiasItem
}

export interface ReproducibilityItem {
  crop: string
  dataset_sha256: string
  feature_matrix_sha256: string
  model_artifact_sha256: string
  run1_prediction_hash: string
  run2_prediction_hash: string
  max_absolute_prediction_diff: number
  bitwise_reproducible: boolean
  status: 'VERIFIED_BITWISE' | 'REPRODUCIBILITY_FAILED' | string
}

export interface ReproducibilityResponse {
  audit_title: string
  overall_status: string
  total_crops_audited: number
  verified_bitwise_count: number
  reproducibility_rate_pct: number
  crops: ReproducibilityItem[]
}

export interface FinalModelCertificationItem {
  crop: string
  final_status: 'PRODUCTION_READY' | 'CONDITIONAL_PRODUCTION' | 'BASELINE_PRODUCTION' | 'RESEARCH_ONLY' | 'NOT_READY' | string
  primary_strategy: string
  fallback_strategy: string
  strategy_mae: number
  ml_mae: number
  baseline_mae: number
  gain_vs_baseline_pct: number
  gain_vs_ml_pct: number
  fold_win_rate_pct: number
  mean_residual: number
  p90_abs_error: number
  bias_status: string
  reproducibility_status: string
  feature_timing_safety: string
  operational_evidence: string
  governance_directive: string
}

export interface FinalModelCertificationResponse {
  total_crops_certified: number
  production_ready_count: number
  conditional_production_count: number
  baseline_production_count: number
  research_only_count: number
  not_ready_count: number
  certifications: FinalModelCertificationItem[]
}

// ---------------------------------------------------------------------------
// Day 24 Production Forecast Serving & Governance Types
// ---------------------------------------------------------------------------

export interface ForecastStrategyItem {
  crop: string
  certification_status: 'PRODUCTION_READY' | 'CONDITIONAL_PRODUCTION' | 'BASELINE_PRODUCTION' | 'RESEARCH_ONLY' | 'NOT_READY' | string
  primary_strategy: string
  fallback_strategy: string
  strategy_mae: number
  baseline_mae: number
  gain_vs_baseline_pct: number
  fold_win_rate_pct: number
  bias_status: string
  reproducibility_status: string
  model_name: string
  model_version: string
  model_artifact?: string | null
  validation_scope: string
  operating_rule: string
  strategy_explanation: string
}

export interface ForecastStrategiesResponse {
  total_strategies: number
  version: string
  validation_scope: string
  temporal_boundary: string
  strategies: ForecastStrategyItem[]
}

export interface ForecastCoverageItem {
  crop: string
  state: string
  district: string
  min_year: number
  max_year: number
  total_observations: number
}

export interface ForecastCoverageResponse {
  total_records: number
  unique_crops: string[]
  unique_states: string[]
  unique_districts_count: number
  coverage: ForecastCoverageItem[]
}

export interface ForecastCertificationSummaryResponse {
  total_crops_certified: number
  production_ready_crops: string[]
  conditional_production_crops: string[]
  baseline_production_crops: string[]
  governance_policy: string
  certification_source: string
}

export interface ForecastPredictRequest {
  crop: string
  state: string
  district: string
  forecast_year?: number
  yield_lag_1?: number
  yield_rolling_3yr_mean?: number
  area_lag_1?: number
}

export interface ForecastPredictResponse {
  status: 'SUCCESS' | 'REJECTED' | string
  request_id: string
  prediction?: number | null
  unit: string
  crop: string
  state: string
  district: string
  forecast_year?: number
  strategy?: string | null
  certification_status: string
  fallback_used: boolean
  fallback_reason?: string | null
  model_version?: string | null
  validation_scope: string
  evidence_type: string
  operating_rule?: string | null
  strategy_explanation?: string | null
  error_code?: string | null
  error_message?: string | null
  provenance?: {
    request_id: string
    timestamp: string
    crop: string
    state: string
    district: string
    forecast_year: number
    strategy: string
    model_name: string
    model_version: string
    model_artifact_hash: string
    certification_status: string
    fallback_used: boolean
    fallback_reason?: string | null
    features_used: Record<string, any>
    data_source: string
    validation_period: string
    validation_mae: number
    baseline_mae: number
    gain_vs_baseline_pct: number
    fold_win_rate_pct: number
    prediction: number
    unit: string
    evidence_type: string
    operating_rule: string
    strategy_explanation: string
    validation_boundary_notice: string
    provenance_hash: string
  } | null
}

export interface ForecastAuditItem {
  request_id: string
  timestamp: string
  crop: string
  state: string
  district: string
  forecast_year: number
  strategy: string
  certification_status: string
  status: string
  prediction?: number | string | null
  unit: string
  fallback_used: boolean
  provenance_hash?: string
  error_code?: string
  error_message?: string
}

export interface ForecastAuditResponse {
  total_events: number
  events: ForecastAuditItem[]
}

export interface ForecastHealthResponse {
  status: string
  service: string
  version: string
  certified_crops_count: number
  coverage_districts_count: number
  governance_guard: string
  provenance_tracking: string
}




