export interface ModelLeaderboardEntry {
  id: string
  modelName: string
  modelType: 'ml' | 'deterministic' | 'baseline'
  featureSet: string
  randomR2: number
  randomMae: number
  randomRmse: number
  temporalR2: number
  temporalMae: number
  temporalRmse: number
  groupKFoldR2: number
  groupKFoldMae: number
  groupKFoldRmse: number
  status: 'production' | 'benchmark' | 'baseline'
  notes: string
}

export interface FeatureImportanceItem {
  feature: string
  nativeMdi: number
  permutationDeltaR2: number
  description: string
}

export interface EstimationParams {
  mode: 'post-harvest' | 'pre-season'
  stateName: string
  distName?: string
  year: number
  area: number // 1000 ha
  production?: number // 1000 tons (only in post-harvest)
  rainfall?: number
  temperature?: number
  irrigationPct?: number
  fertilizer?: number
}

export interface EstimationResult {
  mode: 'post-harvest' | 'pre-season'
  estimatedYield: number // kg/ha
  deterministicYield?: number // kg/ha
  confidenceRange: [number, number]
  method: string
  statusMessage: string
  requiresAdditionalData?: boolean
}

export interface PostHarvestPredictRequest {
  year: number
  state?: string
  state_code?: number
  area: number
  production: number
  district?: string
}

export interface PostHarvestPredictResponse {
  mode: 'post-harvest'
  model_name: string
  predicted_yield: number
  deterministic_yield: number
  actual_yield?: number | null
  ml_error?: number | null
  deterministic_error?: number | null
  difference: number
  historical_matched: boolean
  matched_district?: string | null
  state: string
  state_code: number
  year: number
  area: number
  production: number
  formula: string
  feature_dependency_warning: string
}

export interface PreSeasonPredictRequest {
  year: number
  state?: string
  state_code?: number
  area: number
  district?: string
}

export interface ValidationMetrics {
  random_r2: number
  random_mae: number
  random_rmse: number
  temporal_r2: number
  temporal_mae: number
  group_kfold_r2: number
  group_kfold_mae?: number
  temporal_r2_improvement_vs_baseline?: string
  temporal_mae_reduction_vs_baseline?: string
  generalization_assessment: string
}

export interface PreSeasonPredictResponse {
  mode: 'pre-season'
  model_name: string
  predicted_yield: number
  actual_yield?: number | null
  ml_error?: number | null
  historical_matched: boolean
  matched_district?: string | null
  state: string
  state_code: number
  year: number
  area: number
  validation_metrics: ValidationMetrics
  warning: string
}

export interface PreSeasonAdvancedPredictRequest {
  year: number
  state?: string
  state_code?: number
  area: number
  district?: string
  total_cropped_area?: number
  rice_area_share?: number
  wheat_area?: number
  cotton_area?: number
  sugarcane_area?: number
  rice_yield_lag1?: number
  rice_yield_roll3?: number
}

export interface UncertaintyEstimate {
  predicted_yield: number
  lower_bound_10th_pct: number
  upper_bound_90th_pct: number
  prediction_spread: number
  methodology: string
}

export interface FeatureContributionItem {
  feature: string
  contribution_score: number
  value: string
}

export interface PreSeasonAdvancedPredictResponse {
  mode: 'pre-season-advanced'
  model_name: string
  predicted_yield: number
  actual_yield?: number | null
  ml_error?: number | null
  historical_matched: boolean
  matched_district?: string | null
  state: string
  state_code: number
  year: number
  area: number
  features_used: Record<string, number>
  uncertainty: UncertaintyEstimate
  feature_contributions: FeatureContributionItem[]
  validation_metrics: ValidationMetrics
  warning: string
}

export interface ModelMetadataItem {
  id: string
  name: string
  model_type: string
  feature_set: string
  features: string[]
  random_r2: number
  temporal_r2: number
  group_kfold_r2: number
  mae: number
  rmse: number
  mode_compatibility: string
  badge: string
  recommended_for: string
  is_post_harvest_only: boolean
}

export interface ModelsListResponse {
  models: ModelMetadataItem[]
}

export interface StateErrorItem {
  state: string
  count: number
  mae: number
  rmse: number
  mape: number
}

export interface TopErrorItem {
  state: string
  district: string
  year: number
  area: number
  production: number
  actual: number
  predicted: number
  absolute_error: number
  percentage_error: number
  root_cause: string
}

export interface YearErrorItem {
  year: number
  actual_avg: number
  predicted_avg: number
  mae: number
  rmse: number
  mape: number
}

export interface ErrorAnalysisResponse {
  worst_performing_states: StateErrorItem[]
  top_extreme_errors: TopErrorItem[]
  yearly_error_stability: YearErrorItem[]
}

export interface PredictionHistoryEntry {
  id: string
  timestamp: string
  mode: 'post-harvest' | 'pre-season' | 'pre-season-advanced'
  state: string
  district?: string
  year: number
  area: number
  production?: number
  modelName: string
  predictedYield: number
  deterministicYield?: number
  actualYield?: number | null
  errorDelta?: number | null
  riskLevel?: string
  riskScore?: number
}

// =========================================================================
// DAY 5 AGRICULTURAL RISK, EXPLAINABILITY & ANOMALY INTERFACES
// =========================================================================

export interface RiskAssessmentRequest {
  year?: number
  state?: string
  state_code?: number
  district?: string
  area?: number
  predicted_yield: number
  lower_bound?: number
  upper_bound?: number
  historical_yield_mean?: number
  historical_yield_std?: number
  anomaly_score?: number
}

export interface RiskComponents {
  uncertainty_risk: number
  historical_deviation_risk: number
  model_error_risk: number
  anomaly_risk: number
}

export interface RiskAssessmentResponse {
  risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
  risk_score: number
  confidence_label: string
  uncertainty_percent: number
  spread: number
  risk_factors: string[]
  explanation: string
  components: RiskComponents
}

export interface ExplainabilityRequest {
  year: number
  state?: string
  state_code?: number
  district?: string
  area: number
  total_cropped_area?: number
  rice_area_share?: number
  wheat_area?: number
  cotton_area?: number
  sugarcane_area?: number
  rice_yield_lag1?: number
  rice_yield_roll3?: number
  features?: Record<string, number>
}

export interface FeatureContributionDetail {
  feature: string
  feature_name: string
  raw_value: string
  contribution_score: number
  direction: 'positive' | 'negative' | 'neutral'
  normalized_percentage: number
}

export interface ExplainabilityResponse {
  predicted_yield: number
  summary: string
  top_positive_factor: string
  top_negative_factor: string
  feature_contributions: FeatureContributionDetail[]
  methodology: string
}

export interface AnomalyDetectionRequest {
  year: number
  state?: string
  state_code?: number
  district?: string
  area: number
  production?: number
  yield?: number
  total_cropped_area?: number
  rice_area_share?: number
  rice_yield_lag1?: number
  rice_yield_roll3?: number
}

export interface AnomalyDetectionResponse {
  is_anomaly: boolean
  anomaly_score: number
  raw_decision_score: number
  severity: 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME'
  yield_z_score?: number | null
  yield_deviation_pct?: number | null
  area_z_score?: number | null
  area_deviation_pct?: number | null
  reasons: string[]
  state: string
  district: string
  year: number
}

export interface StateRiskItem {
  state: string
  state_code: number
  record_count: number
  average_yield: number
  yield_volatility: number
  average_prediction_error_mae: number
  anomaly_rate_pct: number
  average_uncertainty_pct: number
  risk_score: number
  risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
}

export interface StateRiskResponse {
  data: StateRiskItem[]
}

export interface AnomalyFeedItem {
  id: string
  state: string
  district: string
  year: number
  area: number
  production: number
  yield_val: number
  anomaly_score: number
  severity: 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME'
  reason: string
  yield_deviation_pct: number
}

export interface AnomalyFeedResponse {
  data: AnomalyFeedItem[]
}

export interface IntelligenceDashboardResponse {
  total_records: number
  anomalies_detected: number
  high_risk_states_count: number
  moderate_risk_states_count: number
  low_risk_states_count: number
  average_uncertainty_pct: number
  highest_risk_states: StateRiskItem[]
  recent_anomalies: AnomalyFeedItem[]
}
