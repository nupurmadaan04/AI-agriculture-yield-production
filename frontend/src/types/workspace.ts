/**
 * Day 32 Decision Workspace & Scenario Comparison Types
 */

export interface WorkspaceAnalyzeRequest {
  crop: string
  state: string
  district?: string
  forecast_year: number
  selected_scenarios?: string[]
  custom_modifications?: Record<string, number>
}

export interface BaselineForecastSummary {
  forecast_yield_kg_ha: number
  unit: string
  strategy: string
  model_name: string
  model_version: string
  dataset_version: string
  certification_status: string
  is_deterministic: boolean
  fallback_used: boolean
  request_id: string
  provenance_hash: string
  timestamp: string
  semantic_classification: string
}

export interface HistoricalObservationPoint {
  year: number
  observed_yield_kg_ha: number
  source: string
  semantic_classification: string
}

export interface HistoricalReferenceContext {
  crop: string
  state: string
  district?: string
  start_year: number
  end_year: number
  sample_count: number
  historical_mean_yield_kg_ha: number
  historical_median_yield_kg_ha: number
  historical_min_yield_kg_ha: number
  historical_max_yield_kg_ha: number
  historical_std_yield_kg_ha: number
  trend_slope_kg_ha_yr: number
  historical_period: string
  recent_observations: HistoricalObservationPoint[]
  semantic_classification: string
}

export interface ValidationContext {
  strategy_tier: string
  primary_strategy: string
  validation_protocol: string
  validation_period: string
  mae_kg_ha: number
  rmse_kg_ha?: number
  r2_score?: number
  fold_win_rate_pct: number
  mean_improvement_pct: number
  baseline_mae_kg_ha: number
  baseline_strategy: string
  is_ml_certified: boolean
  legacy_benchmark_note?: string
  metric_definitions: Record<string, string>
  semantic_classification: string
}

export interface UncertaintyContext {
  is_available: boolean
  empirical_p10_kg_ha?: number
  empirical_p90_kg_ha?: number
  ensemble_spread_kg_ha?: number
  spread_percentage?: number
  methodology: string
  coverage_wording: string
  disclaimer: string
  limitations: string
  semantic_classification: string
}

export interface MonitoringContext {
  drift_status: string
  monitoring_status: string
  overall_psi: number
  outcome_evaluation_status: 'EVALUATION_AVAILABLE' | 'EVALUATION_UNAVAILABLE'
  observed_outcome_kg_ha?: number
  forecast_error_kg_ha?: number
  signed_bias_kg_ha?: number
  active_alerts: string[]
  semantic_classification: string
}

export interface AttributionFeature {
  feature_name: string
  feature_label: string
  importance_or_shap: number
  interpretation: string
}

export interface AttributionContext {
  is_available: boolean
  attribution_type: 'TREE_SHAP' | 'PERSISTENCE_BASELINE'
  top_features: AttributionFeature[]
  methodology: string
  semantic_classification: string
}

export interface ProvenanceContext {
  prediction_fingerprint: string
  dataset_identifier: string
  model_identifier: string
  strategy_identifier: string
  request_id: string
  audit_reference: string
  semantic_classification: string
}

export interface ScenarioItem {
  scenario_id: string
  scenario_name: string
  scenario_type: string
  scenario_assumption: string
  scenario_output_kg_ha: number
  baseline_output_kg_ha: number
  yield_delta_kg_ha: number
  yield_percent_change: number
  uncertainty_note: string
  empirical_p10_kg_ha?: number
  empirical_p90_kg_ha?: number
  evidence_type: string
  status: string
  limitations: string
  changed_features: Record<string, any>
  is_simulated: boolean
}

export interface ScenarioComparisonRow {
  metric_label: string
  baseline_value: string
  scenario_values: Record<string, string>
}

export interface ScenarioComparisonMatrix {
  scenario_headers: string[]
  rows: ScenarioComparisonRow[]
  disclaimer: string
}

export interface DecisionWorkspaceResponse {
  workspace_id: string
  crop: string
  state: string
  district?: string
  forecast_year: number
  generated_at: string
  baseline_forecast: BaselineForecastSummary
  historical_context: HistoricalReferenceContext
  validation: ValidationContext
  uncertainty: UncertaintyContext
  monitoring: MonitoringContext
  attribution: AttributionContext
  provenance: ProvenanceContext
  scenarios: ScenarioItem[]
  comparison_matrix: ScenarioComparisonMatrix
  limitations: string[]
  decision_support_statement: string
}

export interface ScenarioArchetype {
  id: string
  name: string
  description: string
  deltas: Record<string, number>
}

export interface WorkspaceTemplatesResponse {
  archetypes: ScenarioArchetype[]
  supported_features: Record<string, string>
  parameter_bounds: Record<string, [number, number]>
  disclaimer: string
}
