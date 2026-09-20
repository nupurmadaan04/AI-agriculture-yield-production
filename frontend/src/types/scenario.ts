export interface FeatureChangeItem {
  feature_key: string;
  feature_name: string;
  baseline_value: number;
  scenario_value: number;
  absolute_change: number;
  percent_change: number;
}

export interface ValidationContext {
  model_version: string;
  dataset_version: string;
  validation_r2: number;
  validation_mae: number;
  validation_rmse: number;
  drift_status: string;
  data_quality_score: number;
  spread_type: string;
}

export interface ScenarioResult {
  scenario_id: string;
  location: string;
  state?: string;
  district?: string;
  horizon: number;
  scenario_type: string;
  scenario_name: string;
  baseline_prediction: number;
  scenario_prediction: number;
  yield_delta: number;
  yield_percent_change: number;
  risk_score: number;
  risk_delta: number;
  warning_score: number;
  warning_delta: number;
  prediction_spread: number;
  lower_bound_p10: number;
  upper_bound_p90: number;
  changed_features: FeatureChangeItem[];
  unsupported_features_requested: string[];
  validation_context: ValidationContext;
  scientific_disclaimer: string;
}

export interface ScenarioComparisonItem {
  scenario_id: string;
  scenario_type: string;
  scenario_name: string;
  projected_yield: number;
  yield_delta: number;
  yield_percent_change: number;
  risk_score: number;
  risk_delta: number;
  warning_score: number;
  warning_delta: number;
  prediction_spread: number;
  spread_delta: number;
  is_baseline: boolean;
  interpretation: string;
}

export interface ScenarioComparisonResult {
  location: string;
  horizon: number;
  scenarios_compared_count: number;
  baseline_yield: number;
  highest_yield_scenario: string;
  lowest_risk_scenario: string;
  yield_range_kg_ha: number;
  comparison_matrix: ScenarioComparisonItem[];
  scientific_disclaimer: string;
}

export interface PerturbationStep {
  perturbation_pct: number;
  perturbed_input_value: number;
  predicted_yield: number;
  yield_delta: number;
  yield_percent_change: number;
}

export interface FeatureSensitivityItem {
  feature_key: string;
  feature_name: string;
  baseline_value: number;
  elasticity_index: number;
  sensitivity_rank: number;
  perturbation_responses: PerturbationStep[];
}

export interface SensitivityResult {
  location: string;
  horizon: number;
  baseline_prediction: number;
  perturbation_steps: number[];
  features_analyzed: number;
  most_sensitive_feature: string;
  sensitivity_matrix: FeatureSensitivityItem[];
  scientific_disclaimer: string;
}

export interface OptimizationCandidate {
  scenario_id: string;
  scenario_name: string;
  projected_yield: number;
  yield_delta: number;
  yield_percent_change: number;
  risk_score: number;
  resource_change_pct: number;
  decision_score: number;
  rank: number;
  is_feasible: boolean;
  is_pareto_optimal: boolean;
  constraint_status: Record<string, { target: number | null; actual: number; passed: boolean }>;
  strengths: string[];
  limitations: string[];
  tradeoff_summary: string;
}

export interface OptimizationResult {
  location: string;
  horizon: number;
  baseline_yield: number;
  baseline_risk: number;
  weights_used: {
    yield_improvement: number;
    risk_reduction: number;
    resource_efficiency: number;
    model_reliability: number;
  };
  constraints_used: {
    min_yield?: number | null;
    max_resource_change_pct?: number | null;
    max_risk_score?: number | null;
    min_reliability_score?: number | null;
  };
  recommended_scenario: OptimizationCandidate | null;
  pareto_alternatives: OptimizationCandidate[];
  all_ranked_candidates: OptimizationCandidate[];
  total_evaluated: number;
  feasible_count: number;
  scientific_disclaimer: string;
}

export interface ScenarioAuditItem {
  scenario_id: string;
  model_version: string;
  dataset_version: string;
  created_at: string;
  location: string;
  horizon: number;
  scenario_type: string;
  modified_features: FeatureChangeItem[];
  constraints: Record<string, any>;
  baseline_prediction: number;
  scenario_prediction: number;
  yield_delta: number;
  validation_r2: number;
  validation_mae: number;
  validation_rmse: number;
  drift_status: string;
  data_quality_score: number;
  prediction_spread_disclaimer: string;
  is_reproducible: boolean;
}

export interface ScenarioHistoryResult {
  total_records: number;
  history: ScenarioAuditItem[];
}
