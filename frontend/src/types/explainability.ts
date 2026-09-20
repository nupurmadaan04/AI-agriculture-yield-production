/**
 * TypeScript Interfaces for Day 13: Explainable Agricultural AI & Decision Traceability
 */

export interface FeatureImportanceItem {
  feature: string;
  feature_label: string;
  native_importance: number;
  permutation_importance: number;
  native_rank: number;
  permutation_rank: number;
  rank_agreement: boolean;
}

export interface GlobalImportanceResponse {
  model_id: string;
  model_name: string;
  version: string;
  dataset_version: string;
  explanation_method: string;
  total_features_evaluated: number;
  top_feature: string;
  top_feature_label: string;
  features: FeatureImportanceItem[];
  scientific_disclaimer: string;
}

export interface FeatureContribution {
  feature: string;
  feature_label: string;
  feature_value: number;
  baseline_value: number;
  contribution_kg_ha: number;
  contribution_direction: 'POSITIVE' | 'NEGATIVE';
  relative_influence_pct: number;
}

export interface LocalExplanationRequest {
  state: string;
  district?: string;
  year?: number;
  area_1000_ha: number;
  total_cropped_area?: number;
  rice_area_share?: number;
  wheat_area?: number;
  cotton_area?: number;
  sugarcane_area?: number;
  rice_yield_lag1?: number;
  rice_yield_roll3?: number;
}

export interface LocalExplanationResponse {
  entity: string;
  year: number;
  prediction_kg_ha: number;
  baseline_reference_kg_ha: number;
  prediction_delta_kg_ha: number;
  model_version: string;
  dataset_version: string;
  explanation_method: string;
  top_positive_features: string[];
  top_negative_features: string[];
  feature_contributions: FeatureContribution[];
  scientific_disclaimer: string;
  explanation_id?: string;
}

export interface SensitivityCurvePoint {
  step_pct: number;
  perturbed_value: number;
  predicted_yield_kg_ha: number;
  prediction_delta_kg_ha: number;
  relative_delta_pct: number;
}

export interface SensitivityRequest {
  state: string;
  district?: string;
  target_features?: string[];
}

export interface SensitivityResponse {
  model_id: string;
  model_version: string;
  base_prediction_kg_ha: number;
  tested_features: string[];
  perturbation_steps_pct: number[];
  sensitivity_curves: Record<string, SensitivityCurvePoint[]>;
  scientific_disclaimer: string;
}

export interface AlertExplanationResponse {
  alert_id: string;
  location: string;
  state: string;
  district?: string;
  year: number;
  severity: string;
  composite_risk_score: number;
  temporal_diagnostics: {
    dominant_trigger?: string;
    severity_tier?: string;
    composite_risk_score?: number;
    summary?: string;
  };
  signal_breakdown: Array<{
    type: string;
    description: string;
    impact: string;
  }>;
  evidence_chain: string[];
  model_validation_context: {
    r2?: number;
    mae?: number;
    drift_status?: string;
    data_quality_score?: number;
  };
  recommended_action: string;
  model_version: string;
  dataset_version: string;
  explanation_method: string;
  scientific_disclaimer: string;
}

export interface ScenarioExplanationResponse {
  scenario_id: string;
  state: string;
  baseline_yield_kg_ha: number;
  simulated_yield_kg_ha: number;
  simulated_delta_kg_ha: number;
  simulated_delta_pct: number;
  changed_inputs: Array<{
    feature: string;
    feature_label: string;
    modified_value: number;
  }>;
  unchanged_inputs: Array<Record<string, unknown>>;
  model_attribution_summary: string;
  model_version: string;
  dataset_version: string;
  explanation_method: string;
  scientific_disclaimer: string;
}

export interface ExplanationAuditResponse {
  explanation_id: string;
  timestamp: string;
  model_name: string;
  model_version: string;
  dataset_version: string;
  entity: string;
  scenario_id?: string;
  alert_id?: string;
  prediction_kg_ha: number;
  baseline_reference_kg_ha: number;
  prediction_delta_kg_ha: number;
  explanation_method: string;
  input_features: Record<string, unknown>;
  top_positive_features: string[];
  top_negative_features: string[];
  feature_contributions: Array<Record<string, unknown>>;
  limitations: string[];
}

export interface ExplanationValidationCheck {
  rule: string;
  passed: boolean;
  details: string;
}

export interface ExplanationValidationResponse {
  is_valid: boolean;
  passed_checks: number;
  total_checks: number;
  validation_score_pct: number;
  checks: ExplanationValidationCheck[];
  scientific_note: string;
}
