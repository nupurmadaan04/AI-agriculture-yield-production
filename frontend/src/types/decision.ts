/**
 * TypeScript definitions for Day 31 Agricultural Decision Intelligence & Evidence Reports.
 */

export interface DecisionContext {
  crop: string
  state: string
  district?: string | null
  year: number
  decision_horizon: string
  target_area_1000_ha?: number | null
}

export type EvidenceType =
  | 'OBSERVED'
  | 'PREDICTED'
  | 'SIMULATED'
  | 'DERIVED'
  | 'MODEL_ATTRIBUTION'
  | 'VALIDATION'
  | 'MONITORING'
  | 'PROVENANCE'
  | 'DECISION_EVIDENCE'
  | 'ASSUMPTION'
  | 'LIMITATION'

export interface DecisionForecastSummary {
  crop: string
  state: string
  district?: string | null
  forecast_year: number
  forecast_yield_kg_ha: number
  unit: string
  strategy: string
  model_name: string
  model_version: string
  certification_status: string
  is_deterministic: boolean
  fallback_used: boolean
  request_id: string
  provenance_hash: string
  timestamp: string
}

export interface HistoricalObservationPoint {
  year: number
  observed_yield_kg_ha: number
  observed_area_ha?: number | null
  observed_production_tonnes?: number | null
  source: string
  semantic_type: string
}

export interface HistoricalContext {
  crop: string
  state: string
  district?: string | null
  start_year: number
  end_year: number
  sample_count: number
  historical_mean_yield_kg_ha: number
  historical_median_yield_kg_ha: number
  historical_min_yield_kg_ha: number
  historical_max_yield_kg_ha: number
  historical_std_yield_kg_ha: number
  trend_slope_kg_ha_yr: number
  recent_observations: HistoricalObservationPoint[]
  source: string
  semantic_classification: string
}

export interface ValidationEvidence {
  strategy_tier: string
  primary_strategy: string
  validation_protocol: string
  validation_period: string
  mae_kg_ha: number
  rmse_kg_ha?: number | null
  r2_score?: number | null
  fold_win_rate_pct: number
  mean_improvement_pct: number
  baseline_mae_kg_ha: number
  baseline_strategy: string
  is_ml_certified: boolean
  legacy_benchmark_note?: string | null
  source: string
  semantic_classification: string
}

export interface UncertaintyEvidence {
  is_available: boolean
  predicted_yield_kg_ha?: number | null
  empirical_p10_kg_ha?: number | null
  empirical_p90_kg_ha?: number | null
  ensemble_spread_kg_ha?: number | null
  spread_percentage?: number | null
  methodology: string
  disclaimer: string
  semantic_classification: string
}

export interface MonitoringEvidence {
  operational_records_count: number
  monitoring_status: string
  prediction_drift_psi?: number | null
  feature_drift_summary?: string | null
  post_outcome_evaluation_status: string
  observed_harvest_yield_kg_ha?: number | null
  signed_bias_kg_ha?: number | null
  active_alerts_count: number
  alerts_summary: string[]
  source: string
  semantic_classification: string
}

export interface AttributionItem {
  feature_name: string
  feature_label: string
  importance_or_shap: number
  attribution_type: string
  semantic_classification: string
  interpretation: string
}

export interface EvidenceItem {
  evidence_id: string
  category: string
  statement: string
  value: string | number
  unit: string
  source_module: string
  source_method: string
  evidence_type: EvidenceType
  confidence_status: string
  timestamp: string
  model_version: string
  dataset_version: string
  period?: string | null
  population?: string | null
  interpretation?: string | null
  limitation?: string | null
}

export interface DecisionSignal {
  signal_name: string
  signal_label: string
  strength: 'HIGH' | 'MODERATE' | 'LOW' | 'NEGLIGIBLE'
  evidence_count: number
  severity: string
  persistence: string
  supporting_evidence: string[]
  interpretation: string
  semantic_classification?: string
}

export interface DecisionPriority {
  priority_rank: number
  issue: string
  priority_level: 'HIGH' | 'MODERATE' | 'LOW'
  reasoning: string[]
  supporting_evidence: string[]
}

export interface DecisionOption {
  option_id: string
  scenario_id: string
  title: string
  scenario_type: string
  projected_yield_kg_ha: number
  projected_yield_delta_kg_ha: number
  projected_production_delta_pct: number
  risk_change: string
  resource_efficiency: string
  model_reliability: string
  tradeoffs: string
  limitations: string
  supporting_evidence: string[]
  is_simulated?: boolean
  semantic_classification?: string
}

export interface DecisionRobustness {
  option_id: string
  title: string
  classification: 'ROBUST' | 'MODERATELY ROBUST' | 'SENSITIVE' | 'UNSUPPORTED'
  max_tested_deviation_kg_ha: number
  perturbation_range: string
  robustness_notes: string
  is_favorable: boolean
}

export interface DecisionProvenanceNode {
  id: string
  type: string
  label: string
  metadata: Record<string, any>
}

export interface DecisionProvenanceEdge {
  from: string
  to: string
  relation: string
}

export interface DecisionProvenance {
  dataset_version: string
  methodology_version: string
  total_nodes: number
  total_edges: number
  nodes: DecisionProvenanceNode[]
  edges: DecisionProvenanceEdge[]
  context: Record<string, any>
  provenance_hash?: string | null
}

export interface DecisionAudit {
  decision_id: string
  certificate: string
  context: Record<string, any>
  dataset_version: string
  model_version: string
  evidence_count: number
  evidence_ids: string[]
  scenario_count: number
  scenario_ids: string[]
  explanation_count: number
  explanation_ids: string[]
  brief_summary: Record<string, any>
  limitations: string[]
  methodology_version: string
  generated_at: string
  audit_disclaimer: string
}

export interface DecisionSection {
  section_number: number
  title: string
  classification: string
  content: string
}

export interface ExecutiveSummary {
  current_status: string
  outlook: string
  major_risk_signal: string
  strongest_evidence: string
  highest_priority_issue: string
  preferred_option: string
  alternative_option: string
  reliability_note: string
  limitation_note: string
}

export interface EvidenceStatus {
  evidence_agreement: string
  model_reliability: string
  data_quality_score: string
  prediction_spread: string
  signal_persistence: string
  completeness_level?: string
}

export interface DecisionBrief {
  decision_id: string
  context: DecisionContext
  forecast_summary?: DecisionForecastSummary | null
  executive_summary: ExecutiveSummary
  evidence_status: EvidenceStatus
  historical_context?: HistoricalContext | null
  validation_evidence?: ValidationEvidence | null
  uncertainty_evidence?: UncertaintyEvidence | null
  monitoring_evidence?: MonitoringEvidence | null
  attribution_evidence?: AttributionItem[]
  sections: DecisionSection[]
  signals: DecisionSignal[]
  analytical_priorities: DecisionPriority[]
  decision_options: DecisionOption[]
  robustness: DecisionRobustness[]
  evidence_items: EvidenceItem[]
  assumptions?: string[]
  limitations: string[]
  provenance: DecisionProvenance
  audit_record: DecisionAudit
  generated_at: string
  footer_disclaimer: string
}

export interface DecisionAnalyzeRequest {
  crop?: string
  state: string
  district?: string | null
  year?: number
  decision_horizon?: string
}

export interface DecisionAnalyzeResponse {
  decision_id: string
  context: DecisionContext
  brief: DecisionBrief
  is_scientifically_validated: boolean
  validation_checks_passed: number
  validation_total_rules: number
}

export interface DecisionOptionsResponse {
  decision_id: string
  options: DecisionOption[]
  robustness: DecisionRobustness[]
}

export interface DecisionRobustnessResponse {
  decision_id: string
  robustness: DecisionRobustness[]
}

export interface DecisionHistoryResponse {
  total_records: number
  records: DecisionAudit[]
}
