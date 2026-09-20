/**
 * TypeScript definitions for Day 14 Agricultural Decision Intelligence & Evidence Reports.
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
  classification: 'FACT' | 'MODEL OUTPUT' | 'SIMULATION' | 'INTERPRETATION' | 'DERIVED' | 'VALIDATION'
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
}

export interface DecisionBrief {
  decision_id: string
  context: DecisionContext
  executive_summary: ExecutiveSummary
  evidence_status: EvidenceStatus
  sections: DecisionSection[]
  signals: DecisionSignal[]
  analytical_priorities: DecisionPriority[]
  decision_options: DecisionOption[]
  robustness: DecisionRobustness[]
  evidence_items: EvidenceItem[]
  provenance: DecisionProvenance
  audit_record: DecisionAudit
  limitations: string[]
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
