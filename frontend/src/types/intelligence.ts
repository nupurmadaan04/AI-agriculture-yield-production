export interface ScenarioProfile {
  predicted_yield: number
  lower_bound: number
  upper_bound: number
  spread: number
  uncertainty_pct: number
  risk_score: number
  risk_level: string
  is_anomaly: boolean
  anomaly_score: number
  features: Record<string, number>
}

export interface ChangedFeatureItem {
  feature_key: string
  feature_name: string
  baseline_value: number
  scenario_value: number
  absolute_change: number
  percent_change: number
}

export interface ScenarioSimulationDelta {
  yield_delta_kg_ha: number
  yield_percent_change: number
  risk_delta: number
  spread_delta_kg_ha: number
  direction: 'positive' | 'negative' | 'neutral'
  risk_direction: 'increased' | 'decreased' | 'unchanged'
}

export interface ScenarioSimulationRequest {
  year?: number
  state?: string
  state_code?: number
  district?: string
  baseline_rice_area?: number
  scenario_rice_area?: number
  scenario_total_cropped_area?: number
  scenario_rice_area_share?: number
  scenario_wheat_area?: number
  scenario_cotton_area?: number
  scenario_sugarcane_area?: number
  scenario_historical_yield_lag?: number
  scenario_rolling_yield?: number
}

export interface ScenarioSimulationResponse {
  state: string
  district: string
  year: number
  baseline: ScenarioProfile
  scenario: ScenarioProfile
  delta: ScenarioSimulationDelta
  changed_features: ChangedFeatureItem[]
  explanation: string
  warnings: string[]
  disclaimer: string
}

export interface CopilotQueryRequest {
  question: string
  context?: Record<string, any>
}

export interface CopilotEvidenceItem {
  source_name: string
  description: string
  records_count: number
  data_snippet?: any
}

export interface CopilotQueryResponse {
  question: string
  intent: string
  answer: string
  findings: string[]
  evidence: CopilotEvidenceItem[]
  tools_used: string[]
  records_analyzed: number
  model_outputs: any[]
  limitations: string[]
}

export interface ReportGenerateRequest {
  state?: string
  district?: string
  year?: number
  report_type?: string
}

export interface ReportGenerateResponse {
  report_id: string
  report_title: string
  report_type: string
  generated_at: string
  state?: string
  district?: string
  year?: number
  markdown_content: string
  summary_metrics: Record<string, any>
}

export interface RegionalSituationItem {
  state: string
  risk_level: string
  risk_score: number
  avg_yield: number
  volatility: number
  action_note: string
}

export interface ModelSignalItem {
  signal_name: string
  importance_pct: number
  description: string
}

export interface DecisionSupportResponse {
  kpis: {
    total_records: number
    anomalies_detected: number
    high_risk_states_count: number
    moderate_risk_states_count: number
    low_risk_states_count: number
    average_uncertainty_pct: number
    highest_risk_states: any[]
    recent_anomalies: any[]
  }
  regional_situation: RegionalSituationItem[]
  model_signals: ModelSignalItem[]
  recent_anomalies: any[]
  prediction_outlook: {
    national_mean_yield: number
    forecast_confidence: string
    active_pipeline: string
  }
  scenario_snapshot: {
    default_state: string
    simulated_intervention: string
    modeled_yield_shift: string
    risk_shift: string
  }
  ai_insight: string
  scientific_disclaimer: string
}
