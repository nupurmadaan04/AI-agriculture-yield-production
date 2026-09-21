"""
Pydantic Schemas for Agricultural Decision Intelligence & Evidence-Based Forecast Briefs (Day 31).

Enforces strict semantic classifications:
OBSERVED | PREDICTED | DERIVED | HISTORICAL_REFERENCE | MODEL_ATTRIBUTION |
VALIDATION | MONITORING | PROVENANCE | DECISION_EVIDENCE | ASSUMPTION | LIMITATION
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class DecisionContext(BaseModel):
    crop: str = Field("Rice", description="Target crop")
    state: str = Field("Punjab", description="Target agricultural state")
    district: Optional[str] = Field(None, description="Optional target district")
    year: int = Field(2017, description="Agricultural year")
    decision_horizon: str = Field("next_season", description="Planning horizon")
    target_area_1000_ha: Optional[float] = Field(None, description="Cultivated area in 1000 ha")


class DecisionForecastSummary(BaseModel):
    crop: str
    state: str
    district: Optional[str] = None
    forecast_year: int
    forecast_yield_kg_ha: float
    unit: str = "kg/ha"
    strategy: str = "Historical District Mean / Persistence"
    model_name: str = "Baseline (Historical Average)"
    model_version: str = "1.0.0"
    certification_status: str = "BASELINE_PRODUCTION"
    is_deterministic: bool = True
    fallback_used: bool = False
    request_id: str
    provenance_hash: str
    timestamp: str


class HistoricalObservationPoint(BaseModel):
    year: int
    observed_yield_kg_ha: float
    observed_area_ha: Optional[float] = None
    observed_production_tonnes: Optional[float] = None
    source: str = "AGRI_PANEL_1.0 (ICRISAT/DES)"
    semantic_type: str = "OBSERVED"


class HistoricalContext(BaseModel):
    crop: str
    state: str
    district: Optional[str] = None
    start_year: int
    end_year: int
    sample_count: int
    historical_mean_yield_kg_ha: float
    historical_median_yield_kg_ha: float
    historical_min_yield_kg_ha: float
    historical_max_yield_kg_ha: float
    historical_std_yield_kg_ha: float
    trend_slope_kg_ha_yr: float
    recent_observations: List[HistoricalObservationPoint] = []
    source: str = "Datasets/processed/agricultural_panel.csv"
    semantic_classification: str = "HISTORICAL_REFERENCE"


class ValidationEvidence(BaseModel):
    strategy_tier: str = "BASELINE_PRODUCTION"
    primary_strategy: str = "Historical District Mean / Persistence"
    validation_protocol: str = "4-Fold Expanding Walk-Forward Validation"
    validation_period: str = "2014-2017"
    mae_kg_ha: float
    rmse_kg_ha: Optional[float] = None
    r2_score: Optional[float] = None
    fold_win_rate_pct: float
    mean_improvement_pct: float
    baseline_mae_kg_ha: float
    baseline_strategy: str = "Historical District Mean / Persistence"
    is_ml_certified: bool
    legacy_benchmark_note: Optional[str] = None
    source: str = "Datasets/metadata/certified_strategies.json"
    semantic_classification: str = "VALIDATION"


class UncertaintyEvidence(BaseModel):
    is_available: bool
    predicted_yield_kg_ha: Optional[float] = None
    empirical_p10_kg_ha: Optional[float] = None
    empirical_p90_kg_ha: Optional[float] = None
    ensemble_spread_kg_ha: Optional[float] = None
    spread_percentage: Optional[float] = None
    methodology: str = "Empirical P10-P90 ensemble spread across walk-forward estimator predictions"
    disclaimer: str = "This range represents empirical ensemble spread and is not a formal confidence interval."
    semantic_classification: str = "DERIVED"


class MonitoringEvidence(BaseModel):
    operational_records_count: int
    monitoring_status: str
    prediction_drift_psi: Optional[float] = None
    feature_drift_summary: Optional[str] = None
    post_outcome_evaluation_status: str = "EVALUATION_UNAVAILABLE"
    observed_harvest_yield_kg_ha: Optional[float] = None
    signed_bias_kg_ha: Optional[float] = None
    active_alerts_count: int = 0
    alerts_summary: List[str] = []
    source: str = "ForecastMonitoringService (Day 30)"
    semantic_classification: str = "MONITORING"


class AttributionItem(BaseModel):
    feature_name: str
    feature_label: str
    importance_or_shap: float
    attribution_type: str = "TREE_SHAP"
    semantic_classification: str = "MODEL_ATTRIBUTION"
    interpretation: str


class EvidenceItem(BaseModel):
    evidence_id: str
    category: str
    statement: str
    value: Any
    unit: str
    source_module: str
    source_method: str
    evidence_type: str = Field(description="OBSERVED | PREDICTED | SIMULATED | DERIVED | MODEL_ATTRIBUTION | VALIDATION | MONITORING | PROVENANCE | DECISION_EVIDENCE")
    confidence_status: str = "VALIDATED"
    timestamp: str
    model_version: str = "1.0.0"
    dataset_version: str = "AGRI_PANEL_1.0"
    period: Optional[str] = None
    population: Optional[str] = None
    interpretation: Optional[str] = None
    limitation: Optional[str] = None


class DecisionSignal(BaseModel):
    signal_name: str
    signal_label: str
    strength: str = Field(description="HIGH | MODERATE | LOW | NEGLIGIBLE")
    evidence_count: int
    severity: str
    persistence: str
    supporting_evidence: List[str]
    interpretation: str
    semantic_classification: str = "DECISION_EVIDENCE"


class DecisionPriority(BaseModel):
    priority_rank: int
    issue: str
    priority_level: str = Field(description="HIGH | MODERATE | LOW")
    reasoning: List[str]
    supporting_evidence: List[str]


class DecisionOption(BaseModel):
    option_id: str
    scenario_id: str
    title: str
    scenario_type: str
    projected_yield_kg_ha: float
    projected_yield_delta_kg_ha: float
    projected_production_delta_pct: float
    risk_change: str
    resource_efficiency: str
    model_reliability: str
    tradeoffs: str
    limitations: str
    supporting_evidence: List[str]
    is_simulated: bool = True
    semantic_classification: str = "DERIVED"


class DecisionRobustness(BaseModel):
    option_id: str
    title: str
    classification: str = Field(description="ROBUST | MODERATELY ROBUST | SENSITIVE | UNSUPPORTED")
    max_tested_deviation_kg_ha: float
    perturbation_range: str
    robustness_notes: str
    is_favorable: bool


class DecisionProvenanceNode(BaseModel):
    id: str
    type: str
    label: str
    metadata: Dict[str, Any]


class DecisionProvenanceEdge(BaseModel):
    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    relation: str


class DecisionProvenance(BaseModel):
    dataset_version: str
    methodology_version: str
    total_nodes: int
    total_edges: int
    nodes: List[DecisionProvenanceNode]
    edges: List[DecisionProvenanceEdge]
    context: Dict[str, Any]
    provenance_hash: Optional[str] = None


class DecisionAudit(BaseModel):
    decision_id: str
    certificate: str
    context: Dict[str, Any]
    dataset_version: str
    model_version: str
    evidence_count: int
    evidence_ids: List[str]
    scenario_count: int
    scenario_ids: List[str]
    explanation_count: int
    explanation_ids: List[str]
    brief_summary: Dict[str, Any]
    limitations: List[str]
    methodology_version: str
    generated_at: str
    audit_disclaimer: str


class DecisionSection(BaseModel):
    section_number: int
    title: str
    classification: str = Field(description="FACT | MODEL OUTPUT | SIMULATION | INTERPRETATION | DERIVED | VALIDATION | MONITORING")
    content: str


class ExecutiveSummary(BaseModel):
    current_status: str
    outlook: str
    major_risk_signal: str
    strongest_evidence: str
    highest_priority_issue: str
    preferred_option: str
    alternative_option: str
    reliability_note: str
    limitation_note: str


class EvidenceStatus(BaseModel):
    evidence_agreement: str
    model_reliability: str
    data_quality_score: str
    prediction_spread: str
    signal_persistence: str
    completeness_level: str = "PARTIAL_EVIDENCE"


class DecisionBrief(BaseModel):
    decision_id: str
    context: DecisionContext
    forecast_summary: Optional[DecisionForecastSummary] = None
    executive_summary: ExecutiveSummary
    evidence_status: EvidenceStatus
    historical_context: Optional[HistoricalContext] = None
    validation_evidence: Optional[ValidationEvidence] = None
    uncertainty_evidence: Optional[UncertaintyEvidence] = None
    monitoring_evidence: Optional[MonitoringEvidence] = None
    attribution_evidence: List[AttributionItem] = []
    sections: List[DecisionSection]
    signals: List[DecisionSignal]
    analytical_priorities: List[DecisionPriority]
    decision_options: List[DecisionOption]
    robustness: List[DecisionRobustness]
    evidence_items: List[EvidenceItem]
    assumptions: List[str] = []
    limitations: List[str] = []
    provenance: DecisionProvenance
    audit_record: DecisionAudit
    generated_at: str
    footer_disclaimer: str


class DecisionAnalyzeRequest(BaseModel):
    crop: Optional[str] = Field("Rice", description="Target crop name")
    state: str = Field("Punjab", description="State name")
    district: Optional[str] = Field(None, description="District name")
    year: Optional[int] = Field(2017, description="Target agricultural year")
    decision_horizon: Optional[str] = Field("next_season", description="Decision horizon")


class DecisionAnalyzeResponse(BaseModel):
    decision_id: str
    context: DecisionContext
    brief: DecisionBrief
    is_scientifically_validated: bool
    validation_checks_passed: int
    validation_total_rules: int


class DecisionOptionsResponse(BaseModel):
    decision_id: str
    options: List[DecisionOption]
    robustness: List[DecisionRobustness]


class DecisionRobustnessResponse(BaseModel):
    decision_id: str
    robustness: List[DecisionRobustness]


class DecisionHistoryResponse(BaseModel):
    total_records: int
    records: List[DecisionAudit]
