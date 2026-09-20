"""
Pydantic Schemas for Agricultural Decision Intelligence & Automated Evidence Reports (Day 14).
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


class EvidenceItem(BaseModel):
    evidence_id: str
    category: str
    statement: str
    value: Any
    unit: str
    source_module: str
    source_method: str
    evidence_type: str = Field(description="OBSERVED | PREDICTED | SIMULATED | DERIVED | MODEL_ATTRIBUTION | VALIDATION")
    confidence_status: str
    timestamp: str
    model_version: str
    dataset_version: str


class DecisionSignal(BaseModel):
    signal_name: str
    signal_label: str
    strength: str = Field(description="HIGH | MODERATE | LOW | NEGLIGIBLE")
    evidence_count: int
    severity: str
    persistence: str
    supporting_evidence: List[str]
    interpretation: str


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
    classification: str = Field(description="FACT | MODEL OUTPUT | SIMULATION | INTERPRETATION | DERIVED | VALIDATION")
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


class DecisionBrief(BaseModel):
    decision_id: str
    context: DecisionContext
    executive_summary: ExecutiveSummary
    evidence_status: EvidenceStatus
    sections: List[DecisionSection]
    signals: List[DecisionSignal]
    analytical_priorities: List[DecisionPriority]
    decision_options: List[DecisionOption]
    robustness: List[DecisionRobustness]
    evidence_items: List[EvidenceItem]
    provenance: DecisionProvenance
    audit_record: DecisionAudit
    limitations: List[str]
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
