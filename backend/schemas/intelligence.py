from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# =========================================================================
# PART 1: SCENARIO SIMULATION SCHEMAS
# =========================================================================

class ScenarioSimulationRequest(BaseModel):
    year: int = Field(2017, description="Agricultural simulation year")
    state: Optional[str] = Field("Punjab", description="State Name")
    state_code: Optional[int] = Field(None, description="State Code")
    district: Optional[str] = Field("Ludhiana", description="District Name")
    # Base inputs
    baseline_rice_area: Optional[float] = Field(None, description="Optional baseline override ('000 ha)")
    # Scenario modifications
    scenario_rice_area: Optional[float] = Field(None, description="Modified rice area ('000 ha)")
    scenario_total_cropped_area: Optional[float] = Field(None, description="Modified total cropped land ('000 ha)")
    scenario_rice_area_share: Optional[float] = Field(None, description="Modified rice share (0.0 - 1.0)")
    scenario_wheat_area: Optional[float] = Field(None, description="Modified wheat area ('000 ha)")
    scenario_cotton_area: Optional[float] = Field(None, description="Modified cotton area ('000 ha)")
    scenario_sugarcane_area: Optional[float] = Field(None, description="Modified sugarcane area ('000 ha)")
    scenario_historical_yield_lag: Optional[float] = Field(None, description="Modified prior-year yield (kg/ha)")
    scenario_rolling_yield: Optional[float] = Field(None, description="Modified 3-yr rolling yield (kg/ha)")

class ScenarioProfile(BaseModel):
    predicted_yield: float
    lower_bound: float
    upper_bound: float
    spread: float
    uncertainty_pct: float
    risk_score: float
    risk_level: str
    is_anomaly: bool
    anomaly_score: float
    features: Dict[str, float]

class ChangedFeatureItem(BaseModel):
    feature_key: str
    feature_name: str
    baseline_value: float
    scenario_value: float
    absolute_change: float
    percent_change: float

class ScenarioSimulationDelta(BaseModel):
    yield_delta_kg_ha: float
    yield_percent_change: float
    risk_delta: float
    spread_delta_kg_ha: float
    direction: str
    risk_direction: str

class ScenarioSimulationResponse(BaseModel):
    state: str
    district: str
    year: int
    baseline: ScenarioProfile
    scenario: ScenarioProfile
    delta: ScenarioSimulationDelta
    changed_features: List[ChangedFeatureItem]
    explanation: str
    warnings: List[str]
    disclaimer: str

# =========================================================================
# PART 2: COPILOT & QUERY SCHEMAS
# =========================================================================

class CopilotQueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, description="Natural language question")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional conversation context")

class CopilotEvidenceItem(BaseModel):
    source_name: str
    description: str
    records_count: int
    data_snippet: Optional[Any] = None

class CopilotQueryResponse(BaseModel):
    question: str
    intent: str
    answer: str
    findings: List[str]
    evidence: List[CopilotEvidenceItem]
    tools_used: List[str]
    records_analyzed: int
    model_outputs: List[Dict[str, Any]]
    limitations: List[str]

# =========================================================================
# PART 3: AI INTELLIGENCE REPORT SCHEMAS
# =========================================================================

class ReportGenerateRequest(BaseModel):
    state: Optional[str] = Field("Punjab", description="Target State")
    district: Optional[str] = Field("Ludhiana", description="Target District")
    year: Optional[int] = Field(2017, description="Target Agricultural Year")
    report_type: str = Field("comprehensive", description="regional | state | district | model | risk | anomaly | comprehensive")

class ReportGenerateResponse(BaseModel):
    report_id: str
    report_title: str
    report_type: str
    generated_at: str
    state: Optional[str] = None
    district: Optional[str] = None
    year: Optional[int] = None
    markdown_content: str
    summary_metrics: Dict[str, Any]

# =========================================================================
# PART 4: DECISION SUPPORT SCHEMAS
# =========================================================================

class RegionalSituationItem(BaseModel):
    state: str
    risk_level: str
    risk_score: float
    avg_yield: float
    volatility: float
    action_note: str

class ModelSignalItem(BaseModel):
    signal_name: str
    importance_pct: float
    description: str

class DecisionSupportResponse(BaseModel):
    kpis: Dict[str, Any]
    regional_situation: List[RegionalSituationItem]
    model_signals: List[ModelSignalItem]
    recent_anomalies: List[Dict[str, Any]]
    prediction_outlook: Dict[str, Any]
    scenario_snapshot: Dict[str, Any]
    ai_insight: str
    scientific_disclaimer: str
