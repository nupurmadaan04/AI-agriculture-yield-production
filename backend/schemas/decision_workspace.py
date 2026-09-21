"""
Day 32: Decision Workspace & Scenario Comparison Schemas.

Strictly preserves scientific taxonomy:
OBSERVED | PREDICTED | SCENARIO | DERIVED | MODEL_ATTRIBUTION | VALIDATION | MONITORING | PROVENANCE
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class WorkspaceAnalyzeRequest(BaseModel):
    crop: str = Field("Oilseeds", description="Crop commodity name")
    state: str = Field("Punjab", description="State name")
    district: Optional[str] = Field("Ludhiana", description="District name")
    forecast_year: int = Field(2017, description="Forecast target year")
    selected_scenarios: Optional[List[str]] = Field(
        default=["conservative_improvement", "moderate_improvement", "stress_scenario"],
        description="Scenario archetype IDs to simulate"
    )
    custom_modifications: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom percentage deltas (e.g. {'rice_area_pct': 5.0})"
    )


class BaselineForecastSummary(BaseModel):
    forecast_yield_kg_ha: float
    unit: str = "kg/ha"
    strategy: str
    model_name: str
    model_version: str = "1.0.0"
    dataset_version: str = "AGRI_PANEL_1.0 (ICRISAT 1966-2017)"
    certification_status: str
    is_deterministic: bool = True
    fallback_used: bool = False
    request_id: str
    provenance_hash: str
    timestamp: str
    semantic_classification: str = "PREDICTED"


class HistoricalObservationPoint(BaseModel):
    year: int
    observed_yield_kg_ha: float
    source: str = "AGRI_PANEL_1.0 (ICRISAT/DES)"
    semantic_classification: str = "OBSERVED"


class HistoricalReferenceContext(BaseModel):
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
    historical_period: str
    recent_observations: List[HistoricalObservationPoint] = []
    semantic_classification: str = "HISTORICAL_REFERENCE"


class ValidationContext(BaseModel):
    strategy_tier: str
    primary_strategy: str
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
    metric_definitions: Dict[str, str] = {
        "MAE": "Mean Absolute Error over out-of-time walk-forward evaluation folds.",
        "Win Rate": "Percentage of validation folds outperforming historical district baseline.",
        "Gain vs Baseline": "Relative percentage error reduction compared to persistence mean."
    }
    semantic_classification: str = "VALIDATION"


class UncertaintyContext(BaseModel):
    is_available: bool
    empirical_p10_kg_ha: Optional[float] = None
    empirical_p90_kg_ha: Optional[float] = None
    ensemble_spread_kg_ha: Optional[float] = None
    spread_percentage: Optional[float] = None
    methodology: str = "Empirical P10-P90 ensemble spread across walk-forward estimator predictions"
    coverage_wording: str = "Represents tree ensemble variance across trained estimators"
    disclaimer: str = "This range represents empirical ensemble spread and is not a formal distribution-free confidence interval."
    limitations: str = "Unavailable for deterministic baseline strategies."
    semantic_classification: str = "DERIVED"


class MonitoringContext(BaseModel):
    drift_status: str
    monitoring_status: str
    overall_psi: float
    outcome_evaluation_status: str = Field(description="EVALUATION_AVAILABLE | EVALUATION_UNAVAILABLE")
    observed_outcome_kg_ha: Optional[float] = None
    forecast_error_kg_ha: Optional[float] = None
    signed_bias_kg_ha: Optional[float] = None
    active_alerts: List[str] = []
    semantic_classification: str = "MONITORING"


class AttributionFeature(BaseModel):
    feature_name: str
    feature_label: str
    importance_or_shap: float
    interpretation: str


class AttributionContext(BaseModel):
    is_available: bool
    attribution_type: str = Field(description="TREE_SHAP | PERSISTENCE_BASELINE")
    top_features: List[AttributionFeature] = []
    methodology: str
    semantic_classification: str = "MODEL_ATTRIBUTION"


class ProvenanceContext(BaseModel):
    prediction_fingerprint: str
    dataset_identifier: str = "AGRI_PANEL_1.0 (ICRISAT 1966-2017)"
    model_identifier: str
    strategy_identifier: str
    request_id: str
    audit_reference: str
    semantic_classification: str = "PROVENANCE"


class ScenarioItem(BaseModel):
    scenario_id: str
    scenario_name: str
    scenario_type: str
    scenario_assumption: str
    scenario_output_kg_ha: float
    baseline_output_kg_ha: float
    yield_delta_kg_ha: float
    yield_percent_change: float
    uncertainty_note: str = "Uncertainty not available for this scenario."
    empirical_p10_kg_ha: Optional[float] = None
    empirical_p90_kg_ha: Optional[float] = None
    evidence_type: str = "SCENARIO"
    status: str = Field(description="SUPPORTED | NOT_SUPPORTED | EVIDENCE_UNAVAILABLE")
    limitations: str
    changed_features: Dict[str, Any] = {}
    is_simulated: bool = True


class ScenarioComparisonRow(BaseModel):
    metric_label: str
    baseline_value: str
    scenario_values: Dict[str, str] = {}


class ScenarioComparisonMatrix(BaseModel):
    scenario_headers: List[str]
    rows: List[ScenarioComparisonRow]
    disclaimer: str = (
        "Scenario outputs represent hypothetical model-based estimates derived from empirical relationships. "
        "They must not be interpreted as causal conclusions, biological certainties, or prescriptive advice."
    )


class DecisionWorkspaceResponse(BaseModel):
    workspace_id: str
    crop: str
    state: str
    district: Optional[str] = None
    forecast_year: int
    generated_at: str
    baseline_forecast: BaselineForecastSummary
    historical_context: HistoricalReferenceContext
    validation: ValidationContext
    uncertainty: UncertaintyContext
    monitoring: MonitoringContext
    attribution: AttributionContext
    provenance: ProvenanceContext
    scenarios: List[ScenarioItem]
    comparison_matrix: ScenarioComparisonMatrix
    limitations: List[str]
    decision_support_statement: str = (
        "The Decision Workspace is an evidence-based decision-support interface. "
        "It does not autonomously select an action, rank choices, or prescribe agricultural operations."
    )
