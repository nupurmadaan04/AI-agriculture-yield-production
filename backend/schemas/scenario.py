"""
Pydantic Schemas for Agricultural Scenario Simulation & Decision Optimization.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ScenarioRequest(BaseModel):
    state: str = Field(..., description="Target state name or code")
    district: Optional[str] = Field(None, description="Optional target district")
    horizon: int = Field(1, ge=1, le=5, description="Forecast horizon in years (1 to 5)")
    scenario_type: str = Field('custom', description="Scenario archetype: baseline, conservative_improvement, moderate_improvement, stress_scenario, custom")
    modifications: Dict[str, float] = Field(default_factory=dict, description="Feature adjustments (percentages or values)")
    baseline_rice_area: Optional[float] = Field(None, description="Custom baseline rice area in 1000 ha")


class FeatureChangeItem(BaseModel):
    feature_key: str
    feature_name: str
    baseline_value: float
    scenario_value: float
    absolute_change: float
    percent_change: float


class ValidationContext(BaseModel):
    model_version: str
    dataset_version: str
    validation_r2: float
    validation_mae: float
    validation_rmse: float
    drift_status: str
    data_quality_score: float
    spread_type: str


class ScenarioResponse(BaseModel):
    scenario_id: str
    location: str
    state: Optional[str] = None
    district: Optional[str] = None
    horizon: int
    scenario_type: str
    scenario_name: str
    baseline_prediction: float
    scenario_prediction: float
    yield_delta: float
    yield_percent_change: float
    risk_score: float
    risk_delta: float
    warning_score: float
    warning_delta: float
    prediction_spread: float
    lower_bound_p10: float
    upper_bound_p90: float
    changed_features: List[FeatureChangeItem]
    unsupported_features_requested: List[str]
    validation_context: ValidationContext
    scientific_disclaimer: str
    # Backward compatibility fields
    baseline: Optional[Dict[str, Any]] = None
    scenario: Optional[Dict[str, Any]] = None
    delta: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None


class ScenarioComparisonItem(BaseModel):
    scenario_id: str
    scenario_type: str
    scenario_name: str
    projected_yield: float
    yield_delta: float
    yield_percent_change: float
    risk_score: float
    risk_delta: float
    warning_score: float
    warning_delta: float
    prediction_spread: float
    spread_delta: float
    is_baseline: bool
    interpretation: str


class ScenarioComparisonRequest(BaseModel):
    state: str = Field(..., description="Target state name")
    district: Optional[str] = Field(None, description="Optional target district")
    horizon: int = Field(1, ge=1, le=5, description="Forecast horizon in years")
    custom_modifications: Optional[Dict[str, float]] = Field(None, description="Optional custom scenario overrides")


class ScenarioComparisonResponse(BaseModel):
    location: str
    horizon: int
    scenarios_compared_count: int
    baseline_yield: float
    highest_yield_scenario: str
    lowest_risk_scenario: str
    yield_range_kg_ha: float
    comparison_matrix: List[ScenarioComparisonItem]
    scientific_disclaimer: str


class PerturbationStep(BaseModel):
    perturbation_pct: float
    perturbed_input_value: float
    predicted_yield: float
    yield_delta: float
    yield_percent_change: float


class FeatureSensitivityItem(BaseModel):
    feature_key: str
    feature_name: str
    baseline_value: float
    elasticity_index: float
    sensitivity_rank: int
    perturbation_responses: List[PerturbationStep]


class SensitivityRequest(BaseModel):
    state: str = Field(..., description="Target state name")
    district: Optional[str] = Field(None, description="Optional target district")
    horizon: int = Field(1, ge=1, le=5, description="Forecast horizon in years")
    features_to_test: Optional[List[str]] = Field(None, description="Subset of features to perturb")


class SensitivityResponse(BaseModel):
    location: str
    horizon: int
    baseline_prediction: float
    perturbation_steps: List[float]
    features_analyzed: int
    most_sensitive_feature: str
    sensitivity_matrix: List[FeatureSensitivityItem]
    scientific_disclaimer: str


class OptimizationWeights(BaseModel):
    yield_improvement: float = 0.40
    risk_reduction: float = 0.25
    resource_efficiency: float = 0.20
    model_reliability: float = 0.15


class OptimizationConstraints(BaseModel):
    min_yield: Optional[float] = None
    max_resource_change_pct: Optional[float] = None
    max_risk_score: Optional[float] = None
    min_reliability_score: Optional[float] = None


class OptimizationRequest(BaseModel):
    state: str = Field(..., description="Target state name")
    district: Optional[str] = Field(None, description="Optional target district")
    horizon: int = Field(1, ge=1, le=5, description="Forecast horizon in years")
    weights: Optional[OptimizationWeights] = Field(default_factory=OptimizationWeights)
    constraints: Optional[OptimizationConstraints] = Field(default_factory=OptimizationConstraints)


class OptimizationCandidateItem(BaseModel):
    scenario_id: str
    scenario_name: str
    projected_yield: float
    yield_delta: float
    yield_percent_change: float
    risk_score: float
    resource_change_pct: float
    decision_score: float
    rank: int
    is_feasible: bool
    is_pareto_optimal: bool
    constraint_status: Dict[str, Any]
    strengths: List[str]
    limitations: List[str]
    tradeoff_summary: str


class OptimizationResponse(BaseModel):
    location: str
    horizon: int
    baseline_yield: float
    baseline_risk: float
    weights_used: Dict[str, float]
    constraints_used: Dict[str, Any]
    recommended_scenario: Optional[OptimizationCandidateItem]
    pareto_alternatives: List[OptimizationCandidateItem]
    all_ranked_candidates: List[OptimizationCandidateItem]
    total_evaluated: int
    feasible_count: int
    optimization_method: str = "Weighted Linear Scalarization + Pareto Filtering"
    scientific_disclaimer: str


class ScenarioAuditItem(BaseModel):
    scenario_id: str
    model_version: str
    dataset_version: str
    created_at: str
    location: str
    horizon: int
    scenario_type: str
    modified_features: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    baseline_prediction: float
    scenario_prediction: float
    yield_delta: float
    validation_r2: float
    validation_mae: float
    validation_rmse: float
    drift_status: str
    data_quality_score: float
    prediction_spread_disclaimer: str
    is_reproducible: bool


class ScenarioHistoryResponse(BaseModel):
    total_records: int
    total_scenarios: int = 0
    history: List[ScenarioAuditItem]

