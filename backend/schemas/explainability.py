"""
Explainability Pydantic v2 Schemas for Agricultural Decision Intelligence.

Defines validated data transfer schemas for:
- Global feature importance
- Local prediction attribution
- Feature sensitivity curves
- Alert explanations
- Scenario comparison explanations
- Explanation audit certificates
- Validation integrity reports
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class FeatureImportanceItem(BaseModel):
    feature: str
    feature_label: str
    native_importance: float
    permutation_importance: float
    native_rank: int
    permutation_rank: int
    rank_agreement: bool


class GlobalImportanceResponse(BaseModel):
    model_id: str
    model_name: str
    version: str
    dataset_version: str
    explanation_method: str
    total_features_evaluated: int
    top_feature: str
    top_feature_label: str
    features: List[FeatureImportanceItem]
    scientific_disclaimer: str


class FeatureContribution(BaseModel):
    feature: str
    feature_label: str
    feature_value: float
    baseline_value: float
    contribution_kg_ha: float
    contribution_direction: str
    relative_influence_pct: float


class LocalExplanationRequest(BaseModel):
    state: str = Field(..., description="Target agricultural state")
    district: Optional[str] = Field(None, description="Optional target district")
    year: int = Field(2017, description="Observation survey year")
    area_1000_ha: float = Field(..., gt=0, description="Rice area in 1000 ha")
    total_cropped_area: Optional[float] = Field(None, description="Total cropped area")
    rice_area_share: Optional[float] = Field(None, description="Rice area share")
    wheat_area: Optional[float] = Field(None, description="Wheat area")
    cotton_area: Optional[float] = Field(None, description="Cotton area")
    sugarcane_area: Optional[float] = Field(None, description="Sugarcane area")
    rice_yield_lag1: Optional[float] = Field(None, description="Lagged yield (t-1)")
    rice_yield_roll3: Optional[float] = Field(None, description="3-year rolling baseline yield")


class LocalExplanationResponse(BaseModel):
    entity: str
    year: int
    prediction_kg_ha: float
    baseline_reference_kg_ha: float
    prediction_delta_kg_ha: float
    model_version: str
    dataset_version: str
    explanation_method: str
    top_positive_features: List[str]
    top_negative_features: List[str]
    feature_contributions: List[FeatureContribution]
    scientific_disclaimer: str


class SensitivityCurvePoint(BaseModel):
    step_pct: float
    perturbed_value: float
    predicted_yield_kg_ha: float
    prediction_delta_kg_ha: float
    relative_delta_pct: float


class ExplainabilitySensitivityRequest(BaseModel):
    state: str = Field(..., description="Target agricultural state")
    district: Optional[str] = Field(None, description="Optional district")
    target_features: Optional[List[str]] = Field(None, description="Features to perturb")


class ExplainabilitySensitivityResponse(BaseModel):
    model_id: str
    model_version: str
    base_prediction_kg_ha: float
    tested_features: List[str]
    perturbation_steps_pct: List[float]
    sensitivity_curves: Dict[str, List[SensitivityCurvePoint]]
    scientific_disclaimer: str


SensitivityRequest = ExplainabilitySensitivityRequest
SensitivityResponse = ExplainabilitySensitivityResponse


class AlertExplanationResponse(BaseModel):
    alert_id: str
    location: str
    state: str
    district: Optional[str] = None
    year: int
    severity: str
    composite_risk_score: float
    temporal_diagnostics: Dict[str, Any]
    signal_breakdown: List[Dict[str, Any]]
    evidence_chain: List[str]
    model_validation_context: Dict[str, Any]
    recommended_action: str
    model_version: str
    dataset_version: str
    explanation_method: str
    scientific_disclaimer: str


class ScenarioExplanationResponse(BaseModel):
    scenario_id: str
    state: str
    baseline_yield_kg_ha: float
    simulated_yield_kg_ha: float
    simulated_delta_kg_ha: float
    simulated_delta_pct: float
    changed_inputs: List[Dict[str, Any]]
    unchanged_inputs: List[Dict[str, Any]]
    model_attribution_summary: str
    model_version: str
    dataset_version: str
    explanation_method: str
    scientific_disclaimer: str


class ExplanationAuditResponse(BaseModel):
    explanation_id: str
    timestamp: str
    model_name: str
    model_version: str
    dataset_version: str
    entity: str
    scenario_id: Optional[str] = None
    alert_id: Optional[str] = None
    prediction_kg_ha: float
    baseline_reference_kg_ha: float
    prediction_delta_kg_ha: float
    explanation_method: str
    input_features: Dict[str, Any]
    top_positive_features: List[str]
    top_negative_features: List[str]
    feature_contributions: List[Dict[str, Any]]
    limitations: List[str]


class ExplanationValidationCheck(BaseModel):
    rule: str
    passed: bool
    details: str


class ExplanationValidationResponse(BaseModel):
    is_valid: bool
    passed_checks: int
    total_checks: int
    validation_score_pct: float
    checks: List[ExplanationValidationCheck]
    scientific_note: str
