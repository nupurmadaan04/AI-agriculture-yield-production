from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class HealthResponse(BaseModel):
    status: str = "ok"
    dataset_loaded: bool
    records: int

class SummaryResponse(BaseModel):
    total_records: int
    total_states: int
    total_districts: int
    min_year: int
    max_year: int
    average_yield: float
    median_yield: float
    average_area: float
    average_production: float
    total_area: float
    total_production: float
    zero_yield_records: int
    missing_values: int
    duplicate_rows: int
    columns_count: int

class FiltersResponse(BaseModel):
    years: List[int]
    states: List[str]
    districts: List[str]
    crops: List[str]

class RecordItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    state: str
    district: str
    year: int
    area: float = Field(description="Cultivated area in 1000 ha")
    production: float = Field(description="Production in 1000 tons")
    yield_val: float = Field(description="Reported yield in kg/ha", alias="yield")

class PaginationMetadata(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int

class PaginatedRecordsResponse(BaseModel):
    data: List[RecordItem]
    pagination: PaginationMetadata

class TrendPoint(BaseModel):
    year: int
    average_yield: float
    average_area: float
    average_production: float
    total_production: float
    total_area: float
    record_count: int

class TrendsResponse(BaseModel):
    data: List[TrendPoint]

class StateSummary(BaseModel):
    state: str
    rank: int
    record_count: int
    district_count: int
    average_yield: float
    median_yield: float
    total_area: float
    total_production: float

class StatesResponse(BaseModel):
    data: List[StateSummary]

class DistrictSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    district: str
    state: str
    year: int
    area: float
    production: float
    yield_val: float = Field(alias="yield")

class DistrictsResponse(BaseModel):
    data: List[DistrictSummary]

class ModelMetricItem(BaseModel):
    id: str
    model_name: str
    model_type: str
    feature_set: str
    train_r2: Optional[float] = None
    random_r2: Optional[float] = None
    random_mae: Optional[float] = None
    random_rmse: Optional[float] = None
    temporal_r2: Optional[float] = None
    temporal_mae: Optional[float] = None
    temporal_rmse: Optional[float] = None
    cv_r2: Optional[float] = None
    cv_mae: Optional[float] = None
    cv_rmse: Optional[float] = None
    status: str
    notes: Optional[str] = None

class FeatureImportanceResponse(BaseModel):
    native_mdi: Dict[str, float]
    permutation_importance: Dict[str, Any]

class AblationItem(BaseModel):
    config_name: str
    feature_set: str
    num_features: int
    random_r2: float
    random_mae: float
    temporal_r2: float
    temporal_mae: float
    group_kfold_r2: float
    interpretation: str

class ModelMetricsResponse(BaseModel):
    leaderboard: List[ModelMetricItem]
    feature_importance: Optional[FeatureImportanceResponse] = None
    ablation_experiments: List[AblationItem] = []

class DeterministicEstimateRequest(BaseModel):
    area: float = Field(gt=0, description="Cultivated area in 1000 ha (must be > 0)")
    production: float = Field(ge=0, description="Harvest production in 1000 tons")

class DeterministicEstimateResponse(BaseModel):
    estimated_yield: float
    method: str = "Production / Area × 1000"
    type: str = "deterministic"
    unit: str = "kg/ha"
    area: float
    production: float

# =========================================================================
# DAY 3 & DAY 4 ML PREDICTION & ERROR ANALYSIS SCHEMAS
# =========================================================================

class PostHarvestPredictRequest(BaseModel):
    year: int = Field(..., ge=1990, le=2035, description="Harvest year (e.g. 2017)")
    state: Optional[str] = Field(None, description="State Name (e.g. 'Punjab')")
    state_code: Optional[int] = Field(None, ge=1, le=20, description="State Code (1-20)")
    area: float = Field(..., gt=0, description="Cultivated Area in '000 ha (strictly > 0)")
    production: float = Field(..., ge=0, description="Harvest Production in '000 tons")
    district: Optional[str] = Field(None, description="Optional District Name for historical matching")

class PostHarvestPredictResponse(BaseModel):
    mode: str = "post-harvest"
    model_name: str
    predicted_yield: float
    deterministic_yield: float
    actual_yield: Optional[float] = None
    ml_error: Optional[float] = None
    deterministic_error: Optional[float] = None
    difference: float
    historical_matched: bool
    matched_district: Optional[str] = None
    state: str
    state_code: int
    year: int
    area: float
    production: float
    formula: str = "Yield = (Production / Area) * 1000"
    feature_dependency_warning: str

class PreSeasonPredictRequest(BaseModel):
    year: int = Field(..., ge=1990, le=2035, description="Agricultural Year (e.g. 2017)")
    state: Optional[str] = Field(None, description="State Name (e.g. 'Punjab')")
    state_code: Optional[int] = Field(None, ge=1, le=20, description="State Code (1-20)")
    area: float = Field(..., gt=0, description="Cultivated Area in '000 ha (strictly > 0)")
    production: Optional[float] = Field(None, description="Must NOT be provided in pre-season mode")
    district: Optional[str] = Field(None, description="Optional District Name for historical baseline match")

class ValidationMetrics(BaseModel):
    random_r2: float
    random_mae: float
    random_rmse: float
    temporal_r2: float
    temporal_mae: float
    group_kfold_r2: float
    group_kfold_mae: Optional[float] = None
    temporal_r2_improvement_vs_baseline: Optional[str] = None
    temporal_mae_reduction_vs_baseline: Optional[str] = None
    generalization_assessment: str

class PreSeasonPredictResponse(BaseModel):
    mode: str = "pre-season"
    model_name: str
    predicted_yield: float
    actual_yield: Optional[float] = None
    ml_error: Optional[float] = None
    historical_matched: bool
    matched_district: Optional[str] = None
    state: str
    state_code: int
    year: int
    area: float
    validation_metrics: ValidationMetrics
    warning: str

class PreSeasonAdvancedPredictRequest(BaseModel):
    year: int = Field(..., ge=1990, le=2035, description="Agricultural Year")
    state: Optional[str] = Field(None, description="State Name")
    state_code: Optional[int] = Field(None, ge=1, le=20, description="State Code")
    area: float = Field(..., gt=0, description="Cultivated Rice Area ('000 ha)")
    production: Optional[float] = Field(None, description="MUST NOT be provided (pre-season)")
    district: Optional[str] = Field(None, description="District Name")
    total_cropped_area: Optional[float] = Field(None, description="Optional total cultivated land ('000 ha)")
    rice_area_share: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional rice land share (0.0 - 1.0)")
    wheat_area: Optional[float] = Field(None, ge=0.0, description="Optional wheat land area ('000 ha)")
    cotton_area: Optional[float] = Field(None, ge=0.0, description="Optional cotton land area ('000 ha)")
    sugarcane_area: Optional[float] = Field(None, ge=0.0, description="Optional sugarcane land area ('000 ha)")
    rice_yield_lag1: Optional[float] = Field(None, ge=0.0, description="Optional previous year yield (t-1)")
    rice_yield_roll3: Optional[float] = Field(None, ge=0.0, description="Optional 3-yr rolling yield average")

class UncertaintyEstimate(BaseModel):
    predicted_yield: float
    lower_bound_10th_pct: float
    upper_bound_90th_pct: float
    prediction_spread: float
    methodology: str

class FeatureContributionItem(BaseModel):
    feature: str
    contribution_score: float
    value: str

class PreSeasonAdvancedPredictResponse(BaseModel):
    mode: str = "pre-season-advanced"
    model_name: str
    predicted_yield: float
    actual_yield: Optional[float] = None
    ml_error: Optional[float] = None
    historical_matched: bool
    matched_district: Optional[str] = None
    state: str
    state_code: int
    year: int
    area: float
    features_used: Dict[str, float]
    uncertainty: UncertaintyEstimate
    feature_contributions: List[FeatureContributionItem]
    validation_metrics: ValidationMetrics
    warning: str

class ModelMetadataItem(BaseModel):
    id: str
    name: str
    model_type: str
    feature_set: str
    features: List[str]
    random_r2: float
    temporal_r2: float
    group_kfold_r2: float
    mae: float
    rmse: float
    mode_compatibility: str
    badge: str
    recommended_for: str
    is_post_harvest_only: bool

class ModelsListResponse(BaseModel):
    models: List[ModelMetadataItem]

class StateErrorItem(BaseModel):
    state: str
    count: int
    mae: float
    rmse: float
    mape: float

class TopErrorItem(BaseModel):
    state: str
    district: str
    year: int
    area: float
    production: float
    actual: float
    predicted: float
    absolute_error: float
    percentage_error: float
    root_cause: str

class YearErrorItem(BaseModel):
    year: int
    actual_avg: float
    predicted_avg: float
    mae: float
    rmse: float
    mape: float

class ErrorAnalysisResponse(BaseModel):
    worst_performing_states: List[StateErrorItem]
    top_extreme_errors: List[TopErrorItem]
    yearly_error_stability: List[YearErrorItem]

# =========================================================================
# DAY 5 AGRICULTURAL RISK, EXPLAINABILITY & ANOMALY SCHEMAS
# =========================================================================

class RiskAssessmentRequest(BaseModel):
    year: Optional[int] = Field(2017, description="Agricultural Year")
    state: Optional[str] = Field(None, description="State Name")
    state_code: Optional[int] = Field(None, description="State Code")
    district: Optional[str] = Field(None, description="District Name")
    area: Optional[float] = Field(None, description="Cultivated Area ('000 ha)")
    predicted_yield: float = Field(..., description="Estimated/Predicted Yield in kg/ha")
    lower_bound: Optional[float] = Field(None, description="10th percentile bound")
    upper_bound: Optional[float] = Field(None, description="90th percentile bound")
    historical_yield_mean: Optional[float] = Field(None, description="Optional baseline mean")
    historical_yield_std: Optional[float] = Field(None, description="Optional baseline std")
    anomaly_score: Optional[float] = Field(None, description="Optional anomaly score (0-100)")

class RiskComponents(BaseModel):
    uncertainty_risk: float
    historical_deviation_risk: float
    model_error_risk: float
    anomaly_risk: float

class RiskAssessmentResponse(BaseModel):
    risk_level: str = Field(description="LOW | MODERATE | HIGH | CRITICAL")
    risk_score: float = Field(description="Deterministic composite score (0-100)")
    confidence_label: str
    uncertainty_percent: float
    spread: float
    risk_factors: List[str]
    explanation: str
    components: RiskComponents

class ExplainabilityRequest(BaseModel):
    year: int = Field(2017, description="Agricultural Year")
    state: Optional[str] = Field(None, description="State Name")
    state_code: Optional[int] = Field(None, description="State Code")
    district: Optional[str] = Field(None, description="District Name")
    area: float = Field(..., gt=0, description="Rice Area ('000 ha)")
    total_cropped_area: Optional[float] = Field(None, description="Total Cropped Area ('000 ha)")
    rice_area_share: Optional[float] = Field(None, description="Rice Share in Cropland (0-1)")
    wheat_area: Optional[float] = Field(None, description="Wheat Area ('000 ha)")
    cotton_area: Optional[float] = Field(None, description="Cotton Area ('000 ha)")
    sugarcane_area: Optional[float] = Field(None, description="Sugarcane Area ('000 ha)")
    rice_yield_lag1: Optional[float] = Field(None, description="Prior year yield (t-1)")
    rice_yield_roll3: Optional[float] = Field(None, description="3-yr rolling yield average")
    features: Optional[Dict[str, float]] = Field(None, description="Optional feature dict")

class FeatureContributionDetail(BaseModel):
    feature: str
    feature_name: str
    raw_value: str
    contribution_score: float
    direction: str = Field(description="positive | negative | neutral")
    normalized_percentage: float

class ExplainabilityResponse(BaseModel):
    predicted_yield: float
    summary: str
    top_positive_factor: str
    top_negative_factor: str
    feature_contributions: List[FeatureContributionDetail]
    methodology: str

class AnomalyDetectionRequest(BaseModel):
    year: int = Field(2017, description="Agricultural Year")
    state: Optional[str] = Field(None, description="State Name")
    state_code: Optional[int] = Field(None, description="State Code")
    district: Optional[str] = Field(None, description="District Name")
    area: float = Field(..., gt=0, description="Rice Area ('000 ha)")
    production: Optional[float] = Field(None, description="Production in '000 tons")
    yield_val: Optional[float] = Field(None, description="Reported Yield in kg/ha", alias="yield")
    total_cropped_area: Optional[float] = Field(None, description="Total Cropped Area ('000 ha)")
    rice_area_share: Optional[float] = Field(None, description="Rice Cropland Share")
    rice_yield_lag1: Optional[float] = Field(None, description="Prior yield lag")
    rice_yield_roll3: Optional[float] = Field(None, description="3-yr rolling yield lag")

class AnomalyDetectionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(description="Normalized anomaly score (0-100)")
    raw_decision_score: float
    severity: str = Field(description="LOW | MODERATE | HIGH | EXTREME")
    yield_z_score: Optional[float] = None
    yield_deviation_pct: Optional[float] = None
    area_z_score: Optional[float] = None
    area_deviation_pct: Optional[float] = None
    reasons: List[str]
    state: str
    district: str
    year: int

class StateRiskItem(BaseModel):
    state: str
    state_code: int
    record_count: int
    average_yield: float
    yield_volatility: float
    average_prediction_error_mae: float
    anomaly_rate_pct: float
    average_uncertainty_pct: float
    risk_score: float
    risk_level: str

class StateRiskResponse(BaseModel):
    data: List[StateRiskItem]

class AnomalyFeedItem(BaseModel):
    id: str
    state: str
    district: str
    year: int
    area: float
    production: float
    yield_val: float
    anomaly_score: float
    severity: str
    reason: str
    yield_deviation_pct: float

class AnomalyFeedResponse(BaseModel):
    data: List[AnomalyFeedItem]

class IntelligenceDashboardResponse(BaseModel):
    total_records: int
    anomalies_detected: int
    high_risk_states_count: int
    moderate_risk_states_count: int
    low_risk_states_count: int
    average_uncertainty_pct: float
    highest_risk_states: List[StateRiskItem]
    recent_anomalies: List[AnomalyFeedItem]

# =========================================================================
# DAY 17 MULTI-CROP AGRICULTURAL FOUNDATION SCHEMAS
# =========================================================================

class CropItem(BaseModel):
    crop: str
    records: int
    first_year: int
    last_year: int
    states: int
    districts: int
    has_area: bool
    has_production: bool
    has_yield: bool
    forecasting_supported: bool = False
    scenario_supported: bool = False
    xai_supported: bool = False

class CropsResponse(BaseModel):
    total_crops: int
    crops: List[CropItem]

class CropDetailResponse(BaseModel):
    crop: str
    records: int
    first_year: int
    last_year: int
    states: List[str]
    districts_count: int
    total_production_tonnes: float
    average_yield_kg_ha: float
    forecasting_model_status: str
    available_years: List[int]

class CropAvailabilityResponse(BaseModel):
    crop: str
    available: bool
    state_available: bool = True
    district_available: bool = True
    year_available: bool = True
    matching_records: int
    supported_models: List[str] = []

class DatasetMetadataResponse(BaseModel):
    dataset_version: str
    sources: List[str]
    record_count: int
    crop_count: int
    state_count: int
    district_count: int
    year_range: str
    available_crops: List[str]
    available_states: List[str]
    quality_status: str
