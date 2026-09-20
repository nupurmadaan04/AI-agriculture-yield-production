"""
Pydantic Schemas for Multi-Crop Modeling Readiness, Baselines, Forecasting, and Temporal Robustness
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CropReadinessItem(BaseModel):
    crop: str
    readiness_score: float
    readiness_status: str
    data_volume_score: float
    temporal_score: float
    geographic_score: float
    target_quality_score: float
    validation_score: float
    feature_score: float
    blocking_reasons: str
    recommendation: str
    total_records: Optional[int] = None
    active_districts: Optional[int] = None
    zero_yield_pct: Optional[float] = None
    best_baseline_model: Optional[str] = None
    best_baseline_mae: Optional[float] = None
    best_baseline_r2: Optional[float] = None


class CropReadinessResponse(BaseModel):
    total_crops: int
    crops: List[CropReadinessItem]


class CropBaselineItem(BaseModel):
    crop: str
    model: str
    train_period: str
    test_period: str
    train_records: int
    test_records: int
    mae: Optional[float]
    rmse: Optional[float]
    r2: Optional[float]
    mape: Optional[float]
    smape: Optional[float]
    valid_predictions: int
    invalid_predictions: int
    notes: str
    status: str


class CropBaselinesResponse(BaseModel):
    crop: str
    baselines: List[CropBaselineItem]
    best_model_by_mae: Optional[str] = None
    best_mae: Optional[float] = None


class ReadinessSummaryResponse(BaseModel):
    total_crops: int
    model_ready_count: int
    analytics_ready_count: int
    insufficient_data_count: int
    model_ready_crops: List[str]
    analytics_ready_crops: List[str]
    insufficient_data_crops: List[str]
    total_records_evaluated: int
    active_dataset_version: str


class FeatureCompatibilityItem(BaseModel):
    feature_name: str
    source_type: str
    timing: str
    pre_season_valid: bool
    leakage_risk: str
    classification: str
    recommendation: str


class FeatureCompatibilityResponse(BaseModel):
    total_features_audited: int
    features: List[FeatureCompatibilityItem]


class ArchitectureDecisionResponse(BaseModel):
    decision: str
    recommended_architecture: str
    global_model_justified: bool
    crop_specific_justified: bool
    hierarchical_justified: bool
    summary: str
    empirical_justification: List[str]


# ---------------------------------------------------------------------------
# Day 19 Multi-Crop Forecasting Schemas
# ---------------------------------------------------------------------------

class MultiCropModelItem(BaseModel):
    crop: str
    model_id: str
    algorithm: str
    version: str
    training_period: str
    evaluation_period: str
    train_records: int
    test_records: int
    mae: float
    rmse: float
    r2: float
    mape: Optional[float] = None
    smape: Optional[float] = None
    baseline_model: str
    baseline_mae: float
    mae_improvement_pct: float
    model_status: str
    recommendation: str
    artifact_path: str
    sha256: str


class MultiCropModelsResponse(BaseModel):
    total_models: int
    accepted_count: int
    baseline_preferred_count: int
    models: List[MultiCropModelItem]


class CropModelComparisonResponse(BaseModel):
    crop: str
    baseline_model: str
    baseline_mae: float
    baseline_rmse: float
    baseline_r2: Optional[float] = None
    rf_mae: float
    rf_rmse: float
    rf_r2: float
    gb_mae: float
    gb_rmse: float
    gb_r2: float
    ml_winner: str
    overall_winner: str
    best_mae: float
    mae_improvement_vs_baseline: float
    mae_improvement_pct: float
    model_status: str
    recommendation: str


class CropModelMetricsResponse(BaseModel):
    crop: str
    algorithm: str
    status: str
    training_period: str
    evaluation_period: str
    train_records: int
    test_records: int
    metrics: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    error_analysis: Dict[str, Any]
    uncertainty_spread_p10_p90: Optional[float] = None


class CropModelFeaturesResponse(BaseModel):
    crop: str
    algorithm: str
    features: List[str]
    feature_importance_native: Dict[str, float]
    feature_importance_permutation: Dict[str, float]


class MultiCropLeaderboardItem(BaseModel):
    crop: str
    best_model: str
    best_mae: float
    best_rmse: float
    best_r2: Optional[float] = None
    baseline_model: str
    baseline_mae: float
    mae_improvement_pct: float
    model_status: str


class MultiCropLeaderboardResponse(BaseModel):
    total_crops: int
    accepted_count: int
    baseline_preferred_count: int
    leaderboard: List[MultiCropLeaderboardItem]


class MultiCropRegistryResponse(BaseModel):
    version: str
    last_updated: str
    total_models_registered: int
    models: Dict[str, Any]


class CropPredictionRequest(BaseModel):
    crop: str
    state: str
    district: str
    year: int = Field(default=2018, description="Target forecast year")
    yield_lag_1: Optional[float] = Field(default=None, description="Previous season yield (kg/ha)")
    yield_lag_2: Optional[float] = Field(default=None, description="2-season prior yield (kg/ha)")
    yield_rolling_3yr_mean: Optional[float] = Field(default=None, description="3-year rolling average yield (kg/ha)")
    area_lag_1: Optional[float] = Field(default=None, description="Previous season cultivated area (ha)")


class CropPredictionResponse(BaseModel):
    crop: str
    state: str
    district: str
    target_year: int
    predicted_yield_kg_ha: float
    p10_lower_kg_ha: Optional[float] = None
    p90_upper_kg_ha: Optional[float] = None
    model_id: str
    algorithm: str
    model_version: str
    model_scope: str
    model_status: str
    dataset_version: str
    provenance: Dict[str, Any]


# ---------------------------------------------------------------------------
# Day 20 Temporal Robustness Schemas
# ---------------------------------------------------------------------------

class FoldResultItem(BaseModel):
    crop: str
    fold_id: int
    train_start_year: int
    train_end_year: int
    test_year: int
    train_samples: int
    test_samples: int
    model: str
    mae: float
    rmse: float
    r2: float
    mape: Optional[float] = None
    smape: Optional[float] = None
    best_baseline_model: str
    best_baseline_mae: float
    win_vs_baseline: bool
    mae_improvement_pct: float
    mean_residual: float
    std_residual: float


class CropRobustnessItem(BaseModel):
    crop: str
    model: str
    fold_count: int
    mean_mae: float
    median_mae: float
    std_mae: float
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float
    baseline_mae: float
    mean_mae_improvement: float
    median_mae_improvement: float
    win_rate: float
    status: str
    robustness_score: Optional[float] = None


class CropRobustnessResponse(BaseModel):
    total_crops: int
    robust_accepted_count: int
    split_sensitive_count: int
    baseline_preferred_count: int
    crops: List[CropRobustnessItem]


class CropFoldsResponse(BaseModel):
    crop: str
    fold_count: int
    folds: List[FoldResultItem]


class CropRobustnessDetailResponse(BaseModel):
    crop: str
    robustness_status: str
    robustness_score: float
    best_model: str
    evaluated_ml_model: str
    recommendation: str
    models: Dict[str, Any]
    folds: List[FoldResultItem]
    feature_stability: List[Dict[str, Any]]


class RobustnessSummaryResponse(BaseModel):
    total_crops_evaluated: int
    total_walk_forward_folds: int
    robust_accepted_count: int
    split_sensitive_count: int
    baseline_preferred_count: int
    robust_accepted_crops: List[str]
    split_sensitive_crops: List[str]
    baseline_preferred_crops: List[str]
    mean_win_rate: float
    dataset_version: str


# ==========================================
# Day 21: Model Diagnosis & Strategy Schemas
# ==========================================

class CropDiagnosisItem(BaseModel):
    crop: str
    selected_ml_model: str
    total_folds: int
    total_test_observations: int
    historical_median_yield: float
    ml_wins: int
    ml_losses: int
    win_rate: float
    mean_mae_improvement_pct: float
    median_mae_improvement_pct: float
    worst_fold_degradation_pct: float
    best_fold_improvement_pct: float
    ml_mae_mean: float
    ml_mae_median: float
    ml_mae_std: float
    ml_mae_min: float
    ml_mae_max: float
    ml_mae_cv: float
    base_mae_mean: float
    base_mae_median: float
    base_mae_std: float
    base_mae_min: float
    base_mae_max: float
    base_mae_cv: float
    p25_error_ml: float
    p50_error_ml: float
    p75_error_ml: float
    p90_error_ml: float
    p95_error_ml: float
    p25_error_base: float
    p50_error_base: float
    p75_error_base: float
    p90_error_base: float
    p95_error_base: float
    pct_errors_lt_100: float
    pct_errors_lt_250: float
    pct_errors_lt_500: float
    pct_errors_gt_1000: float
    normalized_error_p50: float
    normalized_error_p90: float


class CropDiagnosisSummaryResponse(BaseModel):
    total_crops: int
    crops: List[CropDiagnosisItem]
    methodology_version: str


class CropErrorRegimeItem(BaseModel):
    crop: str
    regime: str
    n_observations: int
    yield_min_kg_ha: float
    yield_max_kg_ha: float
    ml_mae: float
    baseline_mae: float
    ml_improvement_pct: float
    obs_win_rate: float
    regime_status: str


class CropErrorRegimesResponse(BaseModel):
    crop: str
    regimes: List[CropErrorRegimeItem]


class CropDistrictErrorItem(BaseModel):
    crop: str
    state: str
    district: str
    observations_count: int
    ml_mae: float
    baseline_mae: float
    ml_improvement_pct: float
    ml_win_rate: float
    is_best_ml_district: bool
    is_worst_ml_district: bool
    is_high_error_district: bool


class CropDistrictErrorsResponse(BaseModel):
    crop: str
    total_districts: int
    best_ml_districts_count: int
    worst_ml_districts_count: int
    high_error_districts_count: int
    districts: List[CropDistrictErrorItem]


class CropYearErrorItem(BaseModel):
    crop: str
    year: int
    fold_id: int
    ml_model: str
    ml_mae: float
    ml_rmse: float
    baseline_model: str
    baseline_mae: float
    baseline_rmse: float
    ml_improvement_pct: float
    ml_win: bool
    temporal_regime: str


class CropYearErrorsResponse(BaseModel):
    crop: str
    years: List[CropYearErrorItem]


class CropFeatureStabilityItem(BaseModel):
    crop: str
    feature: str
    model: str
    mean_importance: float
    std_importance: float
    mean_rank: float
    rank_variance: float
    feature_stability_score: float
    interpretative_role: str


class CropFeatureStabilityResponse(BaseModel):
    crop: str
    features: List[CropFeatureStabilityItem]
    feature_timing_audit: List[Dict[str, Any]]


class CropModelSelectionItem(BaseModel):
    crop: str
    day19_status: str
    day20_status: str
    day21_status: str
    win_rate: float
    mean_mae_improvement_pct: float
    median_mae_improvement_pct: float
    worst_fold_degradation_pct: float
    ml_mae_cv: float
    feature_timing_status: str
    total_test_observations: int
    decision_basis: str
    methodology_version: str


class CropModelSelectionResponse(BaseModel):
    total_crops: int
    robust_ml_count: int
    ml_with_conditions_count: int
    baseline_preferred_count: int
    research_candidate_count: int
    insufficient_evidence_count: int
    selections: List[CropModelSelectionItem]


class CropForecastingStrategyItem(BaseModel):
    crop: str
    day21_status: str
    primary_forecasting_model: str
    fallback_model: str
    operating_conditions: str
    diagnostic_notes: str
    missing_information_gaps: str
    evidence_required: str
    evidence_strength_score: float
    evidence_interpretation: str


class CropForecastingStrategyResponse(BaseModel):
    total_crops: int
    strategies: List[CropForecastingStrategyItem]


# ---------------------------------------------------------------------------
# Day 22 Exogenous Data & Pre-Season Feature Expansion Schemas
# ---------------------------------------------------------------------------

class ExogenousSourceItem(BaseModel):
    source_id: str
    source_name: str
    provider: str
    tier: str
    spatial_level: str
    temporal_resolution: str
    temporal_coverage: str
    status: str
    preseason_availability_date: str
    license_terms: str


class ExogenousSourcesResponse(BaseModel):
    total_sources: int
    sources: List[ExogenousSourceItem]


class ExogenousCoverageItem(BaseModel):
    crop: str
    records: int
    weather_coverage_pct: float
    soil_coverage_pct: float
    overall_exogenous_coverage_pct: float
    missing_pct: float
    district_coverage: int
    year_coverage: int
    coverage_status: str


class ExogenousCoverageResponse(BaseModel):
    total_crops: int
    crops: List[ExogenousCoverageItem]


class ExogenousFeatureItem(BaseModel):
    feature: str
    source: str
    spatial_level: str
    temporal_resolution: str
    observation_period: str
    availability_date: str
    forecast_origin: str
    lag: int
    unit: str
    transformation: str
    leakage_status: str


class ExogenousFeaturesResponse(BaseModel):
    total_features: int
    features: List[ExogenousFeatureItem]
    temporal_audit: List[Dict[str, Any]]
    leakage_audit: List[Dict[str, Any]]


class ExogenousAblationItem(BaseModel):
    crop: str
    experiment_id: str
    experiment_name: str
    feature_group: str
    algorithm: str
    n_features: int
    mean_mae: float
    median_mae: float
    mean_rmse: float
    mean_r2: float
    mean_mape: float
    mean_smape: float
    baseline_mean_mae: float
    historical_mean_mae: float
    mean_improvement_vs_baseline_pct: float
    median_improvement_vs_baseline_pct: float
    mean_improvement_vs_historical_pct: float
    win_rate_vs_baseline: float
    win_rate_vs_historical: float


class ExogenousAblationResponse(BaseModel):
    total_records: int
    ablations: List[ExogenousAblationItem]


class ExogenousFoldResultItem(BaseModel):
    crop: str
    fold_id: int
    test_year: int
    experiment_id: str
    experiment_name: str
    feature_group: str
    n_features: int
    algorithm: str
    test_records: int
    mae: float
    rmse: float
    r2: float
    mape: float
    smape: float
    baseline_mae: float
    historical_mae: float
    improvement_vs_baseline_pct: float
    improvement_vs_historical_pct: float
    win_vs_baseline: bool
    win_vs_historical: bool


class ExogenousCropFoldsResponse(BaseModel):
    crop: str
    total_folds: int
    folds: List[ExogenousFoldResultItem]


class ExogenousCropResultItem(BaseModel):
    crop: str
    algorithm: str
    model_a_hist_mae: float
    model_a_hist_rmse: float
    model_a_hist_r2: float
    model_b_exo_mae: float
    model_b_exo_rmse: float
    model_b_exo_r2: float
    model_c_base_mae: float
    model_c_base_rmse: float
    exogenous_gain_vs_historical_pct: float
    exogenous_gain_vs_baseline_pct: float
    artifact_path: str


class ExogenousCropResultResponse(BaseModel):
    crop: str
    result: ExogenousCropResultItem
    folds: List[ExogenousFoldResultItem]
    ablation_tiers: List[ExogenousAblationItem]


class ExogenousModelSelectionItem(BaseModel):
    crop: str
    day21_status: str
    day22_status: str
    model_a_hist_mae: float
    model_b_exo_mae: float
    model_c_base_mae: float
    gain_vs_historical_pct: float
    gain_vs_baseline_pct: float
    win_rate_vs_historical: float
    win_rate_vs_baseline: float
    shock_year_2016_gain_pct: float
    best_ablation_tier: str
    decision_basis: str


class ExogenousModelSelectionResponse(BaseModel):
    total_crops: int
    exogenous_robust_count: int
    exogenous_conditional_count: int
    no_meaningful_gain_count: int
    insufficient_coverage_count: int
    selections: List[ExogenousModelSelectionItem]


class ExogenousSummaryResponse(BaseModel):
    total_crops_evaluated: int
    total_sources: int
    total_exogenous_features: int
    exogenous_robust_count: int
    exogenous_conditional_count: int
    no_meaningful_gain_count: int
    insufficient_coverage_count: int
    largest_mae_improvement_crop: str
    largest_mae_improvement_pct: float
    shock_year_2016_average_gain_pct: float
    methodology_version: str


# ============================================================================
# DAY 23: FINAL VALIDATION, STRATEGY, RESIDUALS & CERTIFICATION SCHEMAS
# ============================================================================

class FinalStrategyItem(BaseModel):
    crop: str
    primary_model: str
    fallback_model: str
    policy_type: str
    strategy_mean_mae: float
    ml_mean_mae: float
    baseline_mean_mae: float
    strategy_gain_vs_baseline_pct: float
    strategy_gain_vs_ml_pct: float
    operating_rule: str


class FinalValidationFoldItem(BaseModel):
    crop: str
    fold_id: int
    test_year: int
    test_records: int
    policy_type: str
    strategy_mae: float
    strategy_rmse: float
    strategy_r2: float
    strategy_mape: float
    strategy_smape: float
    ml_mae: float
    ml_rmse: float
    ml_r2: float
    ml_mape: float
    baseline_mae: float
    baseline_rmse: float
    baseline_r2: float
    baseline_mape: float
    strategy_improvement_vs_baseline_pct: float
    strategy_improvement_vs_ml_pct: float
    strategy_win_vs_baseline: bool
    strategy_win_vs_ml: bool


class FinalValidationResponse(BaseModel):
    total_crops_certified: int
    production_ready_count: int
    conditional_production_count: int
    baseline_production_count: int
    research_only_count: int
    not_ready_count: int
    temporal_range_statement: str
    strategies: List[FinalStrategyItem]


class SingleCropFinalValidationResponse(BaseModel):
    crop: str
    strategy: FinalStrategyItem
    folds: List[FinalValidationFoldItem]


class ResidualQuantileItem(BaseModel):
    crop: str
    total_eval_samples: int
    mean_residual: float
    median_residual: float
    std_residual: float
    mae: float
    rmse: float
    p25_abs_error: float
    p50_abs_error: float
    p75_abs_error: float
    p90_abs_error: float
    p95_abs_error: float
    mae_q1_lowest_yield: float
    mae_q2_lower_mid_yield: float
    mae_q3_upper_mid_yield: float
    mae_q4_highest_yield: float


class ResidualYearItem(BaseModel):
    crop: str
    year: int
    fold_id: int
    records: int
    mean_residual: float
    median_residual: float
    mae: float
    rmse: float
    p90_abs_error: float
    temporal_regime: str


class ResidualDiagnosticsResponse(BaseModel):
    crop: str
    quantiles: ResidualQuantileItem
    years: List[ResidualYearItem]


class PredictionBiasItem(BaseModel):
    crop: str
    mean_actual_yield: float
    mean_residual: float
    median_residual: float
    normalized_mean_error_pct: float
    bias_status: str
    bias_description: str
    bias_threshold_rule: str


class PredictionBiasResponse(BaseModel):
    crop: str
    bias: PredictionBiasItem


class ReproducibilityItem(BaseModel):
    crop: str
    dataset_sha256: str
    feature_matrix_sha256: str
    model_artifact_sha256: str
    run1_prediction_hash: str
    run2_prediction_hash: str
    max_absolute_prediction_diff: float
    bitwise_reproducible: bool
    status: str


class ReproducibilityResponse(BaseModel):
    audit_title: str
    overall_status: str
    total_crops_audited: int
    verified_bitwise_count: int
    reproducibility_rate_pct: float
    crops: List[ReproducibilityItem]


class FinalModelCertificationItem(BaseModel):
    crop: str
    final_status: str
    primary_strategy: str
    fallback_strategy: str
    strategy_mae: float
    ml_mae: float
    baseline_mae: float
    gain_vs_baseline_pct: float
    gain_vs_ml_pct: float
    fold_win_rate_pct: float
    mean_residual: float
    p90_abs_error: float
    bias_status: str
    reproducibility_status: str
    feature_timing_safety: str
    operational_evidence: str
    governance_directive: str


class FinalModelCertificationResponse(BaseModel):
    total_crops_certified: int
    production_ready_count: int
    conditional_production_count: int
    baseline_production_count: int
    research_only_count: int
    not_ready_count: int
    certifications: List[FinalModelCertificationItem]


# ---------------------------------------------------------------------------
# Day 24 Production Forecast Serving & Governance Schemas
# ---------------------------------------------------------------------------

class ForecastPredictRequest(BaseModel):
    crop: str = Field(..., description="Crop commodity name (e.g. Oilseeds, Rice, Sugarcane)")
    state: str = Field(..., description="State name (e.g. Punjab, Bihar)")
    district: str = Field(..., description="District name (e.g. Ludhiana, Patna)")
    forecast_year: Optional[int] = Field(default=2018, description="Target forecast year")
    yield_lag_1: Optional[float] = Field(default=None, description="Optional 1-year yield lag (kg/ha)")
    yield_rolling_3yr_mean: Optional[float] = Field(default=None, description="Optional 3-year rolling yield mean (kg/ha)")
    area_lag_1: Optional[float] = Field(default=None, description="Optional 1-year area lag (ha)")


class ForecastPredictResponse(BaseModel):
    status: str
    request_id: str
    prediction: Optional[float] = None
    unit: str = "kg/ha"
    crop: str
    state: str
    district: str
    forecast_year: Optional[int] = 2018
    strategy: Optional[str] = None
    certification_status: str
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    model_version: Optional[str] = None
    validation_scope: str = "walk_forward_2014_2017"
    evidence_type: str = "PREDICTED"
    operating_rule: Optional[str] = None
    strategy_explanation: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None


class ForecastStrategyItem(BaseModel):
    crop: str
    certification_status: str
    primary_strategy: str
    fallback_strategy: str
    strategy_mae: float
    baseline_mae: float
    gain_vs_baseline_pct: float
    fold_win_rate_pct: float
    bias_status: str
    reproducibility_status: str
    model_name: str
    model_version: str
    model_artifact: Optional[str] = None
    validation_scope: str
    operating_rule: str
    strategy_explanation: str


class ForecastStrategiesResponse(BaseModel):
    total_strategies: int
    version: str
    validation_scope: str
    temporal_boundary: str
    strategies: List[ForecastStrategyItem]


class ForecastCoverageItem(BaseModel):
    crop: str
    state: str
    district: str
    min_year: int
    max_year: int
    total_observations: int


class ForecastCoverageResponse(BaseModel):
    total_records: int
    unique_crops: List[str]
    unique_states: List[str]
    unique_districts_count: int
    coverage: List[ForecastCoverageItem]


class ForecastCertificationSummaryResponse(BaseModel):
    total_crops_certified: int
    production_ready_crops: List[str]
    conditional_production_crops: List[str]
    baseline_production_crops: List[str]
    governance_policy: str
    certification_source: str


class ForecastAuditItem(BaseModel):
    request_id: str
    timestamp: str
    crop: str
    state: str
    district: str
    forecast_year: int
    strategy: str
    certification_status: str
    status: str
    prediction: Optional[Any] = None
    unit: str
    fallback_used: bool
    provenance_hash: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ForecastAuditResponse(BaseModel):
    total_events: int
    events: List[ForecastAuditItem]


class ForecastHealthResponse(BaseModel):
    status: str
    service: str
    version: str
    certified_crops_count: int
    coverage_districts_count: int
    governance_guard: str
    provenance_tracking: str


# ---------------------------------------------------------------------------
# Day 29 Prediction Explorer & Forecast Explainability Schemas
# ---------------------------------------------------------------------------

class HistoricalObservationItem(BaseModel):
    year: int
    yield_kg_ha: float
    area_ha: Optional[float] = None
    production_tonnes: Optional[float] = None
    observation_type: str = "OBSERVED"


class ForecastContextResponse(BaseModel):
    crop: str
    state: str
    district: str
    forecast_year: int
    historical_observations_count: int
    district_historical_mean: Optional[float] = None
    previous_year_yield: Optional[float] = None
    rolling_3yr_mean: Optional[float] = None
    historical_min_yield: Optional[float] = None
    historical_max_yield: Optional[float] = None
    historical_yield_std: Optional[float] = None
    recent_observations: List[HistoricalObservationItem] = Field(default_factory=list)
    has_sufficient_history: bool = True
    context_notes: str = ""


class ModelFeatureImportanceItem(BaseModel):
    feature_name: str
    importance_pct: float
    contribution_direction: Optional[str] = None
    description: Optional[str] = None


class ForecastEvidenceResponse(BaseModel):
    crop: str
    strategy_name: str
    certification_status: str
    model_family: Optional[str] = None
    model_version: Optional[str] = None
    validation_protocol: str = "4-Origin Expanding Walk-Forward (2014-2017)"
    mean_mae: Optional[float] = None
    baseline_mae: Optional[float] = None
    mean_improvement_pct: Optional[float] = None
    fold_win_rate_pct: Optional[float] = None
    is_ml_strategy: bool = False
    empirical_p10_p90_spread: Optional[float] = None
    feature_importance: List[ModelFeatureImportanceItem] = Field(default_factory=list)
    operating_rule: str = ""
    fallback_strategy: str = ""
    explanation_notice: str = ""




