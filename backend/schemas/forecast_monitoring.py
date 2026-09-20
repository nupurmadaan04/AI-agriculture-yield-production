"""
Day 30: Forecast Monitoring, Drift Detection & Outcome Intelligence Schemas.

Defines typed Pydantic models for operational forecast telemetry,
empirical prediction distributions, feature/dataset drift (PSI/KS),
leak-free post-outcome evaluations, error & bias decomposition, and evidence-backed alerts.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Executive Monitoring Summary
# ---------------------------------------------------------------------------

class MonitoringSummaryResponse(BaseModel):
    monitoring_status: str = Field(..., description="Overall monitoring state: HEALTHY, WATCH, DRIFT_DETECTED, EVALUATION_UNAVAILABLE")
    status_reason: str = Field(..., description="Evidence-backed description of the overall monitoring status")
    total_forecast_requests: int = Field(..., description="Total recorded forecast operations in audit log")
    successful_forecasts: int = Field(..., description="Total successful forecast predictions served")
    rejected_requests: int = Field(..., description="Total rejected/blocked forecast requests")
    evaluated_outcomes_count: int = Field(..., description="Total frozen forecasts paired with verified observed outcomes")
    active_alerts_count: int = Field(..., description="Number of active operational or statistical drift alerts")
    monitored_crops_count: int = Field(..., description="Number of active monitored crops")
    dataset_version: str = Field(default="AGRI_PANEL_1.0", description="Canonical panel dataset version")
    timestamp: str = Field(..., description="ISO timestamp of summary generation")


# ---------------------------------------------------------------------------
# 2. Operational Telemetry & Request Tracking
# ---------------------------------------------------------------------------

class OperationalTimeSeriesPoint(BaseModel):
    date: str
    total_requests: int
    successful_requests: int
    rejected_requests: int


class CropUsageItem(BaseModel):
    crop: str
    request_count: int
    percentage: float


class StrategyUsageItem(BaseModel):
    strategy: str
    certification_status: str
    request_count: int
    percentage: float


class ForecastOperationsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    rejected_requests: int
    failed_requests: int
    success_rate_pct: float
    time_series: List[OperationalTimeSeriesPoint] = Field(default_factory=list)
    crop_breakdown: List[CropUsageItem] = Field(default_factory=list)
    strategy_breakdown: List[StrategyUsageItem] = Field(default_factory=list)
    semantic_classification: str = "MONITORING"
    notes: str = ""


# ---------------------------------------------------------------------------
# 3. Prediction Distribution Monitoring
# ---------------------------------------------------------------------------

class StatisticalMoments(BaseModel):
    count: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    p10: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    p90: Optional[float] = None


class DistributionHistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int
    density: float


class PredictionDistributionItem(BaseModel):
    crop: str
    strategy: str
    unit: str = "kg/ha"
    current_predictions: StatisticalMoments
    historical_reference: StatisticalMoments
    histogram_bins: List[DistributionHistogramBin] = Field(default_factory=list)
    distribution_shift_detected: bool = False
    shift_metric: Optional[str] = None
    shift_value: Optional[float] = None
    semantic_classification: str = "MONITORING"


class PredictionDistributionResponse(BaseModel):
    total_monitored_crops: int
    distributions: List[PredictionDistributionItem] = Field(default_factory=list)
    semantic_classification: str = "MONITORING"
    evaluation_window: str = "Live Operational Requests"
    reference_window: str = "Historical Panel 1966-2017"


# ---------------------------------------------------------------------------
# 4. Statistical Drift Monitoring (PSI / KS)
# ---------------------------------------------------------------------------

class FeatureDriftItem(BaseModel):
    feature_name: str
    metric: str = "PSI"
    observed_value: float
    p_value: Optional[float] = None
    threshold: float
    status: str = Field(..., description="NO_DRIFT, MODERATE_DRIFT, SIGNIFICANT_DRIFT, MONITORING_ONLY")
    reference_window: str = "2010-2015"
    evaluation_window: str = "2016-2017"
    reference_samples: int
    evaluation_samples: int
    evidence: str
    semantic_classification: str = "MONITORING"


class CoverageDriftItem(BaseModel):
    dimension: str
    reference_count: int
    current_count: int
    coverage_ratio: float
    status: str
    notes: str


class DriftMonitoringResponse(BaseModel):
    overall_drift_status: str
    features: List[FeatureDriftItem] = Field(default_factory=list)
    coverage_drift: List[CoverageDriftItem] = Field(default_factory=list)
    missingness_drift_pct: float = 0.0
    threshold_source: str = "Standard PSI Industry Guidelines (PSI < 0.10: Stable, 0.10-0.25: Moderate, > 0.25: Significant)"
    semantic_classification: str = "MONITORING"
    notes: str = ""


# ---------------------------------------------------------------------------
# 5. Leak-Free Post-Outcome Evaluation
# ---------------------------------------------------------------------------

class OutcomeEvaluationItem(BaseModel):
    crop: str
    state: str
    district: str
    forecast_year: int
    forecast_origin: int
    predicted_yield: float
    observed_yield: float
    signed_error: float
    absolute_error: float
    relative_error_pct: Optional[float] = None
    strategy: str
    model_version: str
    unit: str = "kg/ha"
    evaluation_status: str = "VERIFIED_OUTCOME"
    semantic_classification: str = "POST_OUTCOME_EVALUATION"


class OutcomeEvaluationSummary(BaseModel):
    crop: str
    evaluated_samples: int
    mae: float
    rmse: float
    median_absolute_error: float
    mean_signed_bias: float
    mape: Optional[float] = None
    evaluation_years: List[int] = Field(default_factory=list)
    status: str = "EVALUATED"


class OutcomeEvaluationResponse(BaseModel):
    status: str = Field(..., description="EVALUATED, EVALUATION_UNAVAILABLE, INSUFFICIENT_EVIDENCE")
    reason: str = ""
    summary: Optional[OutcomeEvaluationSummary] = None
    records: List[OutcomeEvaluationItem] = Field(default_factory=list)
    total_records: int = 0
    temporal_boundary_rule: str = "Strict Pre-Forecast Freezing: forecast_origin < forecast_year and observed_year == forecast_year"
    semantic_classification: str = "POST_OUTCOME_EVALUATION"


# ---------------------------------------------------------------------------
# 6. Error Decomposition & Stratification
# ---------------------------------------------------------------------------

class TemporalErrorItem(BaseModel):
    year: int
    evaluated_forecasts: int
    mae: float
    rmse: float
    bias: float
    p25_error: float
    p75_error: float


class GeographicErrorItem(BaseModel):
    state: str
    district: str
    evaluated_forecasts: int
    mae: float
    rmse: float
    bias: float


class RegimeErrorItem(BaseModel):
    regime: str = Field(..., description="Low (<=Q25), Normal (Q25-Q75), High (>=Q75)")
    sample_count: int
    mae: float
    rmse: float
    mean_signed_bias: float


class ErrorDecompositionResponse(BaseModel):
    crop: str
    temporal_breakdown: List[TemporalErrorItem] = Field(default_factory=list)
    geographic_breakdown: List[GeographicErrorItem] = Field(default_factory=list)
    regime_breakdown: List[RegimeErrorItem] = Field(default_factory=list)
    semantic_classification: str = "POST_OUTCOME_EVALUATION"
    notes: str = ""


# ---------------------------------------------------------------------------
# 7. Directional Systematic Bias Analysis
# ---------------------------------------------------------------------------

class CropBiasItem(BaseModel):
    crop: str
    mean_actual_yield: float
    mean_residual: float
    median_residual: float
    normalized_mean_error_pct: float
    bias_status: str = Field(..., description="OVER_PREDICTION_BIAS, UNDER_PREDICTION_BIAS, NO_CLEAR_BIAS")
    bias_description: str
    bias_threshold_rule: str
    sample_count: int = 0
    semantic_classification: str = "POST_OUTCOME_EVALUATION"


class BiasAnalysisResponse(BaseModel):
    crops: List[CropBiasItem] = Field(default_factory=list)
    methodology: str = "bias = mean(predicted - observed), NME% = (mean_residual / mean_actual) * 100"
    semantic_classification: str = "POST_OUTCOME_EVALUATION"
    notes: str = ""


# ---------------------------------------------------------------------------
# 8. Evidence-First Monitoring Alerts
# ---------------------------------------------------------------------------

class MonitoringAlertItem(BaseModel):
    alert_id: str
    timestamp: str
    severity: str = Field(..., description="INFO, WATCH, WARNING, CRITICAL")
    category: str = Field(..., description="DRIFT, BIAS, OPERATIONAL, INTEGRITY, DATASET")
    signal: str
    metric: str
    observed_value: str
    threshold: Optional[str] = None
    reference_window: Optional[str] = None
    evaluation_window: Optional[str] = None
    sample_size: Optional[int] = None
    crop: Optional[str] = None
    evidence: str
    recommended_action: str


class MonitoringAlertsResponse(BaseModel):
    active_alerts: List[MonitoringAlertItem] = Field(default_factory=list)
    total_alerts: int
    has_critical_alerts: bool = False
    timestamp: str
    semantic_classification: str = "MONITORING"


# ---------------------------------------------------------------------------
# 9. Health & System Status
# ---------------------------------------------------------------------------

class MonitoringHealthResponse(BaseModel):
    status: str = "HEALTHY"
    subsystem: str = "forecast-monitoring-engine"
    version: str = "1.0.0"
    audit_records_available: int
    telemetry_records_available: int
    outcomes_dataset_available: bool
    timestamp: str
