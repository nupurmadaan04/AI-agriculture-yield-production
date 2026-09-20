"""
Agricultural Monitoring & Early Warning Pydantic Schemas.

Defines validated data structures for temporal monitoring, early warning signals,
fused alerts, change detection, model health tracking, and historical backtesting.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class TemporalSignal(BaseModel):
    metric_name: str
    record_count: int
    latest_year: Optional[int] = None
    latest_value: float
    yoy_change_pct: float
    yoy_change_absolute: float
    rolling_3yr_mean: float
    rolling_3yr_std: float
    rolling_3yr_zscore: float
    rolling_5yr_mean: float
    rolling_5yr_std: float
    rolling_5yr_zscore: float
    rolling_8yr_mean: float
    rolling_8yr_std: float
    trend_slope: float
    acceleration: float
    volatility_cv: float
    historical_mean: float
    deviation_from_historical: float
    trajectory: List[Dict[str, Any]] = []


class EarlyWarning(BaseModel):
    signal_id: str
    signal_type: str
    state: str
    district: str
    year: int
    severity: str
    trigger_value: float
    threshold: float
    unit: str
    evidence: List[str]
    recommended_action: str
    is_active: bool


class AlertValidationContext(BaseModel):
    r2: float
    mae: float
    drift_status: str
    data_quality_score: float


class Alert(BaseModel):
    alert_id: str
    location: str
    state: str
    district: str
    year: int
    severity: str
    signal_count: int
    dominant_signal: str
    supporting_signals: List[str]
    evidence_strength: str
    composite_risk_score: float
    evidence_chain: List[str]
    recommended_action: str
    model_validation: AlertValidationContext
    priority_score: Optional[float] = None
    priority_rank: Optional[int] = None
    priority_reason: Optional[str] = None


class AlertSummary(BaseModel):
    total_alerts: int
    critical_alerts_count: int
    high_alerts_count: int
    elevated_alerts_count: int
    watch_alerts_count: int
    info_alerts_count: int
    states_under_watch_count: int
    districts_under_watch_count: int


class MonitoringOverview(BaseModel):
    active_alerts_count: int
    high_critical_count: int
    states_under_watch: int
    districts_under_watch: int
    persistent_signals_count: int
    model_monitoring_status: str
    latest_observation_year: int
    summary: AlertSummary
    scientific_disclaimer: str


class RiskSignal(BaseModel):
    signal_type: str
    severity: str
    trigger_value: float
    threshold: float
    unit: str
    description: str


class ChangeDetectionResult(BaseModel):
    metric_name: str
    cusum_analysis: Dict[str, Any]
    trend_break_analysis: Dict[str, Any]
    volatility_shift_cv_delta: float
    overall_change_flag: bool
    scientific_disclaimer: str


class WarningBacktest(BaseModel):
    total_evaluations: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    alert_frequency_pct: float
    mean_lead_time_years: int
    evaluation_years_range: str
    parameters: Dict[str, Any]
    is_chronologically_valid: bool
    scientific_disclaimer: str


class MonitoringHealth(BaseModel):
    status: str
    overall_health_score: float
    data_quality_score: float
    drift_status: str
    prediction_mae: float
    prediction_r2: float
    calibration_quality: str
    data_freshness_label: str
    metrics_breakdown: Dict[str, Any]
    scientific_note: str


class MonitoringHistory(BaseModel):
    total_records: int
    history: List[Alert]


class MonitoringQueryRequest(BaseModel):
    state: Optional[str] = Field(None, description="Filter by state")
    district: Optional[str] = Field(None, description="Filter by district")
    severity: Optional[str] = Field(None, description="Filter by minimum severity")
    signal_type: Optional[str] = Field(None, description="Filter by signal type")
    year: Optional[int] = Field(None, description="Filter by observation year")
    limit: int = Field(50, ge=1, le=200, description="Max alerts to return")


class BacktestRequest(BaseModel):
    yield_drop_threshold_pct: float = Field(-10.0, ge=-50.0, le=-1.0, description="Adverse threshold in %")
    warning_zscore_threshold: float = Field(1.2, ge=0.5, le=4.0, description="Warning trigger z-score")
    lead_time_years: int = Field(1, ge=1, le=3, description="Evaluation lead time in years")
