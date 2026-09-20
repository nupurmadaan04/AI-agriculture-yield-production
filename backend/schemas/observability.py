"""
Pydantic Schemas for Day 28 Production Observability & Operational Intelligence.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SystemHealthItem(BaseModel):
    api_status: str = Field(..., description="API HTTP status (HEALTHY / DEGRADED / UNHEALTHY)")
    readiness_status: str = Field(..., description="Readiness probe status (READY / NOT_READY)")
    backend_status: str = Field(..., description="Backend engine status (OPERATIONAL / DEGRADED)")
    uptime_seconds: float = Field(..., description="Backend process uptime in seconds")
    process_id: int = Field(..., description="Operating system PID")
    active_threads: int = Field(..., description="Active thread count")
    cpu_percent: float = Field(..., description="Current process CPU utilization %")
    system_cpu_percent: float = Field(..., description="Current system-wide CPU utilization %")
    memory_rss_mb: float = Field(..., description="Process resident memory usage in MB")
    system_memory_percent: float = Field(..., description="Host system memory utilization %")
    environment: str = Field(..., description="Deployment environment (LOCAL / CONTAINERIZED / DEVELOPMENT)")
    timestamp: str = Field(..., description="ISO-8601 status timestamp")


class RuntimeMetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    error_requests: int
    error_rate_pct: float
    rps: float
    min_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    sample_count: int
    active_window_seconds: float
    has_runtime_data: bool


class ForecastOperationsMetrics(BaseModel):
    total_forecasts: int
    successful_forecasts: int
    rejected_forecasts: int
    failed_forecasts: int
    success_rate_pct: float
    rejection_rate_pct: float
    forecasts_by_crop: Dict[str, int]
    forecasts_by_strategy: Dict[str, int]
    forecasts_by_status: Dict[str, int]
    recent_forecast_count: int
    has_runtime_data: bool


class StrategyUsageItem(BaseModel):
    crop: str
    strategy: str
    certification_status: str
    algorithm: str
    runtime_invocations_count: int
    runtime_percentage: Optional[float] = None
    fallback_invocations_count: int
    last_used_timestamp: Optional[str] = None


class StrategyMonitoringResponse(BaseModel):
    total_strategies_monitored: int
    total_runtime_observations: int
    has_runtime_data: bool
    strategies: List[StrategyUsageItem]


class TraceStageItem(BaseModel):
    stage_name: str
    status: str
    duration_ms: Optional[float] = None
    timestamp: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class PredictionTraceResponse(BaseModel):
    request_id: str
    timestamp: str
    crop: str
    state: str
    district: str
    forecast_year: int
    strategy: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    prediction: Optional[float] = None
    unit: str = "kg/ha"
    status: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    total_duration_ms: Optional[float] = None
    stages: List[TraceStageItem]
    provenance_hash: Optional[str] = None
    audit_status: str
    model_hash_verified: bool
    dataset_verified: bool


class ModelIntegrityItem(BaseModel):
    model_id: str
    crop: str
    algorithm: str
    version: str
    artifact_path: str
    file_exists: bool
    registered_sha256: str
    actual_sha256: str
    integrity_status: str
    last_checked_timestamp: str


class ModelIntegrityResponse(BaseModel):
    total_models_registered: int
    verified_models_count: int
    failed_models_count: int
    overall_integrity_status: str
    models: List[ModelIntegrityItem]


class DatasetIntegrityResponse(BaseModel):
    dataset_name: str
    dataset_version: str
    file_path: str
    file_exists: bool
    total_records: int
    total_columns: int
    file_size_bytes: int
    sha256_checksum: str
    date_coverage: str
    state_count: int
    district_count: int
    crop_count: int
    schema_status: str
    last_verified_timestamp: str


class StrategyRegistryHealthResponse(BaseModel):
    registry_path: str
    registry_available: bool
    compiled_at: str
    total_strategies_registered: int
    production_ready_count: int
    conditional_production_count: int
    baseline_production_count: int
    coverage_records_count: int
    certification_guard_status: str
    overall_status: str


class OperationalEventItem(BaseModel):
    timestamp: str
    severity: str
    event_type: str
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    status_code: Optional[int] = None
    message: str
    details: Optional[Dict[str, Any]] = None


class OperationalErrorsResponse(BaseModel):
    total_events_logged: int
    total_errors_count: int
    recent_events: List[OperationalEventItem]
    errors_by_category: Dict[str, int]


class AlertConditionItem(BaseModel):
    alert_id: str
    alert_name: str
    severity: str
    metric_name: str
    current_value: Any
    threshold_value: Any
    condition: str
    is_active: bool
    message: str
    timestamp: str


class AlertsResponse(BaseModel):
    active_alerts_count: int
    active_alerts: List[AlertConditionItem]
    resolved_alerts: List[AlertConditionItem]
    configured_thresholds: Dict[str, Any]


class DriftMonitoringResponse(BaseModel):
    monitoring_notice: str
    overall_drift_status: str
    total_features_evaluated: int
    stable_features_count: int
    drifted_features_count: int
    features: List[Dict[str, Any]]


class ObservabilitySummaryResponse(BaseModel):
    system_health: SystemHealthItem
    runtime_metrics: RuntimeMetricsResponse
    forecast_operations: ForecastOperationsMetrics
    model_integrity_status: str
    dataset_integrity_status: str
    strategy_registry_status: str
    active_alerts_count: int
    timestamp: str
