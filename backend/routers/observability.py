"""
Observability & Operational Intelligence API Endpoints for Day 28.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from src.observability_engine import observability_engine
from backend.schemas.observability import (
    SystemHealthItem,
    RuntimeMetricsResponse,
    ForecastOperationsMetrics,
    StrategyMonitoringResponse,
    PredictionTraceResponse,
    ModelIntegrityResponse,
    DatasetIntegrityResponse,
    StrategyRegistryHealthResponse,
    OperationalErrorsResponse,
    AlertsResponse,
    DriftMonitoringResponse,
    ObservabilitySummaryResponse,
)

router = APIRouter(prefix="/api/observability", tags=["Production Observability & Operations"])


@router.get("/summary", response_model=ObservabilitySummaryResponse)
async def get_observability_summary():
    """Returns high-level system operational summary and health status."""
    try:
        return observability_engine.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch observability summary: {str(e)}")


@router.get("/health", response_model=SystemHealthItem)
async def get_system_health():
    """Returns live, non-fabricated system and process telemetry."""
    try:
        return observability_engine.get_system_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch system health: {str(e)}")


@router.get("/metrics", response_model=RuntimeMetricsResponse)
async def get_runtime_metrics():
    """Returns runtime request telemetry, throughput, and latency percentiles (P50, P90, P95, P99)."""
    try:
        return observability_engine.get_runtime_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch runtime metrics: {str(e)}")


@router.get("/forecasts", response_model=ForecastOperationsMetrics)
async def get_forecast_operations():
    """Returns runtime forecast operation counts, success/rejection breakdown, and crop distributions."""
    try:
        return observability_engine.get_forecast_operations_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch forecast operations: {str(e)}")


@router.get("/strategies", response_model=StrategyMonitoringResponse)
async def get_strategy_monitoring():
    """Returns observed runtime invocations and usage per strategy."""
    try:
        return observability_engine.get_strategy_monitoring()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategy monitoring: {str(e)}")


@router.get("/models", response_model=ModelIntegrityResponse)
async def verify_models_integrity():
    """Performs live cryptographic SHA-256 verification of all registered model files on disk."""
    try:
        return observability_engine.verify_model_integrity()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify model integrity: {str(e)}")


@router.get("/dataset", response_model=DatasetIntegrityResponse)
async def verify_dataset_integrity():
    """Validates availability, checksum, row count, and schema of canonical AGRI_PANEL_1.0."""
    try:
        return observability_engine.verify_dataset_integrity()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify dataset integrity: {str(e)}")


@router.get("/registry", response_model=StrategyRegistryHealthResponse)
async def get_strategy_registry_health():
    """Validates multi-crop strategy registry completeness and certification guard status."""
    try:
        return observability_engine.get_strategy_registry_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch strategy registry health: {str(e)}")


@router.get("/trace/{request_id}", response_model=PredictionTraceResponse)
async def get_prediction_trace(request_id: str):
    """Retrieves granular stage-by-stage execution trace and timing for a specific forecast Request ID."""
    trace = observability_engine.get_forecast_trace(request_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Forecast execution trace for '{request_id}' not found.")
    return trace


@router.get("/errors", response_model=OperationalErrorsResponse)
async def get_operational_errors(limit: int = Query(default=50, ge=1, le=200)):
    """Retrieves recent operational events, error categorizations, and warnings."""
    try:
        return observability_engine.get_operational_events(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch operational events: {str(e)}")


@router.get("/alerts", response_model=AlertsResponse)
async def get_operational_alerts():
    """Evaluates live operational alert conditions against configured thresholds."""
    try:
        return observability_engine.evaluate_alerts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate alerts: {str(e)}")


@router.get("/drift", response_model=DriftMonitoringResponse)
async def get_drift_monitoring():
    """Returns scientific feature distribution shift and PSI monitoring signals with non-causal separation notice."""
    try:
        return observability_engine.get_drift_monitoring()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch drift monitoring: {str(e)}")
