"""
Day 30: Forecast Monitoring, Drift Detection & Outcome Intelligence API Endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.services.forecast_monitoring_service import forecast_monitoring_service
from backend.schemas.forecast_monitoring import (
    MonitoringSummaryResponse,
    ForecastOperationsResponse,
    PredictionDistributionResponse,
    DriftMonitoringResponse,
    OutcomeEvaluationResponse,
    ErrorDecompositionResponse,
    BiasAnalysisResponse,
    MonitoringAlertsResponse,
    MonitoringHealthResponse,
)

router = APIRouter(prefix="/api/monitoring", tags=["Forecast Monitoring & Outcome Intelligence"])


@router.get("/summary", response_model=MonitoringSummaryResponse)
async def get_monitoring_summary():
    """
    Returns executive operational summary, monitoring state, evaluated outcomes count, and active alerts.
    """
    try:
        return forecast_monitoring_service.get_monitoring_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch monitoring summary: {str(e)}")


@router.get("/operations", response_model=ForecastOperationsResponse)
async def get_forecast_operations(crop: Optional[str] = Query(None, description="Optional crop filter")):
    """
    Returns operational forecast traffic, success/rejection breakdown, and strategy invocation counts from real audit records.
    """
    try:
        return forecast_monitoring_service.get_forecast_operations(crop=crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch forecast operations: {str(e)}")


@router.get("/distributions", response_model=PredictionDistributionResponse)
async def get_prediction_distributions(crop: Optional[str] = Query(None, description="Optional crop filter")):
    """
    Returns empirical moments (mean, median, std, min, max, quantiles) of runtime generated predictions vs historical panel baseline.
    """
    try:
        return forecast_monitoring_service.get_prediction_distributions(crop=crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch prediction distributions: {str(e)}")


@router.get("/drift", response_model=DriftMonitoringResponse)
async def get_drift_metrics(crop: Optional[str] = Query(None, description="Optional crop filter")):
    """
    Returns feature drift (PSI, KS statistics), dataset coverage drift, and distribution shift indicators.
    """
    try:
        return forecast_monitoring_service.get_drift_metrics(crop=crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch drift metrics: {str(e)}")


@router.get("/outcomes", response_model=OutcomeEvaluationResponse)
async def get_outcome_evaluations(
    crop: Optional[str] = Query(None, description="Crop filter"),
    state: Optional[str] = Query(None, description="State filter"),
    district: Optional[str] = Query(None, description="District filter"),
    forecast_year: Optional[int] = Query(None, description="Target forecast year")
):
    """
    Evaluates frozen pre-season forecasts against actual observed outcomes strictly respecting temporal isolation (forecast_origin < forecast_year).
    Returns EVALUATION_UNAVAILABLE for unharvested future horizons.
    """
    try:
        return forecast_monitoring_service.get_outcome_evaluations(
            crop=crop, state=state, district=district, forecast_year=forecast_year
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch outcome evaluations: {str(e)}")


@router.get("/errors", response_model=ErrorDecompositionResponse)
async def get_error_decomposition(crop: str = Query("Oilseeds", description="Crop identifier")):
    """
    Returns stratified error metrics across temporal walk-forward origins, geographic districts, and yield regimes.
    """
    try:
        return forecast_monitoring_service.get_error_decomposition(crop=crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch error decomposition for {crop}: {str(e)}")


@router.get("/bias", response_model=BiasAnalysisResponse)
async def get_bias_diagnostics(crop: Optional[str] = Query(None, description="Optional crop filter")):
    """
    Returns directional systematic bias analysis (mean residual, normalized mean error %) across multi-crop validation folds.
    """
    try:
        return forecast_monitoring_service.get_bias_diagnostics(crop=crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch bias diagnostics: {str(e)}")


@router.get("/forecast-alerts", response_model=MonitoringAlertsResponse)
async def get_forecast_monitoring_alerts():
    """
    Returns evidence-first operational, statistical drift, and systematic bias alerts citing observed metrics and threshold rules.
    """
    try:
        return forecast_monitoring_service.get_monitoring_alerts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch monitoring alerts: {str(e)}")


@router.get("/forecast-health", response_model=MonitoringHealthResponse)
async def get_forecast_monitoring_health():
    """
    Returns health status, version, and record availability of the forecast monitoring subsystem.
    """
    try:
        return forecast_monitoring_service.get_monitoring_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch monitoring health: {str(e)}")
