"""
Day 24 Prediction Service.
Master end-to-end forecasting pipeline coordinating governance, routing, safety, provenance, and audit logging.
"""

from pathlib import Path
import uuid
import time
from typing import Dict, Any, Optional
import logging

from src.strategy_registry import StrategyRegistry
from src.certification_guard import CertificationGuard
from src.forecast_router import ForecastRouter
from src.prediction_provenance import PredictionProvenanceBuilder
from src.prediction_audit import PredictionAuditLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prediction_service")


class PredictionService:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.registry = StrategyRegistry(self.base_dir)
        self.guard = CertificationGuard(self.base_dir)
        self.router = ForecastRouter(self.base_dir)
        self.provenance_builder = PredictionProvenanceBuilder(self.base_dir)
        self.audit_logger = PredictionAuditLogger(self.base_dir)
        self._provenance_cache: Dict[str, Dict[str, Any]] = {}

    def predict_forecast(
        self,
        crop: str,
        state: str,
        district: str,
        forecast_year: Optional[int] = 2018,
        yield_lag_1: Optional[float] = None,
        yield_rolling_3yr_mean: Optional[float] = None,
        area_lag_1: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end governed forecast request with observability tracing.
        """
        t_start = time.perf_counter()
        request_id = f"REQ-{uuid.uuid4().hex[:12].upper()}"
        year_val = forecast_year or 2018
        stages = []

        # Stage 1: Input Validation
        t0 = time.perf_counter()
        features = {}
        if yield_lag_1 is not None:
            features["yield_lag_1"] = yield_lag_1
        if yield_rolling_3yr_mean is not None:
            features["yield_rolling_3yr_mean"] = yield_rolling_3yr_mean
        if area_lag_1 is not None:
            features["area_lag_1"] = area_lag_1
        val_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        stages.append({"stage_name": "INPUT_VALIDATION", "status": "COMPLETED", "duration_ms": val_ms})

        # Stage 2: Certification Guard Check
        t0 = time.perf_counter()
        is_allowed, status_code, reason_msg, strategy_meta = self.guard.validate_request(
            crop=crop,
            state=state,
            district=district,
            forecast_year=year_val,
            features=features,
        )
        cert_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        if not is_allowed or strategy_meta is None:
            stages.append({"stage_name": "CERTIFICATION_CHECK", "status": "REJECTED", "duration_ms": cert_ms, "details": {"reason": reason_msg}})
            stages.append({"stage_name": "STRATEGY_LOOKUP", "status": "SKIPPED", "duration_ms": 0.0})
            stages.append({"stage_name": "INFERENCE_EXECUTION", "status": "SKIPPED", "duration_ms": 0.0})
            stages.append({"stage_name": "PROVENANCE_GENERATION", "status": "SKIPPED", "duration_ms": 0.0})

            # Audit log rejected request
            t0 = time.perf_counter()
            self.audit_logger.log_event(
                request_id=request_id,
                crop=crop,
                state=state,
                district=district,
                forecast_year=year_val,
                strategy="NONE",
                certification_status="UNSUPPORTED",
                status="REJECTED",
                prediction=None,
                error_code=status_code,
                error_message=reason_msg,
            )
            audit_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            stages.append({"stage_name": "AUDIT_LOGGING", "status": "COMPLETED", "duration_ms": audit_ms})

            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            try:
                from src.observability_engine import observability_engine
                observability_engine.record_forecast_trace(
                    request_id=request_id,
                    crop=crop,
                    state=state,
                    district=district,
                    forecast_year=year_val,
                    strategy="NONE",
                    model_name=None,
                    model_version=None,
                    prediction=None,
                    unit="kg/ha",
                    status="REJECTED",
                    stages=stages,
                    provenance_hash=None,
                    error_code=status_code,
                    error_message=reason_msg,
                    total_duration_ms=total_ms,
                )
            except Exception:
                pass

            return {
                "status": "REJECTED",
                "request_id": request_id,
                "error_code": status_code,
                "error_message": reason_msg,
                "crop": crop,
                "state": state,
                "district": district,
                "forecast_year": year_val,
                "prediction": None,
                "unit": "kg/ha",
                "strategy": None,
                "certification_status": "UNSUPPORTED",
                "fallback_used": False,
                "model_version": None,
                "validation_scope": "walk_forward_2014_2017",
                "evidence_type": "NONE",
                "provenance": None,
            }

        stages.append({"stage_name": "CERTIFICATION_CHECK", "status": "COMPLETED", "duration_ms": cert_ms})

        # Stage 3: Strategy & Artifact Verification
        t0 = time.perf_counter()
        _, artifact_hash = self.guard.verify_artifact_integrity(crop)
        strat_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        stages.append({"stage_name": "STRATEGY_LOOKUP", "status": "COMPLETED", "duration_ms": strat_ms, "details": {"strategy": strategy_meta.get("primary_strategy")}})

        # Stage 4: Strategy Routing & Inference Execution
        t0 = time.perf_counter()
        pred_result = self.router.route_and_predict(
            crop=crop,
            state=state,
            district=district,
            forecast_year=year_val,
            strategy_meta=strategy_meta,
            features=features,
        )
        infer_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        stages.append({"stage_name": "INFERENCE_EXECUTION", "status": "COMPLETED", "duration_ms": infer_ms, "details": {"prediction": pred_result["prediction"]}})

        # Stage 5: Build Cryptographic Provenance Record
        t0 = time.perf_counter()
        provenance = self.provenance_builder.build_provenance(
            request_id=request_id,
            crop=crop,
            state=state,
            district=district,
            forecast_year=year_val,
            strategy_meta=strategy_meta,
            prediction_result=pred_result,
            features_used=features,
            artifact_hash=artifact_hash,
        )
        prov_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        stages.append({"stage_name": "PROVENANCE_GENERATION", "status": "COMPLETED", "duration_ms": prov_ms, "details": {"provenance_hash": provenance.get("provenance_hash")}})

        # Save to in-memory provenance cache
        self._provenance_cache[request_id] = provenance

        # Stage 6: Log Success Audit Event
        t0 = time.perf_counter()
        self.audit_logger.log_event(
            request_id=request_id,
            crop=crop,
            state=state,
            district=district,
            forecast_year=year_val,
            strategy=strategy_meta.get("primary_strategy", ""),
            certification_status=strategy_meta.get("certification_status", ""),
            status="SUCCESS",
            prediction=pred_result["prediction"],
            unit=pred_result["unit"],
            fallback_used=pred_result["fallback_used"],
            provenance_hash=provenance.get("provenance_hash"),
        )
        audit_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        stages.append({"stage_name": "AUDIT_LOGGING", "status": "COMPLETED", "duration_ms": audit_ms})

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        try:
            from src.observability_engine import observability_engine
            observability_engine.record_forecast_trace(
                request_id=request_id,
                crop=crop,
                state=state,
                district=district,
                forecast_year=year_val,
                strategy=strategy_meta.get("primary_strategy", ""),
                model_name=strategy_meta.get("model_name"),
                model_version=strategy_meta.get("model_version"),
                prediction=pred_result["prediction"],
                unit=pred_result["unit"],
                status="SUCCESS",
                stages=stages,
                provenance_hash=provenance.get("provenance_hash"),
                total_duration_ms=total_ms,
            )
        except Exception:
            pass

        return {
            "status": "SUCCESS",
            "request_id": request_id,
            "prediction": pred_result["prediction"],
            "unit": pred_result["unit"],
            "crop": crop,
            "state": state,
            "district": district,
            "forecast_year": year_val,
            "strategy": strategy_meta.get("primary_strategy", ""),
            "certification_status": strategy_meta.get("certification_status", ""),
            "fallback_used": pred_result["fallback_used"],
            "fallback_reason": pred_result.get("fallback_reason"),
            "model_version": strategy_meta.get("model_version", "v1.0"),
            "validation_scope": "walk_forward_2014_2017",
            "evidence_type": pred_result.get("evidence_type", "PREDICTED"),
            "operating_rule": strategy_meta.get("operating_rule", ""),
            "strategy_explanation": strategy_meta.get("strategy_explanation", ""),
            "provenance": provenance,
        }

    def predict(
        self,
        crop: str,
        state: str,
        district: str,
        forecast_year: Optional[int] = 2018,
        yield_lag_1: Optional[float] = None,
        yield_rolling_3yr_mean: Optional[float] = None,
        area_lag_1: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience alias for predict_forecast."""
        year = kwargs.get("year", forecast_year)
        return self.predict_forecast(
            crop=crop,
            state=state,
            district=district,
            forecast_year=year,
            yield_lag_1=yield_lag_1,
            yield_rolling_3yr_mean=yield_rolling_3yr_mean,
            area_lag_1=area_lag_1,
        )

    def get_provenance_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self._provenance_cache.get(request_id)


prediction_service = PredictionService()


def get_prediction_service() -> PredictionService:
    """Returns the singleton PredictionService instance."""
    return prediction_service
