"""
Day 24 Prediction Service.
Master end-to-end forecasting pipeline coordinating governance, routing, safety, provenance, and audit logging.
"""

from pathlib import Path
import uuid
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
        Executes an end-to-end governed forecast request.
        """
        request_id = f"REQ-{uuid.uuid4().hex[:12].upper()}"
        year_val = forecast_year or 2018

        features = {}
        if yield_lag_1 is not None:
            features["yield_lag_1"] = yield_lag_1
        if yield_rolling_3yr_mean is not None:
            features["yield_rolling_3yr_mean"] = yield_rolling_3yr_mean
        if area_lag_1 is not None:
            features["area_lag_1"] = area_lag_1

        # 1. Certification Guard Check
        is_allowed, status_code, reason_msg, strategy_meta = self.guard.validate_request(
            crop=crop,
            state=state,
            district=district,
            forecast_year=year_val,
            features=features,
        )

        if not is_allowed or strategy_meta is None:
            # Audit log rejected request
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

        # 2. Verify artifact hash
        _, artifact_hash = self.guard.verify_artifact_integrity(crop)

        # 3. Strategy Routing & Inference Execution
        pred_result = self.router.route_and_predict(
            crop=crop,
            state=state,
            district=district,
            forecast_year=year_val,
            strategy_meta=strategy_meta,
            features=features,
        )

        # 4. Build Cryptographic Provenance Record
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

        # Save to in-memory provenance cache
        self._provenance_cache[request_id] = provenance

        # 5. Log Success Audit Event
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

    def get_provenance_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self._provenance_cache.get(request_id)
