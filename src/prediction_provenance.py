"""
Day 24 Prediction Provenance & Cryptographic Lineage Builder.
Generates comprehensive provenance records answering "Where did this prediction come from?"
"""

from pathlib import Path
import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prediction_provenance")


class PredictionProvenanceBuilder:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent

    def build_provenance(
        self,
        request_id: str,
        crop: str,
        state: str,
        district: str,
        forecast_year: int,
        strategy_meta: Dict[str, Any],
        prediction_result: Dict[str, Any],
        features_used: Optional[Dict[str, Any]] = None,
        artifact_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Constructs a deterministic, verifiable prediction provenance record."""
        timestamp = datetime.now(timezone.utc).isoformat()

        provenance_payload = {
            "request_id": request_id,
            "timestamp": timestamp,
            "crop": crop,
            "state": state,
            "district": district,
            "forecast_year": forecast_year,
            "strategy": strategy_meta.get("primary_strategy", "Historical District Mean"),
            "model_name": strategy_meta.get("model_name", "Historical_District_Mean"),
            "model_version": strategy_meta.get("model_version", "v1.0"),
            "model_artifact_hash": artifact_hash or "SHA256:baseline_unhashed",
            "certification_status": strategy_meta.get("certification_status", "BASELINE_PRODUCTION"),
            "fallback_used": prediction_result.get("fallback_used", False),
            "fallback_reason": prediction_result.get("fallback_reason"),
            "features_used": features_used or {},
            "data_source": "ICRISAT / Directorate of Economics & Statistics Panel (1966-2017)",
            "validation_period": "Walk-forward expanding window (2014-2017 origins)",
            "validation_mae": strategy_meta.get("strategy_mae", 0.0),
            "baseline_mae": strategy_meta.get("baseline_mae", 0.0),
            "gain_vs_baseline_pct": strategy_meta.get("gain_vs_baseline_pct", 0.0),
            "fold_win_rate_pct": strategy_meta.get("fold_win_rate_pct", 0.0),
            "prediction": prediction_result.get("prediction", 0.0),
            "unit": prediction_result.get("unit", "kg/ha"),
            "evidence_type": prediction_result.get("evidence_type", "PREDICTED"),
            "operating_rule": strategy_meta.get("operating_rule", ""),
            "strategy_explanation": strategy_meta.get("strategy_explanation", ""),
            "validation_boundary_notice": (
                "Validation limitation: The available historical dataset ends in 2017. "
                "No independent post-2017 holdout is available. Strategy certification is based "
                "on expanding walk-forward validation over historical origins 2014–2017."
            ),
        }

        # Generate cryptographic fingerprint of provenance record
        canonical_str = json.dumps(provenance_payload, sort_keys=True)
        provenance_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        provenance_payload["provenance_hash"] = f"SHA256:{provenance_hash}"

        return provenance_payload
