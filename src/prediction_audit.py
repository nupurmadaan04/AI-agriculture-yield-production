"""
Day 24 Prediction Audit Logger.
Maintains an immutable, thread-safe audit log of all forecasting inference requests and rejections.
"""

from pathlib import Path
import os
import threading
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prediction_audit")

AUDIT_COLUMNS = [
    "request_id",
    "timestamp",
    "crop",
    "state",
    "district",
    "forecast_year",
    "strategy",
    "certification_status",
    "status",
    "prediction",
    "unit",
    "fallback_used",
    "provenance_hash",
    "error_code",
    "error_message",
]


class PredictionAuditLogger:
    _lock = threading.Lock()

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.meta_dir = self.base_dir / "Datasets" / "metadata"
        self.audit_csv = self.meta_dir / "prediction_audit_log.csv"
        self._ensure_audit_file()

    def _ensure_audit_file(self):
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        if not self.audit_csv.exists():
            df = pd.DataFrame(columns=AUDIT_COLUMNS)
            df.to_csv(self.audit_csv, index=False)

    def log_event(
        self,
        request_id: str,
        crop: str,
        state: str,
        district: str,
        forecast_year: Optional[int],
        strategy: str,
        certification_status: str,
        status: str,
        prediction: Optional[float] = None,
        unit: str = "kg/ha",
        fallback_used: bool = False,
        provenance_hash: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Appends a new audit record to the CSV file."""
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "request_id": request_id,
            "timestamp": timestamp,
            "crop": crop,
            "state": state,
            "district": district,
            "forecast_year": forecast_year or 2018,
            "strategy": strategy,
            "certification_status": certification_status,
            "status": status,
            "prediction": prediction if prediction is not None else "",
            "unit": unit,
            "fallback_used": fallback_used,
            "provenance_hash": provenance_hash or "",
            "error_code": error_code or "",
            "error_message": error_message or "",
        }

        with self._lock:
            df_row = pd.DataFrame([record])
            df_row.to_csv(self.audit_csv, mode="a", header=not self.audit_csv.exists() or os.stat(self.audit_csv).st_size == 0, index=False)

        logger.debug("Logged audit event %s (%s)", request_id, status)
        return record

    def get_recent_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent audit log events sorted in reverse chronological order."""
        if not self.audit_csv.exists() or os.stat(self.audit_csv).st_size == 0:
            return []

        try:
            with self._lock:
                df = pd.read_csv(self.audit_csv)
            if df.empty:
                return []
            df = df.fillna("")
            return df.tail(limit).iloc[::-1].to_dict(orient="records")
        except Exception as e:
            logger.error("Error reading audit logs: %s", e)
            return []
