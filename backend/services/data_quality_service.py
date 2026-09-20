"""
Data Quality & Integrity Service.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.data_quality_monitor import data_quality_monitor

class DataQualityService:
    _instance: Optional['DataQualityService'] = None
    _cached_audit: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataQualityService, cls).__new__(cls)
        return cls._instance

    def get_data_quality_audit(self) -> Dict[str, Any]:
        if self._cached_audit is None:
            self._cached_audit = data_quality_monitor.audit_dataset()
        return self._cached_audit

data_quality_service = DataQualityService()
