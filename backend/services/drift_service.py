"""
Feature Drift & Distribution Shift Service.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.model_drift import model_drift_engine

class DriftService:
    _instance: Optional['DriftService'] = None
    _cached_drift: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DriftService, cls).__new__(cls)
        return cls._instance

    def get_drift_overview(self) -> Dict[str, Any]:
        if self._cached_drift is None:
            self._cached_drift = model_drift_engine.detect_drift()
        return self._cached_drift

    def get_drift_features(self) -> List[Dict[str, Any]]:
        overview = self.get_drift_overview()
        return overview.get('features', [])

drift_service = DriftService()
