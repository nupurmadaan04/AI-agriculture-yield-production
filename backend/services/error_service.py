"""
Agricultural Error & Residual Intelligence Service.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.error_analysis import error_analysis_engine
from backend.services.validation_service import validation_service

class ErrorService:
    _instance: Optional['ErrorService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorService, cls).__new__(cls)
        return cls._instance

    def get_error_summary(self) -> Dict[str, Any]:
        eval_df = validation_service.get_eval_df()
        return error_analysis_engine.analyze_errors(eval_df)

    def get_state_errors(self) -> List[Dict[str, Any]]:
        summary = self.get_error_summary()
        return summary.get('state_error_rankings', [])

    def get_district_errors(self, limit: int = 20) -> List[Dict[str, Any]]:
        summary = self.get_error_summary()
        return summary.get('largest_errors', [])[:limit]

error_service = ErrorService()
