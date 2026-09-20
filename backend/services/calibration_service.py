"""
Prediction Spread & Calibration Service.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.calibration import calibration_engine
from backend.services.validation_service import validation_service

class CalibrationService:
    _instance: Optional['CalibrationService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CalibrationService, cls).__new__(cls)
        return cls._instance

    def get_calibration_summary(self) -> Dict[str, Any]:
        eval_df = validation_service.get_eval_df()
        return calibration_engine.analyze_calibration(eval_df)

calibration_service = CalibrationService()
