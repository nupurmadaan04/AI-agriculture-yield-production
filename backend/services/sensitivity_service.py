"""
Sensitivity Analysis Service.

Coordinates feature perturbation requests and delegates to SensitivityAnalysisEngine.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.sensitivity_analysis import sensitivity_analysis_engine


class SensitivityService:
    _instance: Optional['SensitivityService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SensitivityService, cls).__new__(cls)
        return cls._instance

    def run_sensitivity_analysis(
        self,
        state: str,
        district: Optional[str] = None,
        horizon: int = 1,
        features_to_test: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes controlled feature perturbations and generates sensitivity rankings.
        """
        return sensitivity_analysis_engine.analyze_sensitivity(
            state=state,
            district=district,
            horizon=horizon,
            features_to_test=features_to_test
        )

    run = run_sensitivity_analysis
    analyze_sensitivity = run_sensitivity_analysis


sensitivity_service = SensitivityService()
