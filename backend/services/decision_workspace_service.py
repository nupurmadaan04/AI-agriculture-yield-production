"""
Day 32 Decision Workspace Service.

Provides caching, audit logging, template retrieval, and workspace synthesis.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.decision_workspace import decision_workspace_engine
from src.scenario_engine import SCENARIO_ARCHETYPES, SUPPORTED_SCENARIO_FEATURES, SCENARIO_BOUNDS


class DecisionWorkspaceService:
    _instance: Optional['DecisionWorkspaceService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DecisionWorkspaceService, cls).__new__(cls)
            cls._instance._init_service()
        return cls._instance

    def _init_service(self):
        self.engine = decision_workspace_engine

    def analyze_workspace(
        self,
        crop: str = "Oilseeds",
        state: str = "Punjab",
        district: Optional[str] = "Ludhiana",
        forecast_year: int = 2017,
        selected_scenarios: Optional[List[str]] = None,
        custom_modifications: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete decision workspace analysis.
        """
        return self.engine.build_workspace(
            crop=crop,
            state=state,
            district=district,
            forecast_year=forecast_year,
            selected_scenarios=selected_scenarios,
            custom_modifications=custom_modifications
        )

    def get_templates(self) -> Dict[str, Any]:
        """
        Returns supported scenario archetypes, parameter bounds, and features.
        """
        return {
            "archetypes": [
                {
                    "id": k,
                    "name": v.get("name", k.replace("_", " ").title()),
                    "description": v.get("description", ""),
                    "deltas": v.get("deltas", {})
                }
                for k, v in SCENARIO_ARCHETYPES.items()
            ],
            "supported_features": SUPPORTED_SCENARIO_FEATURES,
            "parameter_bounds": SCENARIO_BOUNDS,
            "disclaimer": (
                "Scenarios represent hypothetical what-if parameter modifications within trained manifolds. "
                "They are non-prescriptive simulations and must not be interpreted as guaranteed outcomes."
            )
        }


decision_workspace_service = DecisionWorkspaceService()
