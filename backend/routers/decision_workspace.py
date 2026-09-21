"""
Day 32 Decision Workspace & Scenario Comparison REST Endpoints.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body

from backend.schemas.decision_workspace import (
    WorkspaceAnalyzeRequest,
    DecisionWorkspaceResponse,
    ScenarioItem
)
from backend.services.decision_workspace_service import decision_workspace_service

router = APIRouter(prefix="/api/workspace", tags=["Decision Workspace & Scenario Comparison"])


@router.post("/analyze", response_model=DecisionWorkspaceResponse)
async def analyze_workspace(body: WorkspaceAnalyzeRequest):
    """
    Executes full decision workspace synthesis: baseline forecast, historical context,
    validation evidence, uncertainty, drift monitoring, what-if scenarios, and comparison matrix.
    """
    try:
        res = decision_workspace_service.analyze_workspace(
            crop=body.crop,
            state=body.state,
            district=body.district,
            forecast_year=body.forecast_year,
            selected_scenarios=body.selected_scenarios,
            custom_modifications=body.custom_modifications
        )
        return DecisionWorkspaceResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build decision workspace: {str(e)}")


@router.get("/analyze", response_model=DecisionWorkspaceResponse)
async def analyze_workspace_get(
    crop: str = Query("Oilseeds", description="Crop commodity"),
    state: str = Query("Punjab", description="State name"),
    district: Optional[str] = Query("Ludhiana", description="District name"),
    forecast_year: int = Query(2017, description="Forecast target year")
):
    """
    Convenience GET endpoint for workspace synthesis using URL query parameters.
    """
    try:
        res = decision_workspace_service.analyze_workspace(
            crop=crop,
            state=state,
            district=district,
            forecast_year=forecast_year
        )
        return DecisionWorkspaceResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build decision workspace: {str(e)}")


@router.get("/templates")
async def get_workspace_templates():
    """
    Returns supported scenario archetypes, parameter bounds, and input features.
    """
    try:
        return decision_workspace_service.get_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch scenario templates: {str(e)}")


@router.post("/scenarios/simulate", response_model=ScenarioItem)
async def simulate_scenario(
    crop: str = Body("Oilseeds"),
    state: str = Body("Punjab"),
    district: Optional[str] = Body("Ludhiana"),
    forecast_year: int = Body(2017),
    scenario_type: str = Body("custom"),
    modifications: Optional[Dict[str, float]] = Body(None)
):
    """
    Simulates an individual custom scenario within the workspace bounds.
    """
    try:
        res = decision_workspace_service.analyze_workspace(
            crop=crop,
            state=state,
            district=district,
            forecast_year=forecast_year,
            selected_scenarios=[scenario_type],
            custom_modifications=modifications
        )
        if res.get("scenarios"):
            return ScenarioItem(**res["scenarios"][-1])
        raise HTTPException(status_code=400, detail="Scenario simulation returned empty result.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario simulation failed: {str(e)}")
