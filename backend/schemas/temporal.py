from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# =========================================================================
# FORECAST SCHEMAS
# =========================================================================

class HorizonForecastItem(BaseModel):
    horizon_years: int
    forecast_year: int
    predicted_yield: float
    lower_bound_p10: float
    upper_bound_p90: float
    prediction_spread: float
    uncertainty_pct: float

class HistoricalSeriesPoint(BaseModel):
    year: int
    yield_val: float = Field(..., alias="yield")

class ForecastYieldRequest(BaseModel):
    state: Optional[str] = Field("Punjab", description="State Name")
    state_code: Optional[int] = Field(None, description="State Code")
    district: Optional[str] = Field(None, description="District Name")
    horizons: List[int] = Field([1, 2, 3], description="Forecast horizons in years")

class ForecastYieldResponse(BaseModel):
    state: str
    district: str
    latest_observed_year: int
    latest_observed_yield: float
    historical_series: List[Dict[str, Any]]
    forecasts: List[HorizonForecastItem]
    model_name: str
    disclaimer: str

# =========================================================================
# TREND ANALYSIS SCHEMAS
# =========================================================================

class TrendAnalyzeRequest(BaseModel):
    state: Optional[str] = Field(None, description="State Name")
    district: Optional[str] = Field(None, description="District Name")

class TrendAnalyzeResponse(BaseModel):
    state: str
    district: str
    linear_slope: float
    theil_sen_slope: float
    mann_kendall_s: float
    p_value: float
    significance: str
    direction: str
    observations: int
    first_year: int
    last_year: int
    first_yield: float
    last_yield: float
    total_change_pct: float
    yearly_series: List[Dict[str, Any]]

class StateTrendItem(BaseModel):
    state: str
    state_code: int
    district_count: int
    avg_yield: float
    linear_slope: float
    theil_sen_slope: float
    p_value: float
    significance: str
    direction: str
    total_change_pct: float

class StatesTrendsResponse(BaseModel):
    data: List[StateTrendItem]

# =========================================================================
# EARLY WARNING SCHEMAS
# =========================================================================

class EarlyWarningAssessRequest(BaseModel):
    state: Optional[str] = Field("Punjab", description="State Name")
    district: Optional[str] = Field(None, description="District Name")

class EarlyWarningAssessResponse(BaseModel):
    state: str
    district: str
    warning_score: float
    severity: str
    trend_direction: str
    trend_slope_kg_ha_yr: float
    trend_significance: str
    forecast_1yr_kg_ha: float
    forecast_change_pct: float
    latest_observed_yield: float
    prediction_spread_pct: float
    is_anomaly: bool
    anomaly_score: float
    trigger_signals: List[str]
    components: Dict[str, float]
    disclaimer: str

class EarlyWarningDashboardResponse(BaseModel):
    total_states_monitored: int
    critical_states_count: int
    high_states_count: int
    moderate_states_count: int
    low_states_count: int
    declining_states_count: int
    declining_states: List[str]
    average_forecast_spread_pct: float
    top_priority_warnings: List[Dict[str, Any]]
    state_matrix: List[Dict[str, Any]]

class StatesEarlyWarningResponse(BaseModel):
    data: List[Dict[str, Any]]
