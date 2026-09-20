"""
Warning Backtest Service.

Coordinates historical backtesting of early warning rules across all panel districts.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from backend.utils.data_loader import data_loader
from src.warning_backtest import warning_backtester


class WarningBacktestService:
    _instance: Optional['WarningBacktestService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WarningBacktestService, cls).__new__(cls)
        return cls._instance

    def run_backtest(
        self,
        yield_drop_threshold_pct: float = -10.0,
        warning_zscore_threshold: float = 1.2,
        lead_time_years: int = 1
    ) -> Dict[str, Any]:
        """
        Executes chronological backtest on ICRISAT panel data.
        """
        df = data_loader.dataframe
        return warning_backtester.run_backtest(
            df=df,
            yield_drop_threshold_pct=yield_drop_threshold_pct,
            warning_zscore_threshold=warning_zscore_threshold,
            lead_time_years=lead_time_years
        )


warning_backtest_service = WarningBacktestService()
