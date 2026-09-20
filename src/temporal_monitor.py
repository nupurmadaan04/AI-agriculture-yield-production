"""
Temporal Monitoring Engine.

Calculates chronological temporal dynamics across yield, acreage, production,
risk scores, and forecast deviations using rolling multi-period windows (3, 5, 8 years).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


class TemporalMonitor:
    """
    Computes rolling temporal metrics and trajectory dynamics for agricultural time-series.
    """

    @staticmethod
    def calculate_temporal_metrics(
        years: List[int],
        values: List[float],
        metric_name: str = "yield"
    ) -> Dict[str, Any]:
        """
        Calculates YoY change, rolling statistics (3, 5, 8 yr), trend slopes,
        acceleration, and baseline deviations for a chronological series.
        """
        if not years or not values or len(years) != len(values):
            return {
                "metric_name": metric_name,
                "record_count": 0,
                "latest_year": None,
                "latest_value": 0.0,
                "yoy_change_pct": 0.0,
                "yoy_change_absolute": 0.0,
                "rolling_3yr_mean": 0.0,
                "rolling_3yr_std": 0.0,
                "rolling_3yr_zscore": 0.0,
                "rolling_5yr_mean": 0.0,
                "rolling_5yr_std": 0.0,
                "rolling_5yr_zscore": 0.0,
                "rolling_8yr_mean": 0.0,
                "rolling_8yr_std": 0.0,
                "trend_slope": 0.0,
                "acceleration": 0.0,
                "volatility_cv": 0.0,
                "historical_mean": 0.0,
                "deviation_from_historical": 0.0,
                "trajectory": []
            }

        # Sort chronologically
        sorted_pairs = sorted(zip(years, values), key=lambda p: p[0])
        sorted_years = [p[0] for p in sorted_pairs]
        sorted_vals = [float(p[1]) for p in sorted_pairs]
        n = len(sorted_vals)

        latest_year = sorted_years[-1]
        latest_val = sorted_vals[-1]

        # Year-over-Year Change
        if n >= 2:
            prev_val = sorted_vals[-2]
            yoy_abs = round(latest_val - prev_val, 2)
            yoy_pct = round((yoy_abs / prev_val * 100.0) if prev_val > 0 else 0.0, 2)
        else:
            yoy_abs = 0.0
            yoy_pct = 0.0

        # Rolling Statistics Helpers
        def get_rolling_stats(k: int) -> Tuple[float, float, float]:
            if n < 2:
                return latest_val, 0.0, 0.0
            window = sorted_vals[-k:] if n >= k else sorted_vals
            w_mean = float(np.mean(window))
            w_std = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
            zscore = (latest_val - w_mean) / w_std if w_std > 1e-6 else 0.0
            return round(w_mean, 2), round(w_std, 2), round(zscore, 2)

        r3_mean, r3_std, r3_z = get_rolling_stats(3)
        r5_mean, r5_std, r5_z = get_rolling_stats(5)
        r8_mean, r8_std, r8_z = get_rolling_stats(8)

        # Historical Baseline
        hist_mean = round(float(np.mean(sorted_vals)), 2)
        hist_std = round(float(np.std(sorted_vals, ddof=1)), 2) if n > 1 else 0.0
        dev_hist = round((latest_val - hist_mean) / hist_mean * 100.0, 2) if hist_mean > 0 else 0.0

        # Trend Slope (Annual delta via Theil-Sen or linear regression)
        if n >= 3:
            x_arr = np.array(sorted_years, dtype=float)
            y_arr = np.array(sorted_vals, dtype=float)
            x_mean = np.mean(x_arr)
            y_mean = np.mean(y_arr)
            denom = np.sum((x_arr - x_mean) ** 2)
            if denom > 1e-6:
                slope = round(float(np.sum((x_arr - x_mean) * (y_arr - y_mean)) / denom), 2)
            else:
                slope = 0.0
        else:
            slope = yoy_abs

        # Acceleration (Second difference)
        if n >= 3:
            d1 = sorted_vals[-1] - sorted_vals[-2]
            d0 = sorted_vals[-2] - sorted_vals[-3]
            accel = round(d1 - d0, 2)
        else:
            accel = 0.0

        # Volatility (Coefficient of Variation)
        vol_cv = round((hist_std / hist_mean * 100.0) if hist_mean > 0 else 0.0, 2)

        # Full Trajectory with Rolling Windows
        trajectory = []
        for i in range(n):
            sub = sorted_vals[:i+1]
            sub_mean = float(np.mean(sub))
            sub_roll3 = float(np.mean(sub[-3:])) if len(sub) >= 3 else sub_mean
            trajectory.append({
                "year": sorted_years[i],
                "value": round(sorted_vals[i], 1),
                "historical_mean": round(sub_mean, 1),
                "rolling_3yr": round(sub_roll3, 1),
                "yoy_change_pct": round(((sorted_vals[i] - sorted_vals[i-1]) / sorted_vals[i-1] * 100.0) if i > 0 and sorted_vals[i-1] > 0 else 0.0, 2)
            })

        return {
            "metric_name": metric_name,
            "record_count": n,
            "latest_year": latest_year,
            "latest_value": round(latest_val, 1),
            "yoy_change_pct": yoy_pct,
            "yoy_change_absolute": yoy_abs,
            "rolling_3yr_mean": r3_mean,
            "rolling_3yr_std": r3_std,
            "rolling_3yr_zscore": r3_z,
            "rolling_5yr_mean": r5_mean,
            "rolling_5yr_std": r5_std,
            "rolling_5yr_zscore": r5_z,
            "rolling_8yr_mean": r8_mean,
            "rolling_8yr_std": r8_std,
            "trend_slope": slope,
            "acceleration": accel,
            "volatility_cv": vol_cv,
            "historical_mean": hist_mean,
            "deviation_from_historical": dev_hist,
            "trajectory": trajectory
        }


temporal_monitor = TemporalMonitor()
