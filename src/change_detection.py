"""
Temporal Change Detection Engine.

Detects statistical distribution shifts, trend-breaks, volatility regime shifts,
and CUSUM cumulative deviations in agricultural time-series.
Strictly distinguishes 'Change Detected' from 'Cause Identified' (no causal inference).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np


class ChangeDetectionEngine:
    """
    Identifies structural breaks, inflection points, and regime changes in agricultural series.
    """

    @staticmethod
    def detect_cusum_shift(
        values: List[float],
        target_mean: Optional[float] = None,
        threshold: float = 4.0,
        drift: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes tabular CUSUM (Cumulative Sum) quality control procedure.
        Identifies positive (S+) and negative (S-) cumulative deviations.
        """
        if not values or len(values) < 3:
            return {
                "change_detected": False,
                "change_type": "NONE",
                "max_cusum_statistic": 0.0,
                "inflection_index": None,
                "cusum_positive": [],
                "cusum_negative": [],
                "scientific_note": "Insufficient observation length for CUSUM evaluation."
            }

        arr = np.array(values, dtype=float)
        # Baseline reference: user target_mean or initial baseline period
        if target_mean is not None:
            mean_val = target_mean
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 1.0
        else:
            base_len = max(3, len(arr) // 2)
            mean_val = float(np.mean(arr[:base_len]))
            std_val = float(np.std(arr[:base_len], ddof=1)) if base_len > 1 else float(np.std(arr, ddof=1))

        if std_val < 1e-6:
            std_val = 1.0

        # Standardize
        z = (arr - mean_val) / std_val

        s_pos = np.zeros(len(z))
        s_neg = np.zeros(len(z))

        break_idx = None
        change_detected = False
        change_type = "NONE"

        for i in range(1, len(z)):
            s_pos[i] = max(0.0, s_pos[i-1] + z[i] - drift)
            s_neg[i] = max(0.0, s_neg[i-1] - z[i] - drift)

            if s_pos[i] >= threshold and not change_detected:
                change_detected = True
                change_type = "UPWARD_REGIME_SHIFT"
                break_idx = i
            elif s_neg[i] >= threshold and not change_detected:
                change_detected = True
                change_type = "DOWNWARD_REGIME_SHIFT"
                break_idx = i

        max_stat = float(round(max(np.max(s_pos), np.max(s_neg)), 2))

        return {
            "change_detected": bool(change_detected),
            "change_type": change_type,
            "max_cusum_statistic": max_stat,
            "threshold": threshold,
            "inflection_index": break_idx,
            "cusum_positive": [round(float(v), 2) for v in s_pos],
            "cusum_negative": [round(float(v), 2) for v in s_neg],
            "scientific_note": "CUSUM flags cumulative departures from statistical baseline; does not establish biological causality."
        }

    @staticmethod
    def detect_trend_break(
        years: List[int],
        values: List[float]
    ) -> Dict[str, Any]:
        """
        Identifies inflection points where linear slope shifts significantly between two segments.
        """
        if len(years) < 6 or len(years) != len(values):
            return {
                "trend_break_detected": False,
                "inflection_year": None,
                "pre_break_slope": 0.0,
                "post_break_slope": 0.0,
                "slope_delta": 0.0
            }

        n = len(values)
        best_split = None
        max_slope_diff = 0.0
        best_slopes = (0.0, 0.0)

        for split in range(3, n - 2):
            x1, y1 = np.array(years[:split]), np.array(values[:split])
            x2, y2 = np.array(years[split:]), np.array(values[split:])

            s1 = np.polyfit(x1, y1, 1)[0]
            s2 = np.polyfit(x2, y2, 1)[0]

            diff = abs(s2 - s1)
            if diff > max_slope_diff:
                max_slope_diff = diff
                best_split = split
                best_slopes = (round(float(s1), 2), round(float(s2), 2))

        break_detected = bool(max_slope_diff >= 30.0)  # Significant kg/ha/yr acceleration/deceleration

        return {
            "trend_break_detected": break_detected,
            "inflection_year": years[best_split] if best_split else None,
            "pre_break_slope": best_slopes[0],
            "post_break_slope": best_slopes[1],
            "slope_delta": round(float(max_slope_diff), 2),
            "scientific_note": "Trend breaks signify trajectory alterations in empirical records."
        }

    @classmethod
    def analyze_series_changes(
        cls,
        years: List[int],
        values: List[float],
        metric_name: str = "yield"
    ) -> Dict[str, Any]:
        """Runs comprehensive change detection battery on agricultural series."""
        cusum_res = cls.detect_cusum_shift(values)
        break_res = cls.detect_trend_break(years, values)

        # Volatility shift (first half vs second half CV)
        n = len(values)
        if n >= 6:
            mid = n // 2
            cv_pre = float(np.std(values[:mid]) / np.mean(values[:mid]) * 100.0) if np.mean(values[:mid]) > 0 else 0.0
            cv_post = float(np.std(values[mid:]) / np.mean(values[mid:]) * 100.0) if np.mean(values[mid:]) > 0 else 0.0
            vol_shift = round(cv_post - cv_pre, 2)
        else:
            vol_shift = 0.0

        return {
            "metric_name": metric_name,
            "cusum_analysis": cusum_res,
            "trend_break_analysis": break_res,
            "volatility_shift_cv_delta": vol_shift,
            "overall_change_flag": cusum_res["change_detected"] or break_res["trend_break_detected"] or abs(vol_shift) > 15.0,
            "scientific_disclaimer": "Change detection identifies statistical inflection points and must not be interpreted as causal attribution."
        }


change_detection_engine = ChangeDetectionEngine()
