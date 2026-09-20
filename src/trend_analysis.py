"""
Statistical Trend Analysis Engine.

Calculates linear slope, Theil-Sen robust median slope, and Mann-Kendall non-parametric
trend significance (p-value and direction classification) for agricultural time-series.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import stats

class TrendAnalysisEngine:
    @staticmethod
    def theil_sen_slope(years: np.ndarray, yields: np.ndarray) -> float:
        """Computes robust Theil-Sen estimator (median of all pairwise slopes)."""
        n = len(years)
        if n < 2:
            return 0.0
        slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = years[j] - years[i]
                if dx != 0:
                    slopes.append((yields[j] - yields[i]) / dx)
        return float(np.median(slopes)) if slopes else 0.0

    @staticmethod
    def mann_kendall_test(yields: np.ndarray) -> Tuple[float, float, str]:
        """
        Computes Mann-Kendall test statistic S, variance, and two-tailed p-value.
        Returns: (S, p_value, significance_label)
        """
        n = len(yields)
        if n < 3:
            return 0.0, 1.0, "INSUFFICIENT_DATA"

        s = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                s += int(np.sign(yields[j] - yields[k]))

        # Calculate variance of S
        # Under null hypothesis of no trend, Var(S) = n*(n-1)*(2n+5) / 18
        var_s = (n * (n - 1) * (2 * n + 5)) / 18.0

        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0

        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

        if p_value < 0.01:
            sig = "HIGHLY_SIGNIFICANT (p < 0.01)"
        elif p_value < 0.05:
            sig = "SIGNIFICANT (p < 0.05)"
        elif p_value < 0.10:
            sig = "MODERATELY_SIGNIFICANT (p < 0.10)"
        else:
            sig = "NOT_SIGNIFICANT (p >= 0.10)"

        return float(s), round(p_value, 4), sig

    @classmethod
    def analyze_series(cls, years: List[int], yields: List[float]) -> Dict[str, Any]:
        """
        Analyzes an empirical time-series and returns slope, significance, and direction.
        """
        y_arr = np.array(years, dtype=float)
        val_arr = np.array(yields, dtype=float)
        n = len(y_arr)

        if n < 2:
            return {
                'linear_slope': 0.0,
                'theil_sen_slope': 0.0,
                'mann_kendall_s': 0.0,
                'p_value': 1.0,
                'significance': 'INSUFFICIENT_DATA',
                'direction': 'STABLE',
                'observations': n,
                'first_year': int(years[0]) if n > 0 else 2010,
                'last_year': int(years[-1]) if n > 0 else 2017,
                'first_yield': float(yields[0]) if n > 0 else 0.0,
                'last_yield': float(yields[-1]) if n > 0 else 0.0,
                'total_change_pct': 0.0
            }

        # Linear regression
        lin_res = stats.linregress(y_arr, val_arr)
        linear_slope = float(lin_res.slope)

        # Theil-Sen slope
        ts_slope = cls.theil_sen_slope(y_arr, val_arr)

        # Mann-Kendall test
        mk_s, p_val, sig_label = cls.mann_kendall_test(val_arr)

        # Direction classification based on robust slope and p-value
        if ts_slope > 40.0 and p_val <= 0.10:
            direction = "STRONG INCREASING"
        elif ts_slope > 10.0:
            direction = "INCREASING"
        elif ts_slope < -40.0 and p_val <= 0.10:
            direction = "STRONG DECREASING"
        elif ts_slope < -10.0:
            direction = "DECREASING"
        else:
            direction = "STABLE"

        first_y = float(yields[0])
        last_y = float(yields[-1])
        tot_change_pct = ((last_y - first_y) / first_y * 100.0) if first_y > 0 else 0.0

        return {
            'linear_slope': round(linear_slope, 2),
            'theil_sen_slope': round(ts_slope, 2),
            'mann_kendall_s': round(mk_s, 1),
            'p_value': p_val,
            'significance': sig_label,
            'direction': direction,
            'observations': n,
            'first_year': int(years[0]),
            'last_year': int(years[-1]),
            'first_yield': round(first_y, 1),
            'last_yield': round(last_y, 1),
            'total_change_pct': round(tot_change_pct, 2)
        }

trend_analysis_engine = TrendAnalysisEngine()
