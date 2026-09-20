"""
Early Warning Historical Backtesting Engine.

Evaluates whether early warning decision rules generated at observation period t
accurately anticipated adverse model-observed outcomes at period t+1 in historical records.
Enforces strict chronological ordering to eliminate lookahead bias.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class WarningBacktester:
    """
    Backtests early warning rules chronologically across panel history.
    """

    @staticmethod
    def run_backtest(
        df: pd.DataFrame,
        yield_drop_threshold_pct: float = -10.0,
        warning_zscore_threshold: float = 1.2,
        lead_time_years: int = 1
    ) -> Dict[str, Any]:
        """
        Executes chronological step-forward evaluation across all districts and years.

        Adverse Event Definition (at t + lead_time):
            Yield(t + lead_time) - Yield(t) / Yield(t) * 100 <= yield_drop_threshold_pct

        Early Warning Trigger Definition (at t):
            Historical z-score >= warning_zscore_threshold OR YoY change <= -5%
        """
        if df.empty or 'Dist Name' not in df.columns or 'Year' not in df.columns or 'RICE YIELD (Kg per ha)' not in df.columns:
            return {
                "total_evaluations": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "alert_frequency_pct": 0.0,
                "mean_lead_time_years": lead_time_years,
                "evaluation_years_range": "N/A",
                "is_chronologically_valid": True,
                "scientific_disclaimer": "Backtest measures empirical association between warning rules and subsequent adverse yield deviations."
            }

        tp = 0
        fp = 0
        fn = 0
        tn = 0
        total_evals = 0

        min_year = int(df['Year'].min())
        max_year = int(df['Year'].max())

        # Group by district and evaluate chronologically
        for dist_name, group in df.groupby('Dist Name'):
            grp_sorted = group.sort_values('Year')
            years_list = grp_sorted['Year'].values
            yields_list = grp_sorted['RICE YIELD (Kg per ha)'].values

            n = len(years_list)
            if n < 4:
                continue

            for i in range(2, n - lead_time_years):
                hist_window = yields_list[:i+1]
                t_yield = yields_list[i]
                t_lead_yield = yields_list[i + lead_time_years]

                # Compute historical mean and std strictly using data UP TO time t
                h_mean = np.mean(hist_window)
                h_std = np.std(hist_window, ddof=1) if len(hist_window) > 1 else 1.0
                if h_std < 1e-6:
                    h_std = 1.0

                z_val = abs((t_yield - h_mean) / h_std)
                yoy_prior = (t_yield - yields_list[i-1]) / yields_list[i-1] * 100.0 if yields_list[i-1] > 0 else 0.0

                # Warning Rule Trigger at time t
                warning_triggered = (z_val >= warning_zscore_threshold) or (yoy_prior <= -5.0)

                # Actual Adverse Outcome at time t + lead_time
                actual_future_change_pct = (t_lead_yield - t_yield) / t_yield * 100.0 if t_yield > 0 else 0.0
                adverse_occurred = (actual_future_change_pct <= yield_drop_threshold_pct)

                total_evals += 1
                if warning_triggered and adverse_occurred:
                    tp += 1
                elif warning_triggered and not adverse_occurred:
                    fp += 1
                elif not warning_triggered and adverse_occurred:
                    fn += 1
                else:
                    tn += 1

        # Calculate metrics
        precision = round((tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0, 2)
        recall = round((tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0, 2)
        f1 = round((2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0, 2)
        fpr = round((fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0, 2)
        fnr = round((fn / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0, 2)
        alert_freq = round(((tp + fp) / total_evals * 100.0) if total_evals > 0 else 0.0, 2)

        return {
            "total_evaluations": total_evals,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "alert_frequency_pct": alert_freq,
            "mean_lead_time_years": lead_time_years,
            "evaluation_years_range": f"{min_year}–{max_year}",
            "parameters": {
                "yield_drop_threshold_pct": yield_drop_threshold_pct,
                "warning_zscore_threshold": warning_zscore_threshold,
                "lead_time_years": lead_time_years
            },
            "is_chronologically_valid": True,
            "scientific_disclaimer": (
                "Backtesting evaluates statistical association between historical warning triggers at period t "
                "and observed adverse outcomes at period t+lead_time. It does not establish causal predictive certainty."
            )
        }


warning_backtester = WarningBacktester()
