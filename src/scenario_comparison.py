"""
Scenario Comparison Engine.

Evaluates and contrasts multiple simulated scenario runs against the immutable
Baseline Scenario, calculating yield deltas, relative changes, ensemble spread variations,
and risk/warning shifts.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np


class ScenarioComparisonEngine:
    """
    Compares simulated scenarios against a baseline reference.
    """

    @staticmethod
    def compare_scenarios(
        baseline_result: Dict[str, Any],
        scenario_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Builds a structured comparison matrix across multiple scenario outputs.
        """
        base_yield = float(baseline_result.get('scenario_prediction', baseline_result.get('baseline_prediction', 0.0)))
        base_risk = float(baseline_result.get('risk_score', 0.0))
        base_warning = float(baseline_result.get('warning_score', 0.0))
        base_spread = float(baseline_result.get('prediction_spread', 0.0))

        comparison_items = []

        # First add baseline
        comparison_items.append({
            'scenario_id': baseline_result.get('scenario_id', 'SCN-BASE'),
            'scenario_type': 'baseline',
            'scenario_name': 'Baseline Scenario',
            'projected_yield': round(base_yield, 1),
            'yield_delta': 0.0,
            'yield_percent_change': 0.0,
            'risk_score': round(base_risk, 1),
            'risk_delta': 0.0,
            'warning_score': round(base_warning, 1),
            'warning_delta': 0.0,
            'prediction_spread': round(base_spread, 1),
            'spread_delta': 0.0,
            'is_baseline': True,
            'interpretation': 'Standard baseline projection without input modifications.'
        })

        for sc in scenario_results:
            sc_type = sc.get('scenario_type', 'custom')
            if sc_type == 'baseline':
                continue

            sc_yield = float(sc.get('scenario_prediction', base_yield))
            sc_risk = float(sc.get('risk_score', base_risk))
            sc_warning = float(sc.get('warning_score', base_warning))
            sc_spread = float(sc.get('prediction_spread', base_spread))

            delta_yield = round(sc_yield - base_yield, 1)
            pct_yield = round((delta_yield / base_yield * 100.0) if base_yield > 0 else 0.0, 2)
            delta_risk = round(sc_risk - base_risk, 1)
            delta_warning = round(sc_warning - base_warning, 1)
            delta_spread = round(sc_spread - base_spread, 1)

            interp = (
                f"Model-estimated response shows {abs(pct_yield)}% "
                f"{'expansion' if delta_yield > 0 else 'contraction'} relative to baseline."
            )

            comparison_items.append({
                'scenario_id': sc.get('scenario_id', f"SCN-{sc_type}"),
                'scenario_type': sc_type,
                'scenario_name': sc.get('scenario_name', sc_type.replace('_', ' ').title()),
                'projected_yield': round(sc_yield, 1),
                'yield_delta': delta_yield,
                'yield_percent_change': pct_yield,
                'risk_score': round(sc_risk, 1),
                'risk_delta': delta_risk,
                'warning_score': round(sc_warning, 1),
                'warning_delta': delta_warning,
                'prediction_spread': round(sc_spread, 1),
                'spread_delta': delta_spread,
                'is_baseline': False,
                'interpretation': interp
            })

        # Summary statistics
        yields = [item['projected_yield'] for item in comparison_items]
        max_yield_sc = max(comparison_items, key=lambda x: x['projected_yield'])
        min_risk_sc = min(comparison_items, key=lambda x: x['risk_score'])

        return {
            'location': baseline_result.get('location', 'National/Regional'),
            'horizon': baseline_result.get('horizon', 1),
            'scenarios_compared_count': len(comparison_items),
            'baseline_yield': round(base_yield, 1),
            'highest_yield_scenario': max_yield_sc['scenario_name'],
            'lowest_risk_scenario': min_risk_sc['scenario_name'],
            'yield_range_kg_ha': round(max(yields) - min(yields), 1),
            'comparison_matrix': comparison_items,
            'scientific_disclaimer': (
                "Comparative rankings represent model-estimated responses under hypothetical assumptions "
                "and do not constitute causal or guaranteed intervention outcomes."
            )
        }


scenario_comparison_engine = ScenarioComparisonEngine()
