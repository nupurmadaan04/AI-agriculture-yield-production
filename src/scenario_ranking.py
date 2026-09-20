"""
Scenario Multi-Criteria Ranking & Trade-off Analyzer.

Ranks evaluated candidate scenarios according to weighted multi-criteria objectives,
extracting strengths, limitations, and decision tradeoffs for each option.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.decision_optimizer import decision_optimizer


class ScenarioRankingEngine:
    """
    Ranks scenarios and generates transparent strength/weakness narratives.
    """

    @staticmethod
    def rank_scenarios(
        candidates: List[Dict[str, Any]],
        baseline_yield: float,
        baseline_risk: float,
        weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scores, validates constraints, and ranks candidate scenarios.
        """
        constraints = constraints or {}
        min_yield = constraints.get('min_yield')
        max_res = constraints.get('max_resource_change_pct')
        max_risk = constraints.get('max_risk_score')
        min_rel = constraints.get('min_reliability_score')

        scored_list = []

        for item in candidates:
            cand = item.copy()
            # Constraint check
            feasible, c_status = decision_optimizer.evaluate_constraints(
                cand,
                min_yield=min_yield,
                max_resource_change_pct=max_res,
                max_risk_score=max_risk,
                min_reliability_score=min_rel
            )
            cand['is_feasible'] = feasible
            cand['constraint_status'] = c_status

            # Objective score
            cand['decision_score'] = decision_optimizer.compute_objective_score(
                cand,
                baseline_yield=baseline_yield,
                baseline_risk=baseline_risk,
                weights=weights
            )

            # Analyze strengths and limitations
            y_pred = cand.get('projected_yield', cand.get('scenario_prediction', baseline_yield))
            risk = cand.get('risk_score', baseline_risk)
            res_shift = abs(cand.get('resource_change_pct', cand.get('area_change_pct', 0.0)))
            spread = cand.get('prediction_spread', 0.0)

            strengths = []
            limitations = []

            if y_pred > baseline_yield:
                strengths.append(f"+{round(y_pred - baseline_yield, 1)} kg/ha projected yield improvement")
            else:
                limitations.append(f"{round(baseline_yield - y_pred, 1)} kg/ha yield reduction vs baseline")

            if risk < baseline_risk:
                strengths.append(f"Risk reduced by {round(baseline_risk - risk, 1)} points")
            elif risk > baseline_risk:
                limitations.append(f"Elevated risk score (+{round(risk - baseline_risk, 1)} points)")

            if res_shift <= 5.0:
                strengths.append("High resource efficiency (minimal acreage disruption)")
            elif res_shift > 15.0:
                limitations.append(f"Significant cropland allocation shift ({res_shift}%)")

            if spread > 600.0:
                limitations.append(f"Wide ensemble prediction dispersion ({round(spread, 1)} kg/ha)")
            else:
                strengths.append("Stable ensemble prediction spread")

            cand['strengths'] = strengths
            cand['limitations'] = limitations
            cand['tradeoff_summary'] = (
                f"Balances {'high yield' if y_pred > baseline_yield else 'conservative output'} against "
                f"{'low risk' if risk <= baseline_risk else 'elevated risk'} and {res_shift}% input shift."
            )

            scored_list.append(cand)

        # Sort: feasible candidates first (descending score), then infeasible candidates
        scored_list.sort(key=lambda x: (1 if x['is_feasible'] else 0, x['decision_score']), reverse=True)

        for idx, item in enumerate(scored_list, 1):
            item['rank'] = idx

        return scored_list


scenario_ranking_engine = ScenarioRankingEngine()
