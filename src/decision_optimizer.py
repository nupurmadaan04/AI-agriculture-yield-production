"""
Agricultural Decision Optimizer & Multi-Objective Pareto Framework.

Ranks and optimizes candidate agricultural intervention scenarios against user-configured
objective weights and explicit feasibility constraints (yield, risk, resource shift, reliability).
Identifies dominant Pareto alternatives without arbitrary single-winner forcing.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class DecisionOptimizer:
    """
    Evaluates scenario feasibility, computes multi-objective decision scores,
    and extracts Pareto-optimal frontier candidates.
    """

    DEFAULT_WEIGHTS = {
        'yield_improvement': 0.40,
        'risk_reduction': 0.25,
        'resource_efficiency': 0.20,
        'model_reliability': 0.15
    }

    @staticmethod
    def evaluate_constraints(
        candidate: Dict[str, Any],
        min_yield: Optional[float] = None,
        max_resource_change_pct: Optional[float] = None,
        max_risk_score: Optional[float] = None,
        min_reliability_score: Optional[float] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates whether a candidate scenario satisfies all explicit user constraints.
        """
        status: Dict[str, Any] = {}
        is_feasible = True

        y_pred = candidate.get('projected_yield', candidate.get('scenario_prediction', 0.0))
        res_change = abs(candidate.get('resource_change_pct', candidate.get('area_change_pct', 0.0)))
        risk = candidate.get('risk_score', 50.0)
        rel_r2 = candidate.get('validation_r2', 0.7866)

        # 1. Min Yield
        if min_yield is not None:
            passed = y_pred >= min_yield
            status['min_yield'] = {'target': min_yield, 'actual': y_pred, 'passed': passed}
            if not passed:
                is_feasible = False
        else:
            status['min_yield'] = {'target': None, 'actual': y_pred, 'passed': True}

        # 2. Max Resource Change
        if max_resource_change_pct is not None:
            passed = res_change <= max_resource_change_pct
            status['max_resource_change'] = {'target': max_resource_change_pct, 'actual': res_change, 'passed': passed}
            if not passed:
                is_feasible = False
        else:
            status['max_resource_change'] = {'target': None, 'actual': res_change, 'passed': True}

        # 3. Max Risk Score
        if max_risk_score is not None:
            passed = risk <= max_risk_score
            status['max_risk'] = {'target': max_risk_score, 'actual': risk, 'passed': passed}
            if not passed:
                is_feasible = False
        else:
            status['max_risk'] = {'target': None, 'actual': risk, 'passed': True}

        # 4. Min Reliability
        if min_reliability_score is not None:
            passed = rel_r2 >= min_reliability_score
            status['min_reliability'] = {'target': min_reliability_score, 'actual': rel_r2, 'passed': passed}
            if not passed:
                is_feasible = False
        else:
            status['min_reliability'] = {'target': None, 'actual': rel_r2, 'passed': True}

        return is_feasible, status

    @classmethod
    def compute_objective_score(
        cls,
        candidate: Dict[str, Any],
        baseline_yield: float,
        baseline_risk: float,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Computes the normalized 0-100 multi-objective decision score.
        """
        w = cls.DEFAULT_WEIGHTS.copy()
        if weights:
            total_w = sum(weights.values())
            if total_w > 0:
                w = {k: weights.get(k, 0.0) / total_w for k in w}

        y_pred = float(candidate.get('projected_yield', candidate.get('scenario_prediction', baseline_yield)))
        risk = float(candidate.get('risk_score', baseline_risk))
        res_shift = float(abs(candidate.get('resource_change_pct', candidate.get('area_change_pct', 0.0))))
        r2 = float(candidate.get('validation_r2', 0.7866))

        # 1. Yield Improvement Subscore (0 - 100)
        yield_diff_pct = ((y_pred - baseline_yield) / baseline_yield * 100.0) if baseline_yield > 0 else 0.0
        # Maps -20% -> 0, 0% -> 50, +20% -> 100
        yield_subscore = np.clip(50.0 + (yield_diff_pct * 2.5), 0.0, 100.0)

        # 2. Risk Reduction Subscore (0 - 100)
        # Maps risk 0 -> 100, risk 100 -> 0
        risk_subscore = np.clip(100.0 - risk, 0.0, 100.0)

        # 3. Resource Efficiency Subscore (0 - 100)
        # Low input modification = high efficiency (0% shift -> 100, 30% shift -> 25)
        res_subscore = np.clip(100.0 - (res_shift * 2.5), 0.0, 100.0)

        # 4. Model Reliability Subscore (0 - 100)
        # Baseline validation R2 is 0.7866 -> maps to ~85
        rel_subscore = np.clip(r2 * 100.0, 0.0, 100.0)

        decision_score = (
            w['yield_improvement'] * yield_subscore +
            w['risk_reduction'] * risk_subscore +
            w['resource_efficiency'] * res_subscore +
            w['model_reliability'] * rel_subscore
        )

        return round(float(decision_score), 1)

    @staticmethod
    def identify_pareto_frontier(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifies non-dominated candidate scenarios along two key tradeoff axes:
        1. Higher Yield Improvement
        2. Lower Risk / Lower Resource Shift
        """
        if not candidates:
            return []

        pareto_candidates = []
        for i, c1 in enumerate(candidates):
            dominated = False
            y1 = c1.get('projected_yield', 0.0)
            r1 = c1.get('risk_score', 100.0)
            res1 = abs(c1.get('resource_change_pct', 0.0))

            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                y2 = c2.get('projected_yield', 0.0)
                r2 = c2.get('risk_score', 100.0)
                res2 = abs(c2.get('resource_change_pct', 0.0))

                # c2 dominates c1 if c2 is at least as good in all criteria and strictly better in at least one
                if (y2 >= y1 and r2 <= r1 and res2 <= res1) and (y2 > y1 or r2 < r1 or res2 < res1):
                    dominated = True
                    break

            if not dominated:
                pareto_candidates.append(c1)

        return pareto_candidates


decision_optimizer = DecisionOptimizer()
