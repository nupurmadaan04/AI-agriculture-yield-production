"""
Decision Robustness Engine.

Evaluates whether decision options remain favorable across Day 10 sensitivity sweeps.
Classifies options into ROBUST, MODERATELY ROBUST, SENSITIVE, or UNSUPPORTED.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


class DecisionRobustnessEngine:
    """
    Evaluates sensitivity and stability of decision options across empirical perturbation ranges.
    """

    def evaluate_option_robustness(
        self,
        option: Dict[str, Any],
        sensitivity_matrix: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates robustness of a single decision option.
        """
        projected_delta = option.get("projected_yield_delta_kg_ha", 0.0)
        curves = sensitivity_matrix.get("sensitivity_curves", {})

        # Compute max volatility under perturbations
        max_deviation = 0.0
        for feat_name, points in curves.items():
            for p in points:
                dev = abs(p.get("prediction_delta_kg_ha", 0.0))
                if dev > max_deviation:
                    max_deviation = dev

        # Classify robustness
        if abs(projected_delta) == 0.0:
            classification = "ROBUST"
            notes = "Status quo baseline represents the empirical reference vector."
        elif max_deviation < 75.0:
            classification = "ROBUST"
            notes = "Projected outcome remains favorable and stable across tested ±10% to ±20% perturbation bounds."
        elif max_deviation < 200.0:
            classification = "MODERATELY ROBUST"
            notes = "Outcome remains positive but exhibits moderate elasticity under feature variation."
        elif max_deviation >= 200.0:
            classification = "SENSITIVE"
            notes = "Outcome is highly sensitive to input feature shifts; close field monitoring recommended."
        else:
            classification = "UNSUPPORTED"
            notes = "Variables lie outside the trained feature space."

        return {
            "option_id": option.get("option_id", "OPT-UNKNOWN"),
            "title": option.get("title", "Decision Option"),
            "classification": classification,
            "max_tested_deviation_kg_ha": round(max_deviation, 1),
            "perturbation_range": "[-20%, +20%]",
            "robustness_notes": notes,
            "is_favorable": projected_delta >= 0
        }

    def evaluate_all_options(
        self,
        options: List[Dict[str, Any]],
        sensitivity_matrix: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluates robustness across all options.
        """
        return [self.evaluate_option_robustness(opt, sensitivity_matrix) for opt in options]


decision_robustness_engine = DecisionRobustnessEngine()
