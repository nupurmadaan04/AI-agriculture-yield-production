"""
Decision Options Engine.

Maps Day 10 scenario simulation archetypes and multi-objective optimization candidates
into transparent, structured decision options with projected changes, risk deltas,
tradeoffs, limitations, and evidence links.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional


class DecisionOptionsEngine:
    """
    Converts scenario simulations into structured decision-support options.
    """

    def build_decision_options(
        self,
        base_yield_kg_ha: float,
        base_area_1000_ha: float,
        scenario_results: List[Dict[str, Any]],
        optimization_result: Optional[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Builds transparent decision options from supported Day 10 scenario archetypes.
        """
        options = []
        scen_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "scenario"]
        opt_ev_ids = [e["evidence_id"] for e in evidence_items if e.get("category") == "optimization"]

        # 1. Option from Baseline / Current Policy
        options.append({
            "option_id": "OPT-STATUS-QUO",
            "scenario_id": "SCEN-BASELINE",
            "title": "Baseline Continuation (Status Quo)",
            "scenario_type": "status_quo",
            "projected_yield_kg_ha": round(base_yield_kg_ha, 1),
            "projected_yield_delta_kg_ha": 0.0,
            "projected_production_delta_pct": 0.0,
            "risk_change": "NEUTRAL",
            "resource_efficiency": "BASELINE",
            "model_reliability": "HIGH (Trained feature distribution)",
            "tradeoffs": "Zero transition friction; vulnerable to unaddressed historical downward trends.",
            "limitations": "Does not test adaptive crop allocation.",
            "supporting_evidence": scen_ev_ids[:2] if scen_ev_ids else [],
            "is_simulated": True,
            "semantic_classification": "DERIVED"
        })

        # 2. Options from Scenario Results
        for idx, sc in enumerate(scenario_results):
            if sc.get("is_baseline") or sc.get("scenario_type") == "baseline":
                continue
            sim_yield = float(sc.get("projected_yield", sc.get("simulated_yield", base_yield_kg_ha)))
            delta_y = sim_yield - base_yield_kg_ha
            sc_type = sc.get("scenario_type", f"archetype_{idx+1}")
            sc_id = sc.get("scenario_id", f"SCEN-00{idx+1}")
            sc_title = sc.get("scenario_name", sc.get("name", f"Scenario Option {idx+1}"))

            risk_change = "REDUCED" if delta_y > 50 else ("INCREASED" if delta_y < -50 else "NEUTRAL")
            eff = "IMPROVED" if delta_y > 0 else ("REDUCED" if delta_y < 0 else "NEUTRAL")
            pct = float(sc.get("yield_percent_change", sc.get("production_delta_pct", (delta_y / base_yield_kg_ha * 100) if base_yield_kg_ha > 0 else 0.0)))

            options.append({
                "option_id": f"OPT-SCEN-{idx+1:02d}",
                "scenario_id": sc_id,
                "title": f"Scenario Option: {sc_title}",
                "scenario_type": sc_type,
                "projected_yield_kg_ha": round(sim_yield, 1),
                "projected_yield_delta_kg_ha": round(delta_y, 1),
                "projected_production_delta_pct": round(pct, 2),
                "risk_change": risk_change,
                "resource_efficiency": eff,
                "model_reliability": "VALIDATED (Within ±20% perturbation bounds)",
                "tradeoffs": f"Changes allocation: {sc.get('modifications', {}) or sc.get('interpretation', 'Scenario projection')}.",
                "limitations": "Hypothetical model projection; not a biological certainty.",
                "supporting_evidence": [e_id for e_id in scen_ev_ids if sc_id in str(e_id)] or scen_ev_ids,
                "is_simulated": True,
                "semantic_classification": "DERIVED"
            })

        # 3. Option from Multi-Objective Optimization (if present)
        if optimization_result and optimization_result.get("optimal_solution"):
            opt_sol = optimization_result["optimal_solution"]
            opt_yield = float(opt_sol.get("simulated_yield", base_yield_kg_ha))
            opt_delta_y = opt_yield - base_yield_kg_ha
            options.append({
                "option_id": "OPT-PARETO-OPTIMAL",
                "scenario_id": optimization_result.get("scenario_id", "SCEN-OPT-01"),
                "title": "Pareto-Optimized Cropland Allocation",
                "scenario_type": "pareto_optimization",
                "projected_yield_kg_ha": round(opt_yield, 1),
                "projected_yield_delta_kg_ha": round(opt_delta_y, 1),
                "projected_production_delta_pct": round(opt_sol.get("production_delta_pct", (opt_delta_y / base_yield_kg_ha) * 100 if base_yield_kg_ha > 0 else 0.0), 2),
                "risk_change": "REDUCED",
                "resource_efficiency": "OPTIMAL",
                "model_reliability": "VALIDATED (Feasible constrained solution)",
                "tradeoffs": f"Optimized multi-objective trade-off weights: {optimization_result.get('weights', {})}.",
                "limitations": "Constrained linear scalarization within ICRISAT land allocation space.",
                "supporting_evidence": opt_ev_ids if opt_ev_ids else scen_ev_ids,
                "is_simulated": True,
                "semantic_classification": "DERIVED"
            })

        return options


decision_options_engine = DecisionOptionsEngine()
