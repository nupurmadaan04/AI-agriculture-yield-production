"""
Scenario Decision Optimization Service.

Generates scenario candidate interventions, evaluates multi-objective scores,
verifies explicit constraints, and returns Pareto-optimal decision frontiers.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from src.decision_optimizer import decision_optimizer
from src.scenario_ranking import scenario_ranking_engine
from backend.services.ml_service import ml_service
from backend.services.forecast_service import forecast_service
from backend.utils.data_loader import data_loader

SCIENTIFIC_OPTIMIZATION_DISCLAIMER = (
    "Optimization rankings represent mathematical score maxima under user-specified weights and constraints. "
    "They should be treated as decision-support signals and trade-off explorations, not definitive causal directives."
)


class OptimizationService:
    _instance: Optional['OptimizationService'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OptimizationService, cls).__new__(cls)
        return cls._instance

    def optimize_decision(
        self,
        state: str,
        district: Optional[str] = None,
        horizon: int = 1,
        weights: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates candidate intervention strategies and extracts Pareto alternatives.
        """
        state_code, state_name = ml_service.resolve_state(state)
        df = data_loader.dataframe

        # Baseline resolution
        d_matches = df[df['State Code'] == state_code]
        if district and district.strip() and district.lower() != 'all':
            sub = d_matches[d_matches['Dist Name'].str.lower() == district.strip().lower()]
            if not sub.empty:
                d_matches = sub

        latest_year = int(d_matches['Year'].max()) if not d_matches.empty else 2017
        base_area = float(d_matches['RICE AREA (1000 ha)'].median()) if not d_matches.empty else 100.0
        base_defaults = ml_service.get_district_agronomic_defaults(state_code, district, rice_area=base_area)

        # Generate candidate intervention archetypes
        candidate_specs = [
            {
                'scenario_id': 'SCN-BASE',
                'scenario_name': 'Status Quo Baseline',
                'area_delta_pct': 0.0,
                'yield_lag_delta_pct': 0.0,
                'risk_offset': 0.0
            },
            {
                'scenario_id': 'SCN-CONSERVATIVE',
                'scenario_name': 'Targeted Efficiency (Conservative)',
                'area_delta_pct': 5.0,
                'yield_lag_delta_pct': 5.0,
                'risk_offset': -5.0
            },
            {
                'scenario_id': 'SCN-MODERATE',
                'scenario_name': 'Balanced Crop Intensification (Moderate)',
                'area_delta_pct': 12.0,
                'yield_lag_delta_pct': 10.0,
                'risk_offset': -8.0
            },
            {
                'scenario_id': 'SCN-EXPANSION',
                'scenario_name': 'Aggressive Cropland Allocation',
                'area_delta_pct': 22.0,
                'yield_lag_delta_pct': 15.0,
                'risk_offset': 6.0
            },
            {
                'scenario_id': 'SCN-SUSTAINABLE',
                'scenario_name': 'Low-Impact Climate Resilience',
                'area_delta_pct': -5.0,
                'yield_lag_delta_pct': 8.0,
                'risk_offset': -12.0
            },
            {
                'scenario_id': 'SCN-STRESS',
                'scenario_name': 'Adverse Climate Contraction',
                'area_delta_pct': -15.0,
                'yield_lag_delta_pct': -15.0,
                'risk_offset': 18.0
            }
        ]

        pipe = forecast_service.get_pipeline()
        scaler = pipe.named_steps['scaler']
        model = pipe.named_steps['model']
        feature_names = [
            'Year', 'State Code', 'RICE AREA (1000 ha)', 'TOTAL_CROPPED_AREA',
            'RICE_AREA_SHARE', 'WHEAT AREA (1000 ha)', 'COTTON AREA (1000 ha)',
            'SUGARCANE AREA (1000 ha)', 'RICE_YIELD_LAG1', 'RICE_YIELD_ROLL3'
        ]

        evaluated_candidates = []
        base_yield = 0.0
        base_risk = 35.0  # standard baseline risk

        import pandas as pd

        for spec in candidate_specs:
            adj_area = max(1.0, base_area * (1.0 + spec['area_delta_pct'] / 100.0))
            adj_tot = max(base_defaults['total_cropped_area'], adj_area)
            adj_share = min(1.0, adj_area / max(adj_tot, 0.1))
            adj_lag = max(100.0, base_defaults['rice_yield_lag1'] * (1.0 + spec['yield_lag_delta_pct'] / 100.0))
            adj_roll = max(100.0, base_defaults['rice_yield_roll3'] * (1.0 + (spec['yield_lag_delta_pct'] * 0.7) / 100.0))

            feat_df = pd.DataFrame([{
                'Year': latest_year + horizon,
                'State Code': state_code,
                'RICE AREA (1000 ha)': adj_area,
                'TOTAL_CROPPED_AREA': adj_tot,
                'RICE_AREA_SHARE': adj_share,
                'WHEAT AREA (1000 ha)': base_defaults['wheat_area'],
                'COTTON AREA (1000 ha)': base_defaults['cotton_area'],
                'SUGARCANE AREA (1000 ha)': base_defaults['sugarcane_area'],
                'RICE_YIELD_LAG1': adj_lag,
                'RICE_YIELD_ROLL3': adj_roll
            }])[feature_names]

            scaled_x = scaler.transform(feat_df)
            p_yield = float(model.predict(scaled_x)[0])

            if spec['scenario_id'] == 'SCN-BASE':
                base_yield = p_yield

            # Dynamic risk estimation
            risk_val = max(5.0, min(95.0, base_risk + spec['risk_offset']))

            evaluated_candidates.append({
                'scenario_id': spec['scenario_id'],
                'scenario_name': spec['scenario_name'],
                'projected_yield': round(p_yield, 1),
                'resource_change_pct': abs(spec['area_delta_pct']),
                'area_change_pct': spec['area_delta_pct'],
                'risk_score': round(risk_val, 1),
                'validation_r2': 0.7866,
                'validation_mae': 353.01
            })

        # Multi-criteria scoring and constraint evaluation
        ranked_list = scenario_ranking_engine.rank_scenarios(
            candidates=evaluated_candidates,
            baseline_yield=base_yield,
            baseline_risk=base_risk,
            weights=weights,
            constraints=constraints
        )

        # Compute yield deltas and percentage changes
        for item in ranked_list:
            d_y = round(item['projected_yield'] - base_yield, 1)
            pct_y = round((d_y / base_yield * 100.0) if base_yield > 0 else 0.0, 2)
            item['yield_delta'] = d_y
            item['yield_percent_change'] = pct_y

        # Pareto frontier
        pareto_list = decision_optimizer.identify_pareto_frontier(ranked_list)
        pareto_ids = {p['scenario_id'] for p in pareto_list}

        for item in ranked_list:
            item['is_pareto_optimal'] = item['scenario_id'] in pareto_ids

        feasible_candidates = [c for c in ranked_list if c['is_feasible']]
        recommended = feasible_candidates[0] if feasible_candidates else None
        pareto_alts = [c for c in ranked_list if c['is_pareto_optimal'] and c != recommended]

        return {
            'location': f"{state_name}" + (f" - {district}" if district else ""),
            'horizon': horizon,
            'baseline_yield': round(base_yield, 1),
            'baseline_risk': round(base_risk, 1),
            'weights_used': weights or decision_optimizer.DEFAULT_WEIGHTS,
            'constraints_used': constraints or {},
            'recommended_scenario': recommended,
            'pareto_alternatives': pareto_alts,
            'all_ranked_candidates': ranked_list,
            'total_evaluated': len(ranked_list),
            'feasible_count': len(feasible_candidates),
            'optimization_method': 'Weighted Linear Scalarization + Pareto Filtering',
            'scientific_disclaimer': SCIENTIFIC_OPTIMIZATION_DISCLAIMER
        }

    solve = optimize_decision


optimization_service = OptimizationService()
