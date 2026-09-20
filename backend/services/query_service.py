"""
Natural Language Query Service & Intent Classifier.

Parses user queries into structured agricultural query intents and extracts
associated entities (state, district, year range, yield filters).
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple

class QueryService:
    _instance: Optional['QueryService'] = None

    # Known States in ICRISAT dataset
    STATES = [
        'Andhra Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Gujarat',
        'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala',
        'Madhya Pradesh', 'Maharashtra', 'Orissa', 'Punjab', 'Rajasthan',
        'Tamil Nadu', 'Telangana', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryService, cls).__new__(cls)
        return cls._instance

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extracts states, years, numbers, and comparison pairs from text."""
        t_lower = text.lower()
        matched_states = []
        for s in self.STATES:
            if s.lower() in t_lower:
                matched_states.append(s)

        # Extract Years (2010 to 2017)
        years = [int(y) for y in re.findall(r'\b(201[0-7])\b', text)]

        # Extract yield or area thresholds (e.g., above 3000, > 2500)
        yield_min_match = re.search(r'(?:above|over|greater than|>)\s*(\d{3,5})', t_lower)
        yield_min = float(yield_min_match.group(1)) if yield_min_match else None

        yield_max_match = re.search(r'(?:below|under|less than|<)\s*(\d{3,5})', t_lower)
        yield_max = float(yield_max_match.group(1)) if yield_max_match else None

        return {
            'states': matched_states,
            'primary_state': matched_states[0] if matched_states else None,
            'secondary_state': matched_states[1] if len(matched_states) > 1 else None,
            'years': years,
            'year_from': min(years) if years else None,
            'year_to': max(years) if years else None,
            'yield_min': yield_min,
            'yield_max': yield_max
        }

    def classify_intent(self, question: str) -> Dict[str, Any]:
        """
        Classifies user question into structured intent and extracts entity parameters.
        """
        q = question.strip().lower()
        entities = self.extract_entities(question)

        # Intent 1: Comparison between two states
        if len(entities['states']) >= 2 or ('compare' in q and len(entities['states']) >= 1 and 'scenario' not in q and 'simulation' not in q):
            return {
                'intent': 'state_comparison',
                'state_1': entities['states'][0] if entities['states'] else 'Punjab',
                'state_2': entities['states'][1] if len(entities['states']) > 1 else 'Haryana',
                'entities': entities
            }

        # Intent: Explainability & Decision Traceability
        if any(w in q for w in ['feature importance', 'global importance', 'most important feature', 'top features', 'which features matter most']):
            return {
                'intent': 'feature_importance',
                'entities': entities
            }

        if any(w in q for w in ['why this prediction', 'why did the model predict', 'prediction drivers', 'what drove this prediction', 'explain prediction', 'explain yield', 'why is yield']):
            return {
                'intent': 'prediction_explanation',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        if any(w in q for w in ['why this alert', 'why did we get this alert', 'why is this district under alert', 'explain alert', 'alert drivers']):
            return {
                'intent': 'alert_explanation',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        if any(w in q for w in ['model sensitivity', 'sensitivity curve', 'perturbation response', 'feature response curve']):
            return {
                'intent': 'model_sensitivity',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent: Decision Intelligence & Evidence Briefs (Day 14)
        if any(w in q for w in ['decision brief', 'decision report', 'evidence report', 'decision intelligence', 'decision synthesis', 'decision analysis', 'comprehensive brief']):
            return {
                'intent': 'decision_brief',
                'state': entities['primary_state'] or 'Punjab',
                'district': entities['primary_district'],
                'entities': entities
            }

        if any(w in q for w in ['decision options', 'available options', 'policy options', 'recommended options', 'decision alternatives']):
            return {
                'intent': 'decision_options',
                'state': entities['primary_state'] or 'Punjab',
                'district': entities['primary_district'],
                'entities': entities
            }

        if any(w in q for w in ['provenance graph', 'evidence provenance', 'trace evidence', 'where does this evidence come from', 'evidence lineage']):
            return {
                'intent': 'decision_provenance',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        if any(w in q for w in ['decision audit', 'decision certificate', 'audit certificate', 'verify decision', 'sha-256 certificate']):
            return {
                'intent': 'decision_audit',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        if any(w in q for w in ['explainability methodology', 'how does xai work', 'how are explanations calculated', 'shap vs permutation']):
            return {
                'intent': 'explainability_methodology',
                'entities': entities
            }

        # Intent: Warning Backtest & Historical Evaluation
        if any(w in q for w in ['backtest', 'historical backtest', 'lead time', 'false positive rate', 'warning accuracy', 'historical warning']):
            return {
                'intent': 'warning_backtest',
                'entities': entities
            }

        # Intent: Monitoring Health & System Status
        if any(w in q for w in ['monitoring health', 'system health', 'data freshness', 'pipeline health', 'monitoring overview']):
            return {
                'intent': 'monitoring_health',
                'entities': entities
            }

        # Intent: Change Detection & Regime Shifts
        if any(w in q for w in ['cusum', 'change detection', 'change point', 'regime shift', 'trend break', 'structural break', 'inflection']):
            return {
                'intent': 'change_detection',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent: Active Alerts & Prioritized Warnings
        if any(w in q for w in ['active alert', 'active alerts', 'under watch', 'elevated warning', 'critical alert', 'which regions are under', 'which districts are under', 'alert list']):
            return {
                'intent': 'alert_search',
                'state': entities['primary_state'],
                'entities': entities
            }

        # Intent: Temporal Dynamics & Rolling Trajectory
        if any(w in q for w in ['rolling mean', 'rolling 3yr', 'rolling 5yr', 'yoy change', 'volatility cv', 'temporal trajectory', 'temporal trend']):
            return {
                'intent': 'temporal_trend',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent: Sensitivity Analysis & Feature Elasticity
        if any(w in q for w in ['sensitivity', 'sensitive to', 'which variable', 'feature perturbation', 'elasticity']):
            return {
                'intent': 'sensitivity_analysis',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent: Scenario Decision Optimization & Trade-offs
        if any(w in q for w in ['optimal', 'optimize', 'optimization', 'best tradeoff', 'best trade-off', 'balancing', 'resource-efficient', 'lowest risk scenario', 'highest yield scenario', 'pareto']):
            return {
                'intent': 'scenario_optimization',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent: Scenario Comparison
        if any(s in q for s in ['scenario', 'scenarios', 'strategies']) and any(w in q for w in ['compare', 'comparison', 'vs', 'versus', 'baseline and stress', 'all scenarios', 'alternative']):
            return {
                'intent': 'scenario_comparison',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent: Scenario Simulation (What-If)
        if any(w in q for w in ['scenario', 'what happens under', 'what if', 'what-if', 'simulate', 'stress scenario', 'moderate improvement', 'conservative improvement']):
            scen_type = 'custom'
            if 'stress' in q:
                scen_type = 'stress_scenario'
            elif 'moderate' in q:
                scen_type = 'moderate_improvement'
            elif 'conservative' in q:
                scen_type = 'conservative_improvement'

            return {
                'intent': 'scenario_simulation',
                'state': entities['primary_state'] or 'Punjab',
                'scenario_type': scen_type,
                'entities': entities
            }

        # Intent: Model Drift & Distribution Shift
        if any(w in q for w in ['drift', 'distribution shift', 'psi', 'population stability', 'feature shift']):
            return {
                'intent': 'model_drift',
                'entities': entities
            }

        # Intent: Data Quality & Completeness
        if any(w in q for w in ['data quality', 'completeness', 'validity', 'consistency', 'quality score', 'missing value']):
            return {
                'intent': 'data_quality',
                'entities': entities
            }

        # Intent: Error Analysis & Residuals
        if any(w in q for w in ['mae', 'residual', 'residuals', 'prediction error', 'largest error', 'error ranking', 'underpredict', 'overpredict', 'where are errors']):
            return {
                'intent': 'error_analysis',
                'state': entities['primary_state'],
                'entities': entities
            }

        # Intent: Model Validation & Comparison
        if any(w in q for w in ['validate', 'validation', 'which model', 'best model', 'model performance', 'how accurate', 'how reliable', 'r2', 'rmse', 'out of time', 'unseen year', 'unseen years']):
            return {
                'intent': 'model_validation',
                'entities': entities
            }

        # Intent: Model Registry & Versions
        if any(w in q for w in ['registry', 'model version', 'model list', 'active model', 'registered model']):
            return {
                'intent': 'model_registry',
                'entities': entities
            }

        # Intent 2: Spatial Clustering & Regional Archetypes
        if any(w in q for w in ['cluster', 'clusters', 'archetype', 'spatial cluster', 'regional cluster', 'grouping']):
            return {
                'intent': 'cluster_analysis',
                'state': entities['primary_state'],
                'entities': entities
            }

        # Intent 3: Spatial Outliers (Within-State Deviations)
        if any(w in q for w in ['spatial outlier', 'spatial outliers', 'differs from state', 'within-state outlier', 'district outlier']):
            return {
                'intent': 'geographic_outlier',
                'state': entities['primary_state'],
                'entities': entities
            }

        # Intent 4: Early Warning & Deterioration Signals
        if any(w in q for w in ['early warning', 'warning score', 'deterioration', 'distress', 'critical', 'alarm', 'warning']):
            return {
                'intent': 'early_warning',
                'state': entities['primary_state'],
                'entities': entities
            }

        # Intent 5: Forecast & Forward Outlook
        if any(w in q for w in ['forecast', 'future yield', 'projection', 'outlook', 'projected', '2018', '2019', '2020']):
            return {
                'intent': 'forecast_analysis',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent 6: Trend Significance & Slope
        if any(w in q for w in ['statistically significant', 'significance', 'mann-kendall', 'theil-sen', 'slope', 'p-value', 'p value', 'declining', 'decreasing']):
            return {
                'intent': 'trend_significance',
                'state': entities['primary_state'],
                'entities': entities
            }

        # Intent 7: Anomaly / Outlier Analysis
        if any(w in q for w in ['anomaly', 'anomalies', 'outlier', 'unusual', 'irregular', 'shock', 'weird']):
            return {
                'intent': 'anomaly_analysis',
                'state': entities['primary_state'],
                'year': entities['year_from'],
                'entities': entities
            }

        # Intent 8: Risk Analysis
        if any(w in q for w in ['risk', 'uncertainty', 'safety', 'confidence', 'vulnerable', 'threat', 'volatility']):
            return {
                'intent': 'risk_analysis',
                'state': entities['primary_state'],
                'entities': entities
            }

        # Intent 7: Model Performance / Validation / Error / Accuracy
        if any(w in q for w in ['accuracy', 'r2', 'r²', 'mae', 'rmse', 'error', 'model performance', 'leaderboard', 'validation', 'benchmark']):
            return {
                'intent': 'model_performance',
                'entities': entities
            }

        # Intent 8: Scenario / What-If simulation
        if any(w in q for w in ['scenario', 'what if', 'what-if', 'simulate', 'if area increases', 'if we change']):
            return {
                'intent': 'scenario_simulation',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent 9: Explanation / Feature contribution
        if any(w in q for w in ['why', 'contribution', 'driver', 'importance', 'feature contribution', 'explain']):
            return {
                'intent': 'prediction_explanation',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent 10: Trends / Historical Progression
        if any(w in q for w in ['trend', 'progression', 'over time', 'history', 'growth', 'years', 'trajectory']):
            return {
                'intent': 'trend_analysis',
                'state': entities['primary_state'] or 'Punjab',
                'entities': entities
            }

        # Intent 11: Ranking (Highest / Lowest yield or area)
        if any(w in q for w in ['highest', 'top', 'lowest', 'rank', 'leader', 'maximum', 'best', 'worst']):
            metric = 'production' if 'production' in q else ('area' if 'area' in q else 'yield')
            ascending = any(w in q for w in ['lowest', 'worst', 'bottom', 'minimum'])
            return {
                'intent': 'state_ranking',
                'metric': metric,
                'ascending': ascending,
                'entities': entities
            }

        # Intent 12: District search / Filtered Records
        if any(w in q for w in ['district', 'find', 'records', 'above', 'below', 'greater', 'filter']) or entities['yield_min'] or entities['yield_max']:
            return {
                'intent': 'district_search',
                'state': entities['primary_state'],
                'yield_min': entities['yield_min'],
                'yield_max': entities['yield_max'],
                'entities': entities
            }

        # Intent 13: General summary
        if any(w in q for w in ['summary', 'dataset', 'overview', 'how many', 'total records', 'coverage', 'about']):
            return {
                'intent': 'summary',
                'entities': entities
            }

        # Default: general agricultural intelligence / district search
        if entities['primary_state']:
            return {
                'intent': 'trend_analysis',
                'state': entities['primary_state'],
                'entities': entities
            }

        return {
            'intent': 'general_help',
            'entities': entities
        }

query_service = QueryService()

