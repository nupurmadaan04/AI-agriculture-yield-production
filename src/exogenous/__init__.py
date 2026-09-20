"""
Exogenous Data Integration & Pre-Season Feature Expansion Package (Day 22).

Provides leakage-safe, geographically aligned, temporally aligned exogenous feature
engineering and multi-tier ablation benchmarking across Indian agricultural commodities.
"""

from src.exogenous.source_registry import ExogenousSourceRegistry
from src.exogenous.weather_ingestion import WeatherIngestionEngine
from src.exogenous.geographic_alignment import GeographicAlignmentEngine
from src.exogenous.temporal_alignment import TemporalAlignmentEngine
from src.exogenous.feature_engineering import ExogenousFeatureEngineeringEngine
from src.exogenous.coverage_audit import ExogenousCoverageAuditEngine
from src.exogenous.leakage_audit import ExogenousLeakageAuditEngine

__all__ = [
    "ExogenousSourceRegistry",
    "WeatherIngestionEngine",
    "GeographicAlignmentEngine",
    "TemporalAlignmentEngine",
    "ExogenousFeatureEngineeringEngine",
    "ExogenousCoverageAuditEngine",
    "ExogenousLeakageAuditEngine",
]
