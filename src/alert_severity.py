"""
Alert Severity Framework.

Defines deterministic, auditable severity thresholds across agricultural warning signals:
INFO, WATCH, ELEVATED, HIGH, CRITICAL.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from enum import Enum


class SeverityTier(str, Enum):
    INFO = "INFO"
    WATCH = "WATCH"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Severity weight multipliers for prioritization & aggregation
SEVERITY_WEIGHTS = {
    SeverityTier.INFO.value: 1.0,
    SeverityTier.WATCH.value: 2.0,
    SeverityTier.ELEVATED.value: 3.5,
    SeverityTier.HIGH.value: 5.0,
    SeverityTier.CRITICAL.value: 8.0,
}

# Explicit Deterministic Thresholds
SEVERITY_THRESHOLDS = {
    "yoy_decline_pct": {
        SeverityTier.WATCH.value: -5.0,
        SeverityTier.ELEVATED.value: -10.0,
        SeverityTier.HIGH.value: -15.0,
        SeverityTier.CRITICAL.value: -25.0,
    },
    "z_score_abs": {
        SeverityTier.WATCH.value: 1.2,
        SeverityTier.ELEVATED.value: 2.0,
        SeverityTier.HIGH.value: 2.5,
        SeverityTier.CRITICAL.value: 3.2,
    },
    "composite_score": {
        SeverityTier.INFO.value: 0.0,
        SeverityTier.WATCH.value: 25.0,
        SeverityTier.ELEVATED.value: 45.0,
        SeverityTier.HIGH.value: 65.0,
        SeverityTier.CRITICAL.value: 80.0,
    },
    "volatility_cv": {
        SeverityTier.WATCH.value: 15.0,
        SeverityTier.ELEVATED.value: 22.0,
        SeverityTier.HIGH.value: 30.0,
        SeverityTier.CRITICAL.value: 40.0,
    }
}


class AlertSeverityClassifier:
    """
    Assigns deterministic severity tiers based on verified threshold boundaries.
    """

    @staticmethod
    def classify_by_score(score: float) -> str:
        """Assigns severity based on composite 0-100 score."""
        if score >= SEVERITY_THRESHOLDS["composite_score"][SeverityTier.CRITICAL.value]:
            return SeverityTier.CRITICAL.value
        if score >= SEVERITY_THRESHOLDS["composite_score"][SeverityTier.HIGH.value]:
            return SeverityTier.HIGH.value
        if score >= SEVERITY_THRESHOLDS["composite_score"][SeverityTier.ELEVATED.value]:
            return SeverityTier.ELEVATED.value
        if score >= SEVERITY_THRESHOLDS["composite_score"][SeverityTier.WATCH.value]:
            return SeverityTier.WATCH.value
        return SeverityTier.INFO.value

    @staticmethod
    def classify_by_deviation(yoy_pct: float, z_score: float, is_persistent: bool = False) -> str:
        """
        Classifies severity using explicit magnitude, statistical deviation, and persistence.
        """
        z_abs = abs(z_score)

        if yoy_pct <= -25.0 or (z_abs >= 3.0 and is_persistent):
            return SeverityTier.CRITICAL.value
        if yoy_pct <= -15.0 or (z_abs >= 2.5) or (yoy_pct <= -10.0 and is_persistent):
            return SeverityTier.HIGH.value
        if yoy_pct <= -10.0 or z_abs >= 2.0 or (yoy_pct <= -5.0 and is_persistent):
            return SeverityTier.ELEVATED.value
        if yoy_pct <= -5.0 or z_abs >= 1.2:
            return SeverityTier.WATCH.value
        return SeverityTier.INFO.value

    @staticmethod
    def get_tier_color(tier: str) -> str:
        """Returns standard UI badge color representation."""
        colors = {
            SeverityTier.INFO.value: "blue",
            SeverityTier.WATCH.value: "amber",
            SeverityTier.ELEVATED.value: "orange",
            SeverityTier.HIGH.value: "rose",
            SeverityTier.CRITICAL.value: "red",
        }
        return colors.get(tier.upper(), "gray")


alert_severity_classifier = AlertSeverityClassifier()
