"""
Unit tests for AlertSeverityClassifier (Day 12).
"""

import pytest
from src.alert_severity import alert_severity_classifier, SeverityTier, SEVERITY_WEIGHTS


def test_classify_by_score():
    assert alert_severity_classifier.classify_by_score(85.0) == SeverityTier.CRITICAL.value
    assert alert_severity_classifier.classify_by_score(70.0) == SeverityTier.HIGH.value
    assert alert_severity_classifier.classify_by_score(50.0) == SeverityTier.ELEVATED.value
    assert alert_severity_classifier.classify_by_score(30.0) == SeverityTier.WATCH.value
    assert alert_severity_classifier.classify_by_score(15.0) == SeverityTier.INFO.value


def test_classify_by_deviation():
    assert alert_severity_classifier.classify_by_deviation(yoy_pct=-30.0, z_score=-1.0) == SeverityTier.CRITICAL.value
    assert alert_severity_classifier.classify_by_deviation(yoy_pct=-18.0, z_score=-1.0) == SeverityTier.HIGH.value
    assert alert_severity_classifier.classify_by_deviation(yoy_pct=-11.0, z_score=-0.5) == SeverityTier.ELEVATED.value
    assert alert_severity_classifier.classify_by_deviation(yoy_pct=-6.0, z_score=-0.2) == SeverityTier.WATCH.value
    assert alert_severity_classifier.classify_by_deviation(yoy_pct=2.0, z_score=0.1) == SeverityTier.INFO.value


def test_tier_colors():
    assert alert_severity_classifier.get_tier_color("CRITICAL") == "red"
    assert alert_severity_classifier.get_tier_color("HIGH") == "rose"
    assert alert_severity_classifier.get_tier_color("ELEVATED") == "orange"
    assert alert_severity_classifier.get_tier_color("WATCH") == "amber"
    assert alert_severity_classifier.get_tier_color("INFO") == "blue"
