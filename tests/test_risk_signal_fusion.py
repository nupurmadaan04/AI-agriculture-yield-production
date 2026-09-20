"""
Unit tests for RiskSignalFusion engine (Day 12).
"""

import pytest
from src.risk_signal_fusion import risk_signal_fusion
from src.early_warning_engine import early_warning_engine
from src.alert_severity import SeverityTier


def test_fuse_signals_empty():
    res = risk_signal_fusion.fuse_signals(
        location="Punjab - Ludhiana",
        state="Punjab",
        district="Ludhiana",
        year=2017,
        signals=[]
    )
    assert res["severity"] == SeverityTier.INFO.value
    assert res["signal_count"] == 0
    assert "Normal Baseline" in res["dominant_signal"]


def test_fuse_signals_single_watch():
    sig = early_warning_engine.generate_signal(
        signal_id="SIG-01",
        signal_type="Year-over-Year Yield Decline",
        state="Punjab",
        district="Ludhiana",
        year=2017,
        trigger_value=-6.5,
        threshold=-5.0,
        unit="%",
        severity="WATCH",
        evidence=["Reported yield fell 6.5%."],
        recommended_action="Monitor closely."
    )
    res = risk_signal_fusion.fuse_signals(
        location="Punjab - Ludhiana",
        state="Punjab",
        district="Ludhiana",
        year=2017,
        signals=[sig]
    )
    assert res["severity"] == SeverityTier.WATCH.value
    assert res["signal_count"] == 1
    assert "Year-over-Year Yield Decline" in res["dominant_signal"]


def test_fuse_signals_critical_combination():
    sig1 = early_warning_engine.generate_signal(
        signal_id="SIG-01",
        signal_type="Persistent Multi-Year Contraction",
        state="Punjab",
        district="Ludhiana",
        year=2017,
        trigger_value=-22.0,
        threshold=-8.0,
        unit="%",
        severity="HIGH",
        evidence=["Persistent 3-year contraction."],
        recommended_action="Execute scenario analysis."
    )
    sig2 = early_warning_engine.generate_signal(
        signal_id="SIG-02",
        signal_type="Statistical Baseline Departure",
        state="Punjab",
        district="Ludhiana",
        year=2017,
        trigger_value=-3.2,
        threshold=1.5,
        unit="std dev",
        severity="CRITICAL",
        evidence=["Deviates by 3.2 std dev."],
        recommended_action="Examine regional shock."
    )
    res = risk_signal_fusion.fuse_signals(
        location="Punjab - Ludhiana",
        state="Punjab",
        district="Ludhiana",
        year=2017,
        signals=[sig1, sig2]
    )
    assert res["severity"] == SeverityTier.CRITICAL.value
    assert res["signal_count"] == 2
    assert res["evidence_strength"] in ["STRONG", "VERY_STRONG"]
    assert len(res["evidence_chain"]) >= 3
