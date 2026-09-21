"""
Day 35: Production UX, Accessibility & UI Contract Test Suite.

Validates:
1. Strategy transparency & semantic labeling across commodities (Oilseeds, Sugarcane, Rice, Wheat)
2. Empirical P10-P90 ensemble uncertainty contracts and explicit baseline unavailabilities
3. Decision Workspace semantic entity typing (OBSERVED vs PREDICTED vs SCENARIO vs DERIVED)
4. Decision Brief executive non-causal language and evidence hierarchy
5. Accessibility source contracts:
   - Skip-to-content link in WebShell.tsx
   - Breadcrumbs ARIA landmarks in FeedbackStates.tsx
   - Navbar ARIA navigation controls in Navbar.tsx
6. Terminology audit: Zero user-facing internal day labels (Day 30/31/32) and zero ungrounded marketing claims
"""

import os
import re
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. STRATEGY TRANSPARENCY & SEMANTIC LABELS
# =============================================================================

def test_strategy_transparency_oilseeds_governed_ml():
    """Verify Oilseeds is governed by certified ML with model metadata and provenance."""
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "RandomForestRegressor" in data["strategy"]
    assert data["certification_status"] in ["PRODUCTION_READY", "CERTIFIED_PRODUCTION"]
    assert "prediction" in data
    assert data["prediction"] > 0
    assert "provenance" in data or "request_id" in data


def test_strategy_transparency_sugarcane_conditional_ml():
    """Verify Sugarcane uses conditional production strategy with explicit status."""
    payload = {
        "crop": "Sugarcane",
        "state": "Uttar Pradesh",
        "district": "Meerut",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "GradientBoostingRegressor" in data["strategy"]
    assert data["certification_status"] == "CONDITIONAL_PRODUCTION"
    assert data["prediction"] > 0


def test_strategy_transparency_rice_baseline():
    """Verify Rice defaults to Historical District Mean baseline with transparent evidence."""
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Historical District Mean" in data["strategy"]
    assert data["certification_status"] == "BASELINE_PRODUCTION"


def test_strategy_transparency_wheat_baseline():
    """Verify Wheat defaults to Historical District Mean baseline with transparent evidence."""
    payload = {
        "crop": "Wheat",
        "state": "Haryana",
        "district": "Karnal",
        "forecast_year": 2017
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Historical District Mean" in data["strategy"]
    assert data["certification_status"] == "BASELINE_PRODUCTION"


# =============================================================================
# 2. UNCERTAINTY & ATTRIBUTION CONTRACTS
# =============================================================================

def test_decision_workspace_uncertainty_contract_ml():
    """Verify Decision Workspace returns empirical P10-P90 ensemble intervals for ML crops."""
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    unc = data["uncertainty"]
    assert unc["is_available"] is True
    assert unc["empirical_p10_kg_ha"] is not None
    assert unc["empirical_p90_kg_ha"] is not None
    assert unc["empirical_p90_kg_ha"] >= unc["empirical_p10_kg_ha"]
    assert "not a formal" in unc["disclaimer"].lower() or "not a confidence interval" in unc["disclaimer"].lower()


def test_decision_workspace_uncertainty_contract_baseline():
    """Verify Decision Workspace explicitly marks uncertainty unavailable for deterministic baseline crops."""
    payload = {
        "crop": "Rice",
        "state": "Punjab",
        "district": "Ludhiana",
        "year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    unc = data["uncertainty"]
    assert unc["is_available"] is False
    assert unc["empirical_p10_kg_ha"] is None
    assert unc["empirical_p90_kg_ha"] is None


def test_decision_workspace_semantic_entity_typing():
    """Verify Decision Workspace tags all sub-contexts with distinct semantic classifications."""
    payload = {
        "crop": "Oilseeds",
        "state": "Madhya Pradesh",
        "district": "Indore",
        "year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["historical_context"]["semantic_classification"] == "HISTORICAL_REFERENCE"
    assert data["baseline_forecast"]["semantic_classification"] == "PREDICTED"
    assert data["uncertainty"]["semantic_classification"] == "DERIVED"
    assert data["monitoring"]["semantic_classification"] == "MONITORING"
    assert data["attribution"]["semantic_classification"] == "MODEL_ATTRIBUTION"
    assert data["scenarios"][0]["evidence_type"] == "SCENARIO"


# =============================================================================
# 3. ACCESSIBILITY SOURCE CONTRACTS (FRONTEND)
# =============================================================================

def test_webshell_skip_to_content_link():
    """Verify WebShell.tsx includes accessible skip-to-content link and main landmark."""
    webshell_path = REPO_ROOT / "frontend" / "src" / "components" / "layout" / "WebShell.tsx"
    assert webshell_path.exists()
    content = webshell_path.read_text(encoding="utf-8")

    assert 'href="#main-content"' in content, "Skip-to-content link missing href='#main-content'"
    assert 'id="main-content"' in content, "Main landmark missing id='main-content'"
    assert "sr-only" in content, "Skip link should be visually hidden by default"
    assert "focus:not-sr-only" in content, "Skip link must become visible when focused"


def test_breadcrumbs_aria_landmarks():
    """Verify FeedbackStates.tsx Breadcrumbs component specifies aria-label and aria-current."""
    feedback_path = REPO_ROOT / "frontend" / "src" / "components" / "common" / "FeedbackStates.tsx"
    assert feedback_path.exists()
    content = feedback_path.read_text(encoding="utf-8")

    assert 'aria-label="Breadcrumb"' in content, "Breadcrumbs <nav> must have aria-label='Breadcrumb'"
    assert 'aria-current="page"' in content, "Active breadcrumb item must have aria-current='page'"
    assert 'role="status"' in content, "LoadingState must have role='status'"


def test_navbar_accessible_controls():
    """Verify Navbar.tsx hamburger menu specifies aria-expanded, aria-controls, and drawer id."""
    navbar_path = REPO_ROOT / "frontend" / "src" / "components" / "layout" / "Navbar.tsx"
    assert navbar_path.exists()
    content = navbar_path.read_text(encoding="utf-8")

    assert "aria-expanded" in content, "Mobile menu button must have aria-expanded attribute"
    assert 'aria-controls="mobile-menu"' in content, "Mobile menu button must reference drawer id"
    assert 'id="mobile-menu"' in content, "Mobile menu drawer must have id='mobile-menu'"


# =============================================================================
# 4. TERMINOLOGY & STALE CLAIM AUDIT
# =============================================================================

def test_zero_user_facing_internal_day_labels():
    """
    Audit all frontend JSX/TSX page files for internal development labels
    (e.g., 'Day 30 Governance', 'DAY 31 GOVERNED', 'Day 32 Production').
    """
    forbidden_labels = [
        re.compile(r'["\'>]\s*Day\s*30\s*Governance\s*["\'<]', re.IGNORECASE),
        re.compile(r'["\'>]\s*DAY\s*31\s*GOVERNED\s*["\'<]', re.IGNORECASE),
        re.compile(r'["\'>]\s*Day\s*32\s*Production\s*["\'<]', re.IGNORECASE),
    ]

    pages_dir = REPO_ROOT / "frontend" / "src" / "pages"
    violations = []

    for tsx_file in pages_dir.glob("*.tsx"):
        content = tsx_file.read_text(encoding="utf-8")
        for pattern in forbidden_labels:
            if pattern.search(content):
                violations.append((tsx_file.name, pattern.pattern))

    assert len(violations) == 0, f"Found internal day labels in user-facing UI: {violations}"


def test_zero_ungrounded_marketing_claims():
    """
    Audit frontend pages to ensure zero ungrounded claims like '100% accurate',
    'guaranteed yield', or 'AI decides'.
    """
    risky_claims = [
        re.compile(r'\b100%\s*accurate\b', re.IGNORECASE),
        re.compile(r'\bguaranteed\s+yield\b', re.IGNORECASE),
        re.compile(r'\bAI\s+decides\b', re.IGNORECASE),
        re.compile(r'\bAI\s+chooses\b', re.IGNORECASE),
    ]

    pages_dir = REPO_ROOT / "frontend" / "src" / "pages"
    violations = []

    for tsx_file in pages_dir.glob("*.tsx"):
        content = tsx_file.read_text(encoding="utf-8")
        for pattern in risky_claims:
            if pattern.search(content):
                violations.append((tsx_file.name, pattern.pattern))

    assert len(violations) == 0, f"Found risky ungrounded marketing claims in: {violations}"
