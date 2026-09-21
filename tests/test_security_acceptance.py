"""
Day 33: Security Acceptance Test Suite.

Validates:
1. Path Traversal & Directory Breakout Protection
2. Arbitrary File & Sensitive Asset Protection (.env, .git, config)
3. Injection-Style Inputs (SQL, XSS, Shell Metacharacters)
4. Oversized Payloads & Length Boundaries
5. HTTP Security Headers (X-Content-Type-Options, X-Frame-Options, Referrer-Policy, X-Request-ID)
6. Error Leakage & Traceback Suppression
7. CORS Configuration Audit
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# =============================================================================
# 1. PATH TRAVERSAL & DIRECTORY BREAKOUT
# =============================================================================

@pytest.mark.parametrize("malicious_path", [
    "../../etc/passwd",
    "..\\..\\Windows\\win.ini",
    "....//....//.env",
    "%2e%2e%2f%2e%2e%2f.env",
    "nested/../../.git/config",
])
def test_path_traversal_blocked_in_report_and_evidence(malicious_path):
    """
    Attempt path traversal via report and evidence routes.
    Must never return local system or configuration files.
    """
    # 1. Report download endpoint
    resp1 = client.get(f"/api/decision/{malicious_path}/report")
    assert resp1.status_code in (400, 404, 422)
    assert "root:" not in resp1.text
    assert "[extensions]" not in resp1.text

    # 2. Evidence crop endpoint: Must not open arbitrary files
    resp2 = client.get(f"/api/forecast/evidence/{malicious_path}")
    assert resp2.status_code in (200, 400, 404, 422)
    if resp2.status_code == 200:
        # Must be structured unsupported catalog response, never arbitrary file content
        data = resp2.json()
        assert data.get("certification_status") == "UNSUPPORTED"
    assert "root:" not in resp2.text
    assert "[extensions]" not in resp2.text


def test_arbitrary_file_access_blocked():
    """Verify private repository assets (.env, .git) cannot be accessed over API."""
    routes_to_probe = [
        "/.env",
        "/api/.env",
        "/.git/config",
        "/api/.git/HEAD",
        "/Models/oilseeds_rf_production.pkl",
        "/Datasets/processed/agricultural_panel.csv",
    ]
    for r in routes_to_probe:
        resp = client.get(r)
        assert resp.status_code in (404, 405), f"Route {r} exposed with status {resp.status_code}"


# =============================================================================
# 2. INJECTION-STYLE INPUTS (SQL, XSS, SHELL)
# =============================================================================

@pytest.mark.parametrize("injection_payload", [
    "' OR 1=1 --",
    "'; DROP TABLE panel; --",
    "1' UNION SELECT * FROM users --",
    "<script>alert('xss')</script>",
    "<img src=x onerror=alert(1)>",
    "'; cat /etc/passwd; echo '",
    "$(whoami)",
    "`id`",
    "{{ 7 * 7 }}",
])
def test_injection_inputs_handled_as_pure_data(injection_payload):
    """
    Injection attacks in crop, state, or district must be treated strictly as data literals.
    Must result in safe rejection without command execution, SQL evaluation, or HTML reflection.
    """
    payload = {
        "crop": injection_payload,
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    # Must reject safely with 400 Bad Request
    assert resp.status_code == 400
    err = resp.json()
    assert "UNSUPPORTED_CROP" in str(err)
    # Must never serve as text/html
    assert "text/html" not in resp.headers.get("content-type", "")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "root:x:" not in resp.text


# =============================================================================
# 3. OVERSIZED INPUTS & BOUNDARY PROTECTION
# =============================================================================

def test_oversized_string_inputs():
    """Sending a 10,000 character string in crop/state/district must be safely handled."""
    huge_string = "A" * 10000
    payload = {
        "crop": huge_string,
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    assert resp.status_code in (400, 422)


def test_oversized_scenario_modifications():
    """Submitting 500 modification keys in custom_modifications must be bounded."""
    huge_mods = {f"var_{i}": 1.0 for i in range(500)}
    payload = {
        "crop": "Oilseeds",
        "state": "Punjab",
        "district": "Ludhiana",
        "forecast_year": 2017,
        "custom_modifications": huge_mods
    }
    resp = client.post("/api/workspace/analyze", json=payload)
    # The application should either process safely or reject, but NEVER 500 crash
    assert resp.status_code in (200, 400, 422)


# =============================================================================
# 4. HTTP SECURITY HEADERS
# =============================================================================

def test_security_headers_present_on_all_responses():
    """Verify presence and values of mandated HTTP security headers."""
    endpoints = [
        "/health",
        "/ready",
        "/api/workspace/templates",
    ]
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code == 200
        headers = resp.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "X-Request-ID" in headers
        assert "X-Response-Time-Ms" in headers


# =============================================================================
# 5. ERROR LEAKAGE & TRACEBACK SUPPRESSION
# =============================================================================

def test_error_leakage_suppression_on_controlled_failure():
    """
    Triggering a 404 or 422 must never leak stack traces, internal paths, or secrets.
    """
    resp = client.post("/api/workspace/analyze", json={"crop": "Oilseeds", "forecast_year": "invalid_year"})
    assert resp.status_code == 422
    body = resp.text
    # Check that no Python tracebacks or local directory paths leak
    assert "Traceback (most recent call last)" not in body
    assert "C:\\Users\\" not in body
    assert "/home/" not in body
    assert "password" not in body.lower()
    assert "secret" not in body.lower()


# =============================================================================
# 6. CORS CONFIGURATION AUDIT
# =============================================================================

def test_cors_headers_handling():
    """Verify CORS preflight responds cleanly."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    }
    resp = client.options("/api/workspace/analyze", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") in ("http://localhost:3000", "*")
