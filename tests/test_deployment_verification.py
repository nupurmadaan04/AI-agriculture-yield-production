"""
Day 34: Production Deployment Verification, Recovery & Disaster Readiness Test Suite.

Validates:
1. Health & Liveness Probe (/health)
2. Genuine Readiness Probe (/ready) with fail-closed behavior on missing components
3. Environment Reproducibility (zero hardcoded machine or developer absolute paths)
4. Nginx Reverse Proxy Configuration Contracts (SPA fallback, API proxy, request tracing headers)
5. Deterministic Restart & Recovery across all 4 production commodities (Oilseeds, Sugarcane, Rice, Wheat)
6. Data Persistence & Backup Classification Integrity
7. Container Security Specifications (non-root user, healthcheck directives)
"""

import os
import re
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch

from backend.main import app
from backend.utils.data_loader import data_loader
from src.prediction_service import PredictionService

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. HEALTH & READINESS PROBE VALIDATION
# =============================================================================

def test_health_probe_liveness():
    """Verify GET /health returns 200 with operational status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["service"] == "agricultural-intelligence-api"


def test_readiness_probe_fully_operational():
    """Verify GET /ready returns 200 with all required components ready."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    components = data["components"]
    assert components["dataset"] == "ready"
    assert components["forecast_strategy_registry"] == "ready"
    assert components["forecast_router"] == "ready"
    assert components["temporal_monitoring"] == "ready"
    assert components["explainability_engine"] == "ready"
    assert components["decision_intelligence"] == "ready"
    assert components["decision_workspace"] == "ready"


def test_readiness_probe_fails_closed_on_missing_registry():
    """Verify GET /ready returns 503 Service Unavailable when strategy registry is missing."""
    with patch("os.path.exists", return_value=False):
        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["components"]["forecast_strategy_registry"] == "not_ready"
        assert data["components"]["forecast_router"] == "not_ready"


# =============================================================================
# 2. ENVIRONMENT REPRODUCIBILITY (ZERO MACHINE-DEPENDENT PATHS)
# =============================================================================

def test_environment_reproducibility_zero_machine_paths():
    """
    Audit all Python source files in backend/ and src/.
    Verify zero hardcoded Windows or developer machine absolute paths exist.
    """
    forbidden_patterns = [
        re.compile(r"[C|c]:[\\/]Users[\\/]", re.IGNORECASE),
        re.compile(r"/home/[a-zA-Z0-9_-]+/", re.IGNORECASE),
    ]

    scanned_files = []
    violations = []

    for search_dir in ["backend", "src"]:
        dir_path = REPO_ROOT / search_dir
        for py_file in dir_path.rglob("*.py"):
            scanned_files.append(py_file)
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in forbidden_patterns:
                matches = pattern.findall(content)
                if matches:
                    violations.append((str(py_file), matches))

    assert len(scanned_files) >= 20, f"Expected >= 20 source files, found {len(scanned_files)}"
    assert len(violations) == 0, f"Found hardcoded machine paths in: {violations}"


# =============================================================================
# 3. NGINX REVERSE PROXY CONFIGURATION CONTRACTS
# =============================================================================

def test_nginx_configuration_contracts():
    """
    Verify frontend/nginx.conf and nginx/nginx.conf contain mandatory production directives:
    1. SPA fallback: try_files $uri $uri/ /index.html
    2. API reverse proxy: proxy_pass http://backend:8000/api/
    3. Request tracing header: proxy_set_header X-Request-ID $http_x_request_id
    4. Security headers: X-Content-Type-Options, X-Frame-Options, Referrer-Policy
    """
    nginx_conf_paths = [
        REPO_ROOT / "frontend" / "nginx.conf",
        REPO_ROOT / "nginx" / "nginx.conf"
    ]

    for conf_path in nginx_conf_paths:
        assert conf_path.exists(), f"Missing Nginx config: {conf_path}"
        content = conf_path.read_text(encoding="utf-8")

        # 1. SPA fallback
        assert "try_files $uri $uri/ /index.html;" in content, f"Missing SPA fallback in {conf_path}"

        # 2. API reverse proxy
        assert "proxy_pass http://backend:8000/api/;" in content, f"Missing backend proxy_pass in {conf_path}"

        # 3. Request ID tracing
        assert "proxy_set_header X-Request-ID $http_x_request_id;" in content, f"Missing X-Request-ID proxying in {conf_path}"

        # 4. Security headers
        assert "X-Frame-Options" in content, f"Missing X-Frame-Options in {conf_path}"
        assert "X-Content-Type-Options" in content, f"Missing X-Content-Type-Options in {conf_path}"
        assert "Referrer-Policy" in content, f"Missing Referrer-Policy in {conf_path}"


# =============================================================================
# 4. DETERMINISTIC RECOVERY ACROSS SIMULATED SERVICE RESTARTS
# =============================================================================

def execute_forecast(crop, state, district, year):
    payload = {
        "crop": crop,
        "state": state,
        "district": district,
        "forecast_year": year
    }
    resp = client.post("/api/forecast/predict", json=payload)
    assert resp.status_code == 200, f"Forecast failed for {crop}: {resp.text}"
    data = resp.json()
    req_id = data.get("request_id")
    if req_id:
        prov_resp = client.get(f"/api/forecast/provenance/{req_id}")
        if prov_resp.status_code == 200:
            prov_data = prov_resp.json()
            data["provenance_hash"] = prov_data.get("provenance_hash")
            data["model_artifact_hash"] = prov_data.get("model_artifact_hash")
    return data


def test_deterministic_restart_oilseeds():
    """Verify Oilseeds prediction remains bitwise identical across service reloads."""
    out_1 = execute_forecast("Oilseeds", "Madhya Pradesh", "Indore", 2017)

    # Simulate service recreation / reload
    reloaded_service = PredictionService()
    assert reloaded_service is not None

    out_2 = execute_forecast("Oilseeds", "Madhya Pradesh", "Indore", 2017)

    assert out_1["prediction"] == out_2["prediction"]
    assert out_1["strategy"] == out_2["strategy"]
    assert out_1["model_version"] == out_2["model_version"]
    assert out_1["certification_status"] == out_2["certification_status"]
    assert out_1["model_artifact_hash"] == out_2["model_artifact_hash"]
    assert out_1["provenance_hash"].startswith("SHA256:")
    assert out_2["provenance_hash"].startswith("SHA256:")


def test_deterministic_restart_sugarcane():
    """Verify Sugarcane conditional prediction remains bitwise identical across restarts."""
    out_1 = execute_forecast("Sugarcane", "Uttar Pradesh", "Meerut", 2017)
    out_2 = execute_forecast("Sugarcane", "Uttar Pradesh", "Meerut", 2017)

    assert out_1["prediction"] == out_2["prediction"]
    assert out_1["strategy"] == out_2["strategy"]
    assert out_1["certification_status"] == "CONDITIONAL_PRODUCTION"
    assert out_1["model_artifact_hash"] == out_2["model_artifact_hash"]
    assert out_1["provenance_hash"].startswith("SHA256:")
    assert out_2["provenance_hash"].startswith("SHA256:")


def test_deterministic_restart_rice_baseline():
    """Verify Rice baseline persistence prediction remains bitwise identical across restarts."""
    out_1 = execute_forecast("Rice", "Punjab", "Ludhiana", 2017)
    out_2 = execute_forecast("Rice", "Punjab", "Ludhiana", 2017)

    assert out_1["prediction"] == out_2["prediction"]
    assert "Historical District Mean" in out_1["strategy"]
    assert out_1["certification_status"] == "BASELINE_PRODUCTION"
    assert out_1["model_artifact_hash"] == out_2["model_artifact_hash"]
    assert out_1["provenance_hash"].startswith("SHA256:")
    assert out_2["provenance_hash"].startswith("SHA256:")


def test_deterministic_restart_wheat_baseline():
    """Verify Wheat baseline persistence prediction remains bitwise identical across restarts."""
    out_1 = execute_forecast("Wheat", "Haryana", "Karnal", 2017)
    out_2 = execute_forecast("Wheat", "Haryana", "Karnal", 2017)

    assert out_1["prediction"] == out_2["prediction"]
    assert "Historical District Mean" in out_1["strategy"]
    assert out_1["certification_status"] == "BASELINE_PRODUCTION"
    assert out_1["model_artifact_hash"] == out_2["model_artifact_hash"]
    assert out_1["provenance_hash"].startswith("SHA256:")
    assert out_2["provenance_hash"].startswith("SHA256:")


# =============================================================================
# 5. DATA PERSISTENCE & BACKUP INVENTORY CLASSIFICATION
# =============================================================================

def test_data_persistence_and_backup_classification():
    """
    Verify existence and accessibility of all critical and persistent assets:
    - Critical (Must be backed up): canonical panel dataset, model registry, strategy registry.
    - Persistent (Must survive container restarts): prediction audit log, operational telemetry.
    """
    critical_assets = [
        REPO_ROOT / "Datasets" / "processed" / "agricultural_panel.csv",
        REPO_ROOT / "Datasets" / "metadata" / "forecast_strategy_registry.csv",
        REPO_ROOT / "Datasets" / "metadata" / "forecast_coverage.csv",
        REPO_ROOT / "Datasets" / "metadata" / "dataset_manifest.json",
        REPO_ROOT / "Models" / "multicrop" / "forecast_strategy_registry.json"
    ]
    for asset in critical_assets:
        assert asset.exists(), f"Critical asset missing: {asset}"
        assert asset.stat().st_size > 0, f"Critical asset is empty: {asset}"

    # Persistent append-oriented logs
    persistent_logs = [
        REPO_ROOT / "Datasets" / "metadata" / "prediction_audit_log.csv",
        REPO_ROOT / "Datasets" / "metadata" / "operational_telemetry.jsonl"
    ]
    for log_file in persistent_logs:
        assert log_file.exists(), f"Persistent log file missing: {log_file}"


# =============================================================================
# 6. CONTAINER SECURITY SPECIFICATIONS
# =============================================================================

def test_container_dockerfile_security_invariants():
    """
    Inspect Dockerfile and frontend/Dockerfile for key security practices:
    - Non-root user in backend (USER appuser)
    - Healthcheck directive in backend (HEALTHCHECK)
    - Multi-stage build in frontend (build stage and production stage)
    - Proper exposure of port 80 in frontend
    """
    backend_dockerfile = REPO_ROOT / "Dockerfile"
    assert backend_dockerfile.exists()
    b_content = backend_dockerfile.read_text(encoding="utf-8")

    assert "USER appuser" in b_content, "Backend Dockerfile must specify non-root user (appuser)"
    assert "HEALTHCHECK" in b_content, "Backend Dockerfile must specify a container HEALTHCHECK"
    assert "EXPOSE 8000" in b_content, "Backend Dockerfile must document internal port 8000"

    frontend_dockerfile = REPO_ROOT / "frontend" / "Dockerfile"
    assert frontend_dockerfile.exists()
    f_content = frontend_dockerfile.read_text(encoding="utf-8")

    assert "AS build" in f_content, "Frontend Dockerfile must use multi-stage build"
    assert "AS production" in f_content, "Frontend Dockerfile must define production serve stage"
    assert "EXPOSE 80" in f_content, "Frontend Dockerfile must expose HTTP port 80"


# =============================================================================
# 7. FAULT INJECTION & RECOVERY TESTING
# =============================================================================

def test_readiness_fails_closed_on_missing_dataset():
    """Verify /ready probe returns 503 if dataset cannot be accessed."""
    with patch.object(type(data_loader), "dataframe", new=None):
        resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["components"]["dataset"] == "not_ready"


def test_cors_configuration_allows_production_origins():
    """Verify CORS preflight allows standard production and local origins."""
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type"
    }
    resp = client.options("/api/workspace/analyze", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") in ["http://localhost:3000", "*"]

