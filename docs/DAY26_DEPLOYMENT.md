# Day 26: Production Deployment & Operation Guide

## 1. Executive Summary

This guide provides the complete deployment manual for serving the **AI Agriculture Decision Intelligence Platform** in production. The platform is containerized using a multi-service architecture comprising an **Nginx Reverse Proxy & Static SPA Server** and a **FastAPI Analytical Backend**.

> [!NOTE]
> **Deployment Status Declaration**: *Containerized and validated for production-style deployment; cloud deployment architecture prepared.*

---

## 2. Architecture Overview

```
                          INTERNET / CLIENT BROWSER
                                     │
                                     ▼
                            ┌─────────────────┐
                            │      NGINX      │ (Port 80)
                            │  Reverse Proxy  │
                            │ Static SPA Host │
                            └────────┬────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
          React Static SPA                         /api/* (Reverse Proxy)
        (Client-Side Routing)                            │
                                                         ▼
                                                ┌─────────────────┐
                                                │    FastAPI      │ (Port 8000)
                                                │ Production API  │
                                                └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┼────────────────────────────────┐
                        ▼                                ▼                                ▼
                 Forecast Router                  XAI & Analytics                 Data Services
                        │
                        ▼
                Strategy Registry
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        Certified ML         Statistical
           Models             Baselines
```

---

## 3. Prerequisites

- **Docker Engine**: Version 24.0+
- **Docker Compose**: Version 2.20+
- **Host Resources**: Minimum 2 CPU cores, 4 GB RAM, 10 GB disk space
- **Network Ports**: Port `80` (HTTP web interface & API gateway) and Port `8000` (Direct API access, optional)

---

## 4. Environment Configuration

Copy the template configuration and customize as necessary:

```bash
cp .env.example .env
```

### Production Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Application operating mode (`production`, `development`, `testing`) |
| `API_HOST` | `0.0.0.0` | IP interface binding for FastAPI |
| `API_PORT` | `8000` | Port for FastAPI service |
| `CORS_ORIGINS` | `http://localhost,http://localhost:80` | Comma-separated allowed CORS web origins |
| `LOG_LEVEL` | `INFO` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MODEL_ROOT` | `Models` | Path to serialized model artifacts and registries |
| `DATA_ROOT` | `Datasets` | Path to longitudinal panels and metadata tables |
| `VITE_API_BASE_URL`| `/api` | Base API route for the frontend client |

---

## 5. Deployment Options

### Option A: Standard Production Serving via Docker Compose (Recommended)

To build and launch the multi-container stack:

```bash
# 1. Build container images
docker compose build

# 2. Start services in detached mode
docker compose up -d

# 3. Verify running containers and health status
docker compose ps
```

### Option B: Standalone Manual Production Serving

If running directly on host virtual environments:

```bash
# 1. Start FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4

# 2. Build and serve frontend
cd frontend
npm ci
npm run build
# Serve dist/ using Nginx, Caddy, or static file server
```

---

## 6. Health & Readiness Probes

### Liveness Probe (`GET /health`)
Verifies that the API process is alive and responsive.
```bash
curl -f http://localhost/health
# Response: {"status": "ok", "service": "agricultural-intelligence-api", "version": "1.0.0"}
```

### Readiness Probe (`GET /ready`)
Verifies that all required datasets, model registries, strategy registries, and analytical engines are loaded and ready to serve requests.
```bash
curl -f http://localhost/ready
# Response: {"status": "ready", "version": "1.0.0", "components": {...}}
```

---

## 7. Nginx Routing & SPA Client Fallback

The Nginx configuration (`frontend/nginx.conf`) handles two primary routing responsibilities:

1. **SPA Client-Side Routing**:
   ```nginx
   location / {
       try_files $uri $uri/ /index.html;
   }
   ```
   Ensures deep links like `/forecast`, `/modeling-readiness`, and `/decision-intelligence` load cleanly on page refresh without throwing HTTP 404 errors.

2. **API Proxying & Tracing**:
   ```nginx
   location /api/ {
       proxy_pass http://backend:8000/api/;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
       proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
       proxy_set_header X-Request-ID $http_x_request_id;
   }
   ```

---

## 8. Model Artifact Requirements

The production container automatically packages and verifies the following certified assets:

- `Models/multicrop/oilseeds/model_pipeline.pkl` (Oilseeds Random Forest Forecaster)
- `Models/multicrop/sugarcane/model_pipeline.pkl` (Sugarcane Gradient Boosting Forecaster)
- `Models/multicrop/forecast_strategy_registry.json` (Certified Strategy Registry)
- `Datasets/metadata/forecast_coverage.csv` (9,019 Geographic Coverage Mappings)
- `Datasets/processed/agricultural_panel.csv` (`AGRI_PANEL_1.0` Harmonized Dataset)

---

## 9. Security & Operational Hardening

1. **Non-Root Execution**: Backend runs under unprivileged system user `appuser` (UID 10001).
2. **Zero Secret Leakage**: No credentials, keys, or `.env` files are baked into container layers.
3. **CORS Hardening**: Explicit origin validation enforced via `CORS_ORIGINS`.
4. **Pre-Inference Rejection Guards**: Rejects unsupported inputs (`UNSUPPORTED_CROP`, `DISTRICT_UNSUPPORTED`) rather than manufacturing artificial estimates.
5. **Append-Oriented Audit Trail**: All predictions and rejections are logged with cryptographic SHA-256 provenance hashes.
