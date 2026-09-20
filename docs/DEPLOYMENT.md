# Production Deployment & Infrastructure Guide

## 1. Overview
The Agricultural Decision Intelligence Platform is built with a **FastAPI** backend and a **React (Vite + TypeScript + TailwindCSS)** frontend, operating over verified ICRISAT empirical datasets (1966–2017) and registered scikit-learn model pipelines.

---

## 2. Local Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Verified dataset in `Datasets/rice_data_outlier_removed.csv`
- Model artifacts in `Models/`

### Backend Setup
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```
3. Copy environment configuration:
   ```bash
   cp .env.example .env
   ```
4. Start the development API server:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
5. Verify health:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/ready
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Copy frontend environment configuration:
   ```bash
   cp .env.example .env
   ```
4. Start Vite development server:
   ```bash
   npm run dev
   ```
5. Open browser at `http://localhost:5173`.

---

## 3. Containerized Deployment (Docker)

### Single-command Docker Compose:
```bash
docker-compose up --build -d
```

### Verification:
- Frontend: `http://localhost`
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs` or `http://localhost:8000/redoc`
- Liveness probe: `http://localhost:8000/health`
- Readiness probe: `http://localhost:8000/ready`
- System Info: `http://localhost:8000/api/system/info`

---

## 4. Production Hardening Specifications

1. **Path Isolation**: All filesystem operations use `backend.core.paths` rooted dynamically. Absolute or machine-dependent paths are disallowed.
2. **Readiness Probe**: `/ready` evaluates dataset loading, pre-season exogenous models, anomaly models, spatial clusterers, and decision engines before traffic is routed.
3. **Structured Logging & Tracing**: Every request is tagged with an `X-Request-ID` header and timed in milliseconds (`X-Response-Time-Ms`).
4. **Error Sanitization**: Production exceptions are mapped to standard `StructuredErrorResponse` JSON with zero stack trace or path exposure.
5. **Report Isolation**: Generated HTML and Markdown evidence reports are strictly isolated in `reports/` and sanitized against directory traversal.
