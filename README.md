# Agricultural Intelligence & Forecasting Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Tests Passing](https://img.shields.io/badge/tests-456%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidence-driven, scientifically audited agricultural decision intelligence and forecast serving platform combining longitudinal multi-crop panel data (29 verified crops, 71,601 records, 1966–2017), temporally validated machine learning forecasting, statistical persistence baselines, Tree SHAP explainability, SLSQP scenario optimization, cryptographic provenance, runtime observability, and post-outcome monitoring with Population Stability Index (PSI) drift detection.

---

## Overview

The **Agricultural Intelligence Platform** transforms decades of longitudinal district-level agricultural panel records and meteorological telemetry into structured, scientifically audited, and cryptographically verifiable **Decision Briefs**, **Model Governance Scorecards**, **Production Forecasts**, and **Post-Outcome Monitoring Intelligence**.

Unlike traditional black-box platforms that apply complex machine learning models uniformly without verification, this system implements an **evidence-first model governance framework**: machine learning models are deployed into production only when empirical temporal walk-forward validation demonstrates statistically significant superiority over simpler historical persistence baselines.

```
DATA
  ↓
GOVERNED FORECAST
  ↓
EXPLAINABILITY (Tree SHAP)
  ↓
PROVENANCE (SHA-256 Lineage)
  ↓
MONITORING
  ↓
OBSERVED OUTCOME (Post-Harvest)
  ↓
OUTCOME EVALUATION
  ↓
ERROR / BIAS / DRIFT DIAGNOSIS
  ↓
MONITORING STATUS & EVIDENCE ALERTS
```

---

## Key Capabilities

- **Standardized Multi-Crop Panel (`AGRI_PANEL_1.0`)**: 71,601 verified records across 29 crops, 20 states, and 311 districts (1966–2017).
- **Multi-Crop Pre-Season Modeling**: Zero-leakage pre-season lag feature generators evaluated across 14 major agricultural commodities.
- **Temporally Ordered Walk-Forward Validation**: 4-fold expanding walk-forward temporal evaluation (origins 2014, 2015, 2016, 2017).
- **Evidence-Based Model Certification**: Multi-crop governance matrix certifying 1 Production-Ready ML (Oilseeds), 1 Conditional ML (Sugarcane), and 12 Statistical Baseline models.
- **Explainable AI (Tree SHAP)**: Local and global feature attributions with force plots and summary distributions.
- **Pareto Decision Intelligence & Scenarios**: SLSQP optimization for multi-crop acreage allocation and input sensitivity analysis.
- **Governed Production Forecast API**: Pre-inference rejection guards (`UNSUPPORTED_CROP`, `DISTRICT_UNSUPPORTED`), 3-sigma variance clipping, and append-oriented audit logging.
- **Cryptographic Provenance**: Every prediction includes an SHA-256 lineage fingerprint answering *"Why this prediction?"*.
- **Prediction Explorer**: Filterable historical archive connecting predictions to local SHAP attributions, fallback traces, and raw JSON export.
- **Production Observability Center**: Live CPU/RSS telemetry, stage latency breakdowns ($P_{50}..P_{99}$), cryptographic dataset/model integrity checks, and FIFO event rings.
- **Forecast Monitoring & Outcome Intelligence**: Post-harvest evaluation against ground truth, Population Stability Index (PSI) drift tracking, directional signed bias diagnostics ($\text{predicted} - \text{observed}$), and evidence-first alerts.

---

## System Architecture

```
                 ┌──────────────────────────────────────┐
                 │ Authoritative Sources (ICRISAT / DES)│
                 └──────────────────┬───────────────────┘
                                    ↓
                 ┌──────────────────────────────────────┐
                 │ Data Ingestion & Quality Audit (14)  │
                 └──────────────────┬───────────────────┘
                                    ↓
                 ┌──────────────────────────────────────┐
                 │ Feature Engineering & Pre-Season Lags│
                 └──────────────────┬───────────────────┘
                                    ↓
          ┌─────────────────────────┴─────────────────────────┐
          ↓                                                   ↓
┌───────────────────────────┐                       ┌───────────────────────────┐
│   Statistical Baselines   │                       │   Machine Learning Models │
│ (District / Rolling Mean) │                       │  (Random Forest / GB)     │
└─────────┬─────────────────┘                       └───────────┬───────────────┘
          └─────────────────────────┬───────────────────────────┘
                                    ↓
                 ┌──────────────────────────────────────┐
                 │ 4-Fold Expanding Walk-Forward Valid. │
                 └──────────────────┬───────────────────┘
                                    ↓
                 ┌──────────────────────────────────────┐
                 │ Model Governance & Certification Gate│
                 └──────────────────┬───────────────────┘
                                    ↓
                 ┌──────────────────────────────────────┐
                 │ Certified Strategy Router & Guards   │
                 └──────────────────┬───────────────────┘
                                    ↓
                 ┌──────────────────────────────────────┐
                 │ Prediction Service + SHA-256 Lineage │
                 └──────────────────┬───────────────────┘
                                    ↓
          ┌─────────────────────────┴─────────────────────────┐
          ↓                                                   ↓
┌───────────────────────────┐                       ┌───────────────────────────┐
│ Observability & Telemetry │                       │ Post-Outcome Monitoring   │
│ (Latency, Memory, Traces) │                       │ (Drift PSI, Bias, Errors) │
└───────────────────────────┘                       └───────────────────────────┘
```

---

## Data Sources & Harmonization

The platform ingests and standardizes longitudinal panel data from verified authorities:

| Source Identifier | Source Authority | Records | Temporal Coverage | Role in Platform |
| :--- | :--- | :--- | :--- | :--- |
| `ICRISAT_DLD_1966_2017` | ICRISAT District Level Data | 71,601 | 1966–2017 | Primary yield, area, and production panel |
| `DES_GOI_OGD` | Directorate of Economics & Statistics | Verified | 1997–2017 | District crop verification and cross-validation |
| `IMD_GRIDDED_PRECIP` | India Meteorological Department | Gridded | 1970–2017 | Gridded rainfall exogenous ablation features |

1. **Unit Harmonization**: Standardized into `production_tonnes`, `area_ha`, and `yield_kg_ha`.
2. **Quality & Invariant Auditing**: 14 automated invariant checks (boundary clamping $[0, 150000]$, negative area removal, duplicate detection).
3. **Zero-Leakage Lag Generator**: Shift operators ($t-1, t-2$, rolling 3-year mean) computed strictly within grouped district-crop panels.

---

## Validation & Certified Governance Matrix

| Crop Commodity | Certified Strategy Status | Primary Deployed Strategy | Mean Strategy MAE | Relative Gain vs Baseline | Fold Win Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Oilseeds** | `PRODUCTION_READY` | Historical ML (`RandomForestRegressor`) | 549.67 kg/ha | **+10.85%** | 75.0% |
| **Sugarcane** | `CONDITIONAL_PRODUCTION` | Historical ML (`GradientBoostingRegressor` + Clip) | 9,469.76 kg/ha | **+5.62%** | 50.0% |
| **Chickpea** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 610.86 kg/ha | 0.00% | Baseline Preferred |
| **Kharif Sorghum**| `BASELINE_PRODUCTION` | Historical District Mean Persistence | 645.28 kg/ha | 0.00% | Baseline Preferred |
| **Minor Pulses** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 503.16 kg/ha | 0.00% | Baseline Preferred |
| **Maize** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 3,707.98 kg/ha | 0.00% | Baseline Preferred |
| **Wheat** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 696.89 kg/ha | 0.00% | Baseline Preferred |
| **Rice** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 2,625.84 kg/ha | 0.00% | Baseline Preferred |
| **Sesamum** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 0.00 kg/ha | 0.00% | Baseline Preferred |
| **Pigeonpea** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 146.33 kg/ha | 0.00% | Baseline Preferred |
| **Rapeseed & Mustard**| `BASELINE_PRODUCTION` | Historical District Mean Persistence | 0.00 kg/ha | 0.00% | Baseline Preferred |
| **Groundnut** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 436.55 kg/ha | 0.00% | Baseline Preferred |
| **Sorghum** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 532.95 kg/ha | 0.00% | Baseline Preferred |
| **Pearl Millet** | `BASELINE_PRODUCTION` | Historical District Mean Persistence | 815.38 kg/ha | 0.00% | Baseline Preferred |

---

## Reproducibility & Provenance

Dual independent inference runs across all 14 commodities produced **bitwise identical outputs**:

```
Δ = |Prediction (Run 1) - Prediction (Run 2)| = 0.00000000
```

Every prediction payload includes an SHA-256 cryptographic lineage hash. All inference requests, strategy fallbacks, and governance rejections are written to an append-oriented audit log (`Datasets/metadata/prediction_audit_log.csv`).

---

## API & Endpoints

The FastAPI backend exposes 60+ production endpoints:

### Forecast & Governance
- `POST /api/forecast/predict`: Governed forecasting inference with provenance and audit logging.
- `GET /api/forecast/strategies`: Multi-crop strategy registry and evaluation evidence.
- `GET /api/forecast/certification`: High-level governance status breakdown.
- `GET /api/forecast/coverage`: 9,019 crop-state-district geographic coverage mappings.
- `GET /api/forecast/provenance/{request_id}`: Cryptographic lineage record lookup.
- `GET /api/forecast/audit`: Append-oriented prediction audit trail events.
- `GET /api/forecast/health`: Subsystem health and governance guard status.

### Observability & Runtime Telemetry
- `GET /api/observability/summary`: Executive system telemetry, memory, and latency percentiles.
- `GET /api/observability/trace/{request_id}`: Step-level execution trace ($P_{50}..P_{99}$ latency).
- `GET /api/observability/models`: Cryptographic SHA-256 hash checks of deployed model artifacts.
- `GET /api/observability/dataset`: Cryptographic SHA-256 hash check of canonical dataset panel.

### Forecast Monitoring & Outcome Intelligence
- `GET /api/monitoring/summary`: Executive monitoring status and active alerts summary.
- `GET /api/monitoring/operations`: Audit log request volumes and crop/strategy breakdowns.
- `GET /api/monitoring/distributions`: Actual prediction moments vs historical baseline.
- `GET /api/monitoring/drift`: Population Stability Index (PSI) feature and prediction drift.
- `GET /api/monitoring/outcomes`: Strict post-harvest evaluated outcomes.
- `GET /api/monitoring/errors`: Stratified error decompositions (temporal, district, yield regime).
- `GET /api/monitoring/bias`: Directional signed bias metrics and analytical interpretations.
- `GET /api/monitoring/forecast-alerts`: Evidence-first operational alerts.
- `GET /api/monitoring/forecast-health`: System diagnostic health check.

---

## Frontend Web Application

A responsive React 18 + TypeScript + Tailwind CSS application featuring:
- **Forecast Decision Wizard (`/forecast`)**: 4-step guided request form, dynamic district filtering, validation limitation notices, Result Cards, expandable *"Why this prediction?"* explanations, and live audit logs.
- **Prediction Explorer (`/prediction-explorer`)**: Filterable forecast archive with inline local Tree SHAP attributions, fallback status indicators, and provenance inspection.
- **Forecast Monitoring (`/forecast-monitoring`)**: Post-outcome evaluation against ground truth, Population Stability Index (PSI) drift tracking, stratified error tabs, and evidence-first alerts.
- **Observability Center (`/observability`)**: Runtime latency percentiles, process RSS telemetry, model hash verification, and prediction execution traces.
- **Modeling Readiness & Robustness (`/modeling-readiness`)**: 7-tab matrix showing baseline benchmarking, walk-forward folds, error regimes, exogenous ablations, and final certification scorecards.
- **Decision Intelligence (`/decision-intelligence`)**: Executive decision briefs and interactive provenance DAGs.

---

## Project Structure

```
AI-agriculture-yield-production/
├── src/                               # Core Python modeling & governance engines
│   ├── strategy_registry.py           # Multi-crop strategy registry compiler
│   ├── certification_guard.py         # Pre-inference governance guards
│   ├── forecast_router.py             # Inference router & fallback handler
│   ├── prediction_service.py          # Master forecasting pipeline coordinator
│   ├── prediction_provenance.py       # SHA-256 cryptographic provenance builder
│   ├── prediction_audit.py            # Append-oriented audit logger
│   ├── forecast_validation.py         # Determinism benchmark runner
│   ├── multicrop_pipeline.py          # Zero-leakage pre-season lag feature pipeline
│   ├── observability_engine.py        # Host & process telemetry & trace buffers
│   └── ...
├── backend/                           # FastAPI backend server
│   ├── main.py                        # Master API router
│   ├── routers/                       # Modular REST routers (forecast, observability, monitoring)
│   ├── schemas/                       # Pydantic data contracts
│   └── services/                      # Domain query handlers & monitoring services
├── frontend/                          # React + TypeScript client application
│   ├── src/pages/ForecastMonitoring.tsx   # Day 30 Outcome Intelligence & Drift UI
│   ├── src/pages/PredictionExplorer.tsx   # Day 29 Prediction Explorer & SHAP UI
│   ├── src/pages/ObservabilityCenter.tsx  # Day 28 Production Observability UI
│   ├── src/pages/ForecastIntelligence.tsx # Production forecast serving UI
│   ├── src/pages/ModelingReadiness.tsx    # Multi-crop certification UI
│   └── src/services/api.ts            # TanStack Query API client
├── Datasets/                          # Longitudinal panels & metadata
│   ├── processed/agricultural_panel.csv # Unified AGRI_PANEL_1.0 (71,601 records)
│   └── metadata/                      # Registry, coverage, and audit logs
├── Models/                            # Certified model artifacts
│   └── multicrop/                     # Joblib pipelines & JSON registries
├── tests/                             # Pytest test suites (456 tests, 100% passing)
├── docs/                              # Comprehensive scientific & architecture documentation
│   ├── DOCUMENTATION_INDEX.md         # Master index of all 30-day documentation
│   ├── DAY30_FORECAST_MONITORING.md   # Day 30 Monitoring Architecture
│   ├── DAY30_OUTCOME_EVALUATION.md    # Day 30 Post-Harvest Evaluation Methodology
│   ├── DAY30_DRIFT_AND_BIAS.md        # Day 30 Covariate Drift & Bias Analysis
│   ├── DAY30_SCIENTIFIC_VALIDATION.md # Day 30 Validation & Golden Cases
│   ├── DAY30_FINAL_STATUS.md          # Day 30 Executive Status Report
│   └── ...
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # Multi-container orchestration
├── pyproject.toml                     # Python dependencies & pytest configuration
├── LICENSE                            # MIT License
└── README.md                          # Project README
```

---

## Installation & Running Locally

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) Docker and Docker Compose

### 1. Set Up Backend
```bash
git clone https://github.com/nupurmadaan04/AI-agriculture-yield-production.git
cd AI-agriculture-yield-production

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r Requirements.txt
```

### 2. Set Up Frontend
```bash
cd frontend
npm install
```

### 3. Run Development Servers
```bash
# Terminal 1: Backend API (Port 8000)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend Dashboard (Port 3000 / 5173)
cd frontend
npm run dev
```

- Web Dashboard: [http://localhost:3000](http://localhost:3000)
- Interactive Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Testing & Validation

Execute the full automated test suite:

```bash
pytest tests/ -v
```

```
============================= 456 passed in 405.02s =============================
```

To verify deterministic serving invariance:
```bash
python -m src.forecast_validation
```

---

## Limitations

1. **Historical Domain Boundaries**: The harmonized ground truth dataset covers 1966–2017. Post-2017 forecasts represent frozen pre-season projections and return `EVALUATION_UNAVAILABLE` until official harvest figures are released.
2. **Observational Evidence**: The panel reflects observational district records and does not establish biological causality.
3. **District Spatial Aggregation**: Yield metrics represent district-wide averages and should not be used as individual field-level agronomic prescriptions.

---

## Contributing

We welcome contributions! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) guide for setup instructions, contribution workflow, and coding standards.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
