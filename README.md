# Agricultural Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Tests Passing](https://img.shields.io/badge/tests-100%25%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidence-driven, scientifically audited agricultural decision intelligence platform combining longitudinal multi-crop panel data (29 verified crops, 71,601 records), temporally validated machine learning forecasting, statistical baseline benchmarking, Tree SHAP explainability, SLSQP scenario optimization, and governed production forecast serving with cryptographic provenance.

---

## Overview

The **Agricultural Intelligence Platform** transforms decades of longitudinal district-level agricultural panel records and meteorological telemetry into structured, scientifically audited, and cryptographically verifiable **Decision Briefs**, **Model Governance Scorecards**, and **Production Forecasts**.

Unlike traditional black-box platforms that apply complex machine learning models uniformly without verification, this system implements an **evidence-first model governance framework**: machine learning models are deployed into production only when empirical temporal walk-forward validation demonstrates statistically significant superiority over simpler historical persistence baselines.

---

## Problem Statement

1. **Optimistic Bias & Data Leakage**: Standard agricultural machine learning studies frequently use randomized train/test splits that leak future temporal signals into training sets.
2. **Universal ML Forcing**: Machine learning models are often deployed blindly across all commodities even when simpler historical district averages achieve lower error and higher stability.
3. **Black-Box Opacity**: Agronomists and policymakers cannot audit how yield predictions or risk recommendations were derived without transparent feature attribution and data lineage.
4. **Lack of Governance & Provenance**: Production agricultural forecasts rarely carry verifiable cryptographic provenance linking predictions back to model versions, training boundaries, and audit logs.

---

## Research Question

> **"Can a multi-crop agricultural forecasting system systematically evaluate machine learning algorithms against statistical persistence baselines across multi-fold temporal walk-forward validation, enforce evidence-based model governance, and serve certified forecasts with complete cryptographic provenance?"**

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
                 ┌──────────────────────────────────────┐
                 │ FastAPI Backend + React Dashboard    │
                 └──────────────────────────────────────┘
```

---

## Data Sources

The platform ingests and standardizes longitudinal panel data from verified authorities:

| Source Identifier | Source Authority | Records | Temporal Coverage | Role in Platform |
| :--- | :--- | :--- | :--- | :--- |
| `ICRISAT_DLD_1966_2017` | ICRISAT District Level Data | 71,601 | 1966–2017 | Primary yield, area, and production panel |
| `DES_GOI_OGD` | Directorate of Economics & Statistics | Verified | 1997–2017 | District crop verification and cross-validation |
| `IMD_GRIDDED_PRECIP` | India Meteorological Department | Gridded | 1970–2017 | Gridded rainfall exogenous ablation features |

---

## Data Pipeline

1. **Unit Harmonization**: Standardized into `production_tonnes`, `area_ha`, and `yield_kg_ha`.
2. **Quality & Invariant Auditing**: 14 automated invariant checks (boundary clamping $[0, 150000]$, negative area removal, duplicate detection).
3. **Zero-Leakage Lag Generator**: Shift operators ($t-1, t-2$, rolling 3-year mean) computed strictly within grouped district-crop panels to prevent future temporal leakage.

---

## Multi-Crop Modeling

The platform evaluates machine learning algorithms (`RandomForestRegressor`, `GradientBoostingRegressor`) against statistical persistence baselines across 14 major agricultural commodities:

- **Cereals & Millets**: Rice, Wheat, Maize, Sorghum, Kharif Sorghum, Pearl Millet
- **Pulses**: Chickpea, Pigeonpea, Minor Pulses
- **Oilseeds & Cash Crops**: Total Oilseeds, Groundnut, Rapeseed & Mustard, Sesamum, Sugarcane

---

## Temporal Validation

Validation is conducted strictly using **4-Fold Expanding Walk-Forward Validation**:

- **Fold 1**: Train $\le 2013$ $\rightarrow$ Evaluate 2014
- **Fold 2**: Train $\le 2014$ $\rightarrow$ Evaluate 2015
- **Fold 3**: Train $\le 2015$ $\rightarrow$ Evaluate 2016
- **Fold 4**: Train $\le 2016$ $\rightarrow$ Evaluate 2017

---

## Model Selection Philosophy

> **The system does not assume that machine learning outperforms statistical baselines. Crop-specific models are evaluated using temporally ordered validation and are deployed only when their performance demonstrates sufficient robustness relative to simpler historical baselines.**

To achieve production certification, an algorithm must meet two non-negotiable gates:
1. **Error Reduction**: **MAE (ML) < MAE (Baseline)** with at least **+5.0% relative gain**.
2. **Fold Consistency**: Win rate **≥ 75%** across temporal walk-forward folds.

Where ML fails these gates, the platform certifies the **Historical District Mean Persistence Baseline**. Choosing statistical baselines when ML is unproven is a core strength of responsible engineering.

---

## Explainable AI & Decision Intelligence

- **Tree SHAP Attributions**: Exact local Shapley contributions calculated per prediction, decomposing how previous yield lags and cultivated area influenced the estimate.
- **Executive Decision Briefs**: Multi-signal fusion combining yield forecasts, historical volatility, anomaly detections, and Pareto-optimal scenario recommendations into auditable markdown/HTML reports.

---

## Forecast Governance & Production Routing

When a forecast request is received, the governance engine executes a 5-step pipeline:

```
[Request Input] 
  ↓ (1. Completeness Check)
  ↓ (2. Crop Registration Check)
  ↓ (3. District Coverage Check) ──► Rejects unmapped geographies (DISTRICT_UNSUPPORTED)
  ↓ (4. Model Artifact & SHA-256 Hash Verification)
  ↓ (5. Strategy Dispatcher)
      ├── PRODUCTION_READY     → Oilseeds Random Forest (with Sparse District Fallback)
      ├── CONDITIONAL_PROD     → Sugarcane Gradient Boosting (with 3-Sigma Variance Clipping)
      └── BASELINE_PRODUCTION  → Historical District Mean Persistence (12 Commodities)
```

---

## Validation Results

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

*Illustrative Prediction Examples*:
- Oilseeds in Ludhiana, Punjab: **`817.06 kg/ha`** (Random Forest ML)
- Sugarcane in Meerut, Uttar Pradesh: **`9,469.76 kg/ha`** (Gradient Boosting with Variance Clipping)

---

## Reproducibility

Dual independent inference runs across all 14 commodities produced **bitwise identical outputs**:

```
Δ = |Prediction (Run 1) - Prediction (Run 2)| = 0.00000000
```

Every prediction payload includes an SHA-256 cryptographic lineage hash. All inference requests and governance rejections are written to an append-oriented audit log (`Datasets/metadata/prediction_audit_log.csv`).

---

## API & Endpoints

The FastAPI backend exposes 50+ endpoints including 7 dedicated forecast serving routes:

- `GET /api/forecast/strategies`: Multi-crop strategy registry and evaluation evidence.
- `GET /api/forecast/certification`: High-level governance status breakdown.
- `GET /api/forecast/coverage`: 9,019 crop-state-district geographic coverage mappings.
- `POST /api/forecast/predict`: Governed forecasting inference with provenance and audit logging.
- `GET /api/forecast/provenance/{request_id}`: Cryptographic lineage record lookup.
- `GET /api/forecast/audit`: Append-oriented prediction audit trail events.
- `GET /api/forecast/health`: Subsystem health and governance guard status.

---

## Frontend Web Application

A responsive React 18 + TypeScript + Tailwind CSS application featuring:
- **Forecast Decision Wizard (`/forecast`)**: 4-step guided request form, dynamic district filtering, validation limitation notices, Result Cards, expandable *"Why this prediction?"* explanations, and live audit logs.
- **Modeling Readiness & Robustness (`/modeling-readiness`)**: 7-tab matrix showing baseline benchmarking, walk-forward folds, error regimes, exogenous ablations, and final certification scorecards.
- **Decision Intelligence (`/decision-intelligence`)**: Executive decision briefs and interactive provenance DAGs.
- **Agricultural Monitoring (`/monitoring`)**: Live CUSUM drift and temporal anomaly heatmaps.

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
│   └── ...
├── backend/                           # FastAPI backend server
│   ├── main.py                        # Master API router
│   ├── schemas/modeling.py            # Pydantic data contracts
│   └── services/modeling_service.py   # Domain query handlers
├── frontend/                          # React + TypeScript client application
│   ├── src/pages/ForecastIntelligence.tsx # Production forecast serving UI
│   ├── src/pages/ModelingReadiness.tsx    # Multi-crop certification UI
│   └── src/services/api.ts            # TanStack Query API client
├── Datasets/                          # Longitudinal panels & metadata
│   ├── processed/agricultural_panel.csv # Unified AGRI_PANEL_1.0 (71,601 records)
│   └── metadata/                      # Registry, coverage, and audit logs
├── Models/                            # Certified model artifacts
│   └── multicrop/                     # Joblib pipelines & JSON registries
├── tests/                             # Pytest test suites (100% passing)
├── docs/                              # Comprehensive scientific & architecture documentation
│   ├── DATASET_CARD.md                # Dataset Card for AGRI_PANEL_1.0
│   ├── MODEL_CARDS.md                 # Production model cards & baseline rationale
│   ├── ARCHITECTURE.md                # 10-tier architecture specification
│   ├── FINAL_SCIENTIFIC_AUDIT.md      # Canonical methodology & terminology audit
│   ├── DEMO_SCRIPT.md                 # 5-7 minute interview demonstration script
│   └── PROJECT_SUMMARY.md             # Executive portfolio summary
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # Multi-container orchestration
├── pyproject.toml                     # Python dependencies & pytest configuration
├── LICENSE                            # MIT License
└── README.md                          # Project README
```

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) Docker and Docker Compose

### 1. Clone & Set Up Backend
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

---

## Running Locally

### Backend Server
```bash
# From repository root
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend Development Server
```bash
cd frontend
npm run dev
```
Client Application: [http://localhost:5173](http://localhost:5173) (or [http://localhost:3000](http://localhost:3000))

### Running with Docker Compose
```bash
docker-compose up --build
```

---

## Testing

Execute the automated test suite across all engines, guards, and serving routers:

```bash
pytest tests/ -v
```

To verify deterministic serving invariance:
```bash
python -m src.forecast_validation
```

---

## Limitations

1. **Historical Domain Boundaries**: The harmonized dataset covers 1966–2017. Post-2017 forecasts reflect historical lag persistence.
2. **Observational Evidence**: The panel reflects observational district records and does not establish biological causality.
3. **District Spatial Aggregation**: Yield metrics represent district-wide averages and should not be used as individual field-level agronomic prescriptions.

---

## Future Work

- Ingestion of real-time satellite vegetation index telemetry (Sentinel-2 NDVI/EVI).
- Gridded meteorological reanalysis (ERA5-Land daily temperature and precipitation).
- Distribution-free conformal prediction intervals for theoretical coverage guarantees.

---

## Research Timeline

- **Days 1–16**: Single-crop baseline modeling, Tree SHAP explainability, spatial clustering, and Pareto scenario optimization.
- **Days 17–19**: Multi-crop data harmonization (`AGRI_PANEL_1.0`), baseline benchmarking, and algorithm screening across 14 commodities.
- **Days 20–22**: Temporal walk-forward validation, error regime diagnostics, and exogenous weather ablation audits.
- **Day 23**: Final model certification audit, residual distributions, and bitwise reproducibility verification.
- **Day 24**: Production forecast serving, pre-inference guards, 3-sigma variance clipping, cryptographic provenance, and audit logging.
- **Day 25**: Final scientific audit, model cards, dataset card, architecture documentation, and product release packaging.

---

## Contributing

We welcome contributions! Please review our [CONTRIBUTING.md](CONTRIBUTING.md) guide for setup instructions, contribution workflow, and coding standards.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
