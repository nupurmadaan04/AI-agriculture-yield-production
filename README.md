# Agricultural Forecasting & Decision Intelligence Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E.svg)](https://scikit-learn.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Pytest Suite](https://img.shields.io/badge/pytest-541%20passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidence-governed multi-crop agricultural forecasting platform combining leakage-aware temporal walk-forward validation, crop-specific model selection, empirical forecast governance, explainability, MLOps drift monitoring, and decision intelligence.

[Quick Start](#15-quick-start) • [Architecture](#4-platform-overview) • [Methodology](#6-modeling) • [Validation](#7-validation) • [Contributing](CONTRIBUTING.md) • [Code of Conduct](CODE_OF_CONDUCT.md) • [License](LICENSE)

---

## 1. One-Line Summary

An evidence-governed multi-crop agricultural forecasting platform combining leakage-aware temporal validation, crop-specific model selection, forecast governance, explainability, monitoring, and decision intelligence.

---

## 2. Why This Project Exists

District-level agricultural yields in developing economies exhibit extreme volatility driven by monsoon oscillations, localized drought shocks, and heterogeneous irrigation infrastructure. Government planners, buffer stock managers, and commodity analysts require reliable pre-season yield forecasts 3–6 months prior to harvest to allocate grain storage and plan disaster relief. 

However, operational machine learning deployments in this domain frequently fail: standard models suffer from severe data leakage, random cross-validation failure under spatial-temporal autocorrelation, and catastrophic tail errors during climate anomalies. This platform was engineered to replace blind model adoption with an empirical, evidence-first governance architecture.

---

## 3. What Makes This Different

- **Chronological Walk-Forward Validation**: Replaces optimistic random train/test splits with 4 expanding historical origins (2014–2017) to stress-test models against severe climate shocks (e.g., the 2015 pan-India drought).
- **Crop-Specific Strategy Selection**: Recognizes that agricultural commodities operate under distinct agronomic regimes; evaluates candidate algorithms independently across 14 crops.
- **Statistical Persistence Baselines**: Benchmarks machine learning against Historical District Mean persistence, responsibly defaulting to baselines whenever ML fails to demonstrate consistent superiority.
- **Automated Fallback Strategies**: Enforces runtime 3-$\sigma$ district variance clipping and graceful degradation to prevent catastrophic out-of-distribution prediction spikes.
- **Explicit Leakage Controls**: Masks contemporaneous harvest production figures from pre-season feature sets, eliminating mathematical identity leakage ($Yield = Production / Area$).
- **Empirical Uncertainty Evidence**: Extracts empirical P10–P90 ensemble dispersion intervals across 150 decision tree estimators without assuming Gaussian residuals.
- **Cryptographic Provenance**: Computes bitwise reproducible SHA-256 digital execution signatures for every forecast, synchronously logged to an append-only audit trail.
- **MLOps Drift Observability**: Continuously tracks feature distribution shifts via Population Stability Index (PSI) and evaluates post-harvest signed bias ($\hat{y} - y$).
- **Strict Scenario Separation**: Structurally separates hypothetical simulation perturbations (`[SCENARIO]`) from empirical forecasts (`[PREDICTED]`) and historical ground truth (`[OBSERVED]`).

---

## 4. Platform Overview

```
                          ┌──────────────────────────────────────┐
                          │ Authoritative Sources (ICRISAT / DES)│
                          └──────────────────┬───────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ Data Quality & Normalization Audit   │
                          └──────────────────┬───────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ Canonical Agricultural Panel (71.6k) │
                          └──────────────────┬───────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ Leakage Audit & Pre-Season Features  │
                          └──────────────────┬───────────────────┘
                                             ↓
                   ┌─────────────────────────┴─────────────────────────┐
                   ↓                                                   ↓
         ┌───────────────────┐                               ┌───────────────────┐
         │Historical Baseline│                               │  Machine Learning │
         │(District Persistence)                             │  (Random Forest)  │
         └─────────┬─────────┘                               └─────────┬─────────┘
                   └─────────────────────────┬─────────────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ 4-Fold Expanding Walk-Forward Valid. │
                          └──────────────────┬───────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ Strategy Governance Registry Gate    │
                          └──────────────────┬───────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ FastAPI Forecast Service (<50ms P95) │
                          └──────────────────┬───────────────────┘
                                             ↓
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ↓                                   ↓                                   ↓
┌───────────────────┐               ┌───────────────────┐               ┌───────────────────┐
│Explainability(XAI)│               │Uncertainty (P1090)│               │SHA-256 Provenance │
└────────┬──────────┘               └────────┬──────────┘               └────────┬──────────┘
         └───────────────────────────────────┼───────────────────────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ Continuous Drift Monitoring (PSI)    │
                          └──────────────────┬───────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ Decision Intelligence & Workspace    │
                          └──────────────────┬───────────────────┘
                                             ↓
                          ┌──────────────────────────────────────┐
                          │ Frontend WebShell (React 18 / Vite)  │
                          └──────────────────────────────────────┘
```

---

## 5. Dataset

The platform is grounded in the **Agricultural Intelligence Unified Panel (`AGRI_PANEL_1.0`)**:
- **Total Records**: 71,601 verified district-year observations.
- **Commodity Breadth**: 29 standardized crops.
- **Geographic Extent**: 20 Indian States and 311 Districts (harmonized to 1966 base boundaries).
- **Active Panel Temporal Coverage**: 2010–2017 (8 normalized multi-crop agricultural years).
- **Historical Panel Context**: 1966–2017 (ICRISAT District Level Database).
- **Primary Target Metric**: `yield_kg_ha` (Kilograms per Hectare).
- **Exposure Metric**: `area_ha` (Cultivated Area in Hectares).
- **Data Provenance**: Directorate of Economics & Statistics (DES) & ICRISAT.
- **Integrity Digest**: SHA-256 `13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b`.

The multi-crop panel enforces strict missing-value rejection, unit standardization (kg/ha and hectares), and 1966 geographic boundary harmonization.

---

## 6. Modeling

Candidate forecasting strategies are evaluated independently for each crop:
1. **Historical District Mean**: Expanding historical average of district yields prior to the forecast origin year ($\tau < T$).
2. **District 3-Year Rolling Mean**: Moving average of the preceding three historical seasons.
3. **Random Forest Regressor**: 150 estimators, maximum depth 12, minimum samples per leaf 4.
4. **Gradient Boosted Decision Trees (GBDT)**: 100 boosting stages, learning rate 0.05, maximum depth 4.

Input features are restricted strictly to pre-season variables: prior-year yield ($y_{t-1}$), two-year lag ($y_{t-2}$), 3-year rolling mean, and pre-season cultivated area share.

All candidate models and crop-specific strategies are fully certified and tracked in the multi-crop strategy registry.

---

## 7. Validation

To prevent temporal leakage and evaluate climate shock resilience, candidate models undergo **4-fold expanding walk-forward temporal cross-validation** across historical origins $T \in \{2014, 2015, 2016, 2017\}$:
- **Origin 2014**: Train on 2010–2013; Test on 2014.
- **Origin 2015**: Train on 2010–2014; Test on 2015 (*Major Pan-India Drought Shock*).
- **Origin 2016**: Train on 2010–2015; Test on 2016 (*Post-drought recovery*).
- **Origin 2017**: Train on 2010–2016; Test on 2017 (*Favorable monsoon*).

```
Fold 1 (2014): [== Train (2010-2013) ==] [ Test: 2014 ]
Fold 2 (2015): [===== Train (2010-2014) =====] [ Test: 2015 (DROUGHT) ]
Fold 3 (2016): [======== Train (2010-2015) ========] [ Test: 2016 ]
Fold 4 (2017): [=========== Train (2010-2016) ===========] [ Test: 2017 ]
```

Evaluation criteria require candidate ML models to achieve:
1. Mean MAE improvement over baseline persistence ($>0\%$).
2. Fold win rate $\ge 50\%$ across historical origins.
3. Bounded worst-fold degradation ($< 5\%$ loss).

---

## 8. Final Forecast Governance

Validation evidence is codified into three non-overlapping governance tiers in `Models/multicrop/forecast_strategy_registry.json`:

| Governance Status | Qualifying Evidence | Operational Policy | Certified Crops |
|---|---|---|---|
| **`PRODUCTION_READY`** | Win-rate $\ge 75\%$, positive mean MAE gain, bounded worst-fold degradation. | Deploys unconstrained ML with sparse-district fallback. | **Oilseeds** (+12.79% mean MAE gain, 75% win rate) |
| **`CONDITIONAL_PRODUCTION`** | Win-rate $\ge 50\%$, positive gain, but exhibits drought shock volatility. | Deploys ML with mandatory 3-$\sigma$ district variance clipping. | **Sugarcane** (+1.19% governed MAE gain, 50% win rate) |
| **`BASELINE_PRODUCTION`** | Baseline MAE $\le$ ML MAE across walk-forward folds or win-rate $< 50\%$. | Mandates Historical District Mean persistence in production. | **12 Crops** (Rice, Wheat, Chickpea, Maize, Sorghum, etc.) |

*Note: These tiers represent empirical governance policies, not subjective model rankings.*

---

## 9. Explainability

Feature attributions are generated in real time to provide transparent local interpretability:
- **Marginal Reference Perturbation Attribution**: Measures isolated model sensitivity by substituting live feature values with the district's historical median reference vector $\mathbf{x}^{(0)}$.
- **Tree SHAP**: Decomposes tree ensemble predictions into additive feature attributions for interactive decision support.
- **Scientific Guardrail**: All explanations carry the mandatory tag `MODEL_ATTRIBUTION` with explicit disclaimers: attributions reflect mathematical sensitivity within the trained model manifold, not agronomic causality.

---

## 10. Uncertainty

Forecast uncertainty is quantified non-parametrically:
- **Empirical P10–P90 Ensemble Intervals**: Evaluates dispersion across all 150 individual decision tree estimators:
  $$\hat{y}_{P10} = \text{Quantile}_{0.10}\left(\{f_b(X)\}_{b=1}^{150}\right), \quad \hat{y}_{P90} = \text{Quantile}_{0.90}\left(\{f_b(X)\}_{b=1}^{150}\right)$$
- **Standardized Nomenclature**: Explicitly tagged in API responses and UI views as **"Empirical P10–P90 Ensemble Interval"**.
- **Limitation**: *This is empirical ensemble uncertainty evidence, not a formal frequentist confidence interval.*

---

## 11. Monitoring

The platform maintains continuous runtime and post-harvest MLOps observability:
- **Operational Health**: Sub-second tracking of CPU utilization, RSS memory footprint (~340 MB), and $P_{50}..P_{99}$ latency distributions.
- **Covariate Drift**: Computes **Population Stability Index (PSI)** across 10 empirical quantile bins for incoming feature distributions:
  - $\text{PSI} < 0.10$: Normal (Green)
  - $0.10 \le \text{PSI} < 0.25$: Moderate Shift (Amber)
  - $\text{PSI} \ge 0.25$: Significant Drift (Red Alert)
- **Post-Harvest Outcome Evaluation**: Once official harvest census records are ingested, an asynchronous pipeline computes directional **signed bias** ($\hat{y} - y$), MAE, RMSE, and MAPE across agro-climatic zones.

---

## 12. Decision Intelligence

The interactive Decision Workspace (`/decision-workspace`) synthesizes multi-dimensional evidence for policy planners while enforcing strict semantic entity typing:

```
+---------------------------------------------------------------------------------------------------------+
|                                    EVIDENCE ENTITY TAXONOMY                                             |
+---------------------+-----------------------------------------------------------------------------------+
| Entity Type         | Definition & Presentation Rule                                                    |
+---------------------+-----------------------------------------------------------------------------------+
| `[OBSERVED]`        | Empirical historical ground truth recorded in official agricultural census data.  |
| `[PREDICTED]`       | Pre-season point forecasts generated by certified production strategies.         |
| `[SCENARIO]`        | Bounded what-if simulations (e.g. SLSQP acreage optimization); purely hypothetical.|
| `[VALIDATION]`      | Historical walk-forward benchmark results from completed tournament folds.        |
| `[MONITORING]`      | Live runtime metrics, latency profiles, and Population Stability Index drift flags|
+---------------------+-----------------------------------------------------------------------------------+
```

---

## 13. Technology Stack

- **Backend & Serving**: Python 3.11.9, FastAPI, Uvicorn, Pydantic v2, Scikit-learn 1.6.1, SciPy (SLSQP).
- **Data Engineering**: Pandas 2.2.3, NumPy 2.2.3, Time-Series Econometrics, PSI Drift Tracking.
- **Frontend & WebShell**: React 18, TypeScript 5, Vite, Tailwind CSS, Lucide Icons, WCAG 2.1 AA.
- **Infrastructure & MLOps**: Docker Engine, Docker Compose, Nginx (Alpine), Pytest (581 collected tests), Git.

---

## 14. Project Structure

```text
AI-agriculture-yield-production/
├── backend/                  # FastAPI serving application & route controllers
│   ├── main.py               # API entrypoint, CORS, routes & probes
│   └── core/                 # App configuration & structured error schemas
├── src/                      # Core modeling, governance & intelligence engines
│   ├── prediction_service.py # Governed inference & 3-sigma fallback runtime
│   ├── provenance_service.py # SHA-256 cryptographic lineage generation
│   ├── monitoring_service.py # Population Stability Index (PSI) drift tracking
│   ├── decision_workspace.py # Decision brief synthesis & scenario engine
│   └── explainability_engine.py # Marginal Reference Perturbation & Tree SHAP
├── frontend/                 # React 18 + TypeScript production WebShell
│   ├── src/pages/            # View pages (Forecast, Explore, Monitoring, Workspace)
│   └── src/components/       # UI tokens, accessible navigation & data badges
├── Datasets/                 # Canonical data assets & append-only audit logs
│   ├── processed/            # agricultural_panel.csv (71,601 records)
│   └── metadata/             # Audit logs, telemetry & certification matrices
├── Models/                   # Serialized model pipelines & strategy registries
│   └── multicrop/            # forecast_strategy_registry.json
├── nginx/                    # Production reverse proxy configuration
│   └── nginx.conf            # OWASP security headers & proxy routing
├── tests/                    # Automated test suite (541 verified passing tests)
│   ├── test_forecast_router.py
│   ├── test_api_contracts.py
│   └── test_deployment_verification.py
├── Dockerfile                # Hardened multi-stage container build (non-root)
├── docker-compose.yml        # Multi-container orchestration (Nginx + Backend)
├── CONTRIBUTING.md           # Contribution guidelines & development setup
├── CODE_OF_CONDUCT.md        # Contributor Covenant v2.1 code of conduct
├── LICENSE                   # MIT open-source license
└── README.md                 # Platform architecture & scientific documentation
```

---

## 15. Quick Start

### 15.1 Local Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/nupurmadaan04/AI-agriculture-yield-production.git
cd AI-agriculture-yield-production

# Create and activate Python 3.11 virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r Requirements.txt
```

### 15.2 Start the Backend API
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# API Docs available at: http://localhost:8000/docs
```

### 15.3 Start the Frontend WebShell
```bash
cd frontend
npm install
npm run dev
# WebShell available at: http://localhost:5173
```

### 15.4 Multi-Container Docker Deployment
```bash
docker-compose up --build -d
# Frontend & API proxied via Nginx on: http://localhost
```

### 15.5 Run Automated Verification Tests
```bash
# Execute core Day 34-36 reproducibility, UI contracts & deployment suites (32 tests)
pytest tests/test_day36_reproducibility.py tests/test_day35_ui_contracts.py tests/test_deployment_verification.py -v

# Run full test suite (541 passing tests)
pytest tests/ -v
```

---

## 16. 5-Minute Golden Demonstration Tour

Follow this guided tour using verified golden test cases:
1. **Explore the Data (`/data`)**: Inspect the 71,601-record multi-crop panel. Filter by State (`Madhya Pradesh`), District (`Ujjain`), and Crop (`Total Oilseeds`).
2. **Generate a Governed Forecast (`/forecast`)**:
   - Select **Crop**: `Total Oilseeds`, **State**: `Madhya Pradesh`, **District**: `Ujjain`, **Year**: `2017`.
   - Click **Generate Forecast**: Note sub-50ms latency, active `PRODUCTION_READY` badge, predicted yield (~1,180 kg/ha), and empirical P10–P90 spread [1,020–1,340 kg/ha].
3. **Inspect Conditional Safety (`/forecast`)**:
   - Select **Crop**: `Sugarcane`, **State**: `Uttar Pradesh`, **District**: `Muzaffarnagar`, **Year**: `2017`.
   - Notice the `CONDITIONAL_PRODUCTION` badge indicating active 3-$\sigma$ variance clipping protection.
4. **Inspect Baseline Certification (`/forecast`)**:
   - Select **Crop**: `Rice`, **State**: `Punjab`, **District**: `Ludhiana`, **Year**: `2017`.
   - Observe the `BASELINE_PRODUCTION` badge confirming deployment of Historical District Mean persistence.
5. **Audit Prediction Lineage (`/predictions`)**: Search for recent forecast records; inspect immutable SHA-256 provenance hashes and model metadata.
6. **Check Drift & Health (`/monitoring` & `/observability`)**: Review Population Stability Index (PSI) quantile distributions and live server resource telemetry.
7. **Simulate What-If Scenarios (`/decision-workspace`)**: Enter the Decision Workspace; adjust acreage allocation sliders under bounded historical constraints to evaluate trade-offs.

---

## 17. Repository Documentation & Guidelines

- **Project README**: [README.md](README.md) - Complete platform architecture, methodology, validation evidence, and deployment guide.
- **Contribution Guidelines**: [CONTRIBUTING.md](CONTRIBUTING.md) - Setup instructions, coding standards, and pull request workflow.
- **Code of Conduct**: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Contributor Covenant v2.1 standards for community participation.
- **Open Source License**: [LICENSE](LICENSE) - MIT License.

---

## 18. Transparent Limitations

- **Temporal Boundary**: The harmonized agricultural panel ends in 2017; forecasts for post-2017 horizons rely on historical autoregressive lag persistence rather than post-2017 census ground truth.
- **Geographic Aggregation**: Calibrated specifically for district-level administrative planning; not designed for field-scale precision farming or plot-level fertilizer prescriptions.
- **Non-Causal Interpretation**: Model attributions reflect mathematical feature sensitivity within the trained model manifold, not agronomic causality.
- **Decision Support Role**: Synthesizes structured quantitative evidence to assist human agronomic experts; strictly disclaims autonomous policy triggers or automated subsidy disbursement.
- **Deployment Profile**: Containerized and verified via Docker Compose with Nginx reverse proxy; public cloud deployment (e.g. AWS/GCP) is simulated in local container environments.
