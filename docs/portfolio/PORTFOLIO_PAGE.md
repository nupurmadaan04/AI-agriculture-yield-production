# Portfolio Project Showcase: AI Agricultural Forecasting & Decision Intelligence

---

## 1. Hero Section

### **Evidence-Governed Multi-Crop Yield Forecasting & Decision Intelligence**
*A production-grade spatio-temporal platform replacing blind machine learning adoption with walk-forward validation, empirical strategy governance, cryptographic provenance, and real-time MLOps observability.*

- **Repository**: [GitHub Repository Link]
- **Research Paper**: [docs/research_paper/paper.md](file:///docs/research_paper/paper.md)
- **Technical Case Study**: [docs/portfolio/PROJECT_CASE_STUDY.md](file:///docs/portfolio/PROJECT_CASE_STUDY.md)
- **Tech Stack**: Python 3.11 · FastAPI · Scikit-learn · Pandas · TypeScript · React 18 · Docker · Nginx

---

## 2. The Problem

Agricultural yield forecasting at the district level is essential for national buffer stock procurement, export tariffs, and disaster relief. However, conventional ML models deployed in agriculture suffer from three major vulnerabilities:
1. **Target Leakage**: Using contemporaneous harvest production figures to predict yield.
2. **Random Split Fallacy**: Shuffling cross-sectional records across time masks severe climate shock failures.
3. **Catastrophic Tail Errors**: Non-linear tree models over-predicting yields during extreme drought regimes.

---

## 3. The Solution & Architecture

Instead of assuming machine learning is universally optimal, the platform enforces an **evidence-based model governance gate**: candidate algorithms must outperform historical persistence baselines across multi-origin expanding walk-forward validation before receiving production certification.

```
+---------------------------------------------------------------------------------------------------------+
|                                    PLATFORM ARCHITECTURE FLOW                                           |
+---------------------------------------------------------------------------------------------------------+
  [DATA]        ICRISAT / DES Panel (71,601 records | 29 crops | 311 districts | 2010–2017)
                  ↓
  [FEATURES]    Leakage-Free Pre-Season Lag Generators (Lag-1, 3-Yr Rolling Mean, Area Share)
                  ↓
  [VALIDATION]  4-Fold Expanding Walk-Forward Tournament (Origins 2014, 2015, 2016, 2017)
                  ↓
  [GOVERNANCE]  Strategy Certification Gate:
                • PRODUCTION_READY (Oilseeds RF: +12.79% MAE gain, 75% win-rate)
                • CONDITIONAL_PRODUCTION (Sugarcane GBDT: +1.19% gain with 3-sigma clip)
                • BASELINE_PRODUCTION (12 crops: Historical District Mean persistence)
                  ↓
  [SERVING]     FastAPI Runtime (<50ms P95 | SHA-256 Provenance Signatures | Safety Guards)
                  ↓
  [OBSERVE]     MLOps Telemetry (Population Stability Index Drift | Post-Harvest Bias)
                  ↓
  [DECIDE]      Decision Workspace (React/TS | Multi-Evidence Briefs | Bounded Scenarios)
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Key Highlights & Verified Results

### 4.1 Multi-Crop Strategy Matrix
- **Oilseeds**: Passed all validation gates; deployed unconstrained Random Forest achieving **75.0% fold win-rate** and **+12.79% mean MAE gain** over historical persistence.
- **Sugarcane**: Raw GBDT failed in the 2015 drought (-10.19% fold loss, -1.60% net degradation). Deployed governed GBDT with mandatory 3-$\sigma$ district variance clipping, bounding downside risk and achieving **+1.19% aggregate gain** over baseline.
- **12 Baseline Crops**: In Rice, Wheat, Chickpea, Maize, Sorghum, and 7 other staples, historical district mean persistence demonstrated superior stability across climate shocks, leading to certified baseline deployment.

### 4.2 Exogenous Weather Ablation (Negative Finding)
A 5-tier tournament testing pre-season rainfall anomalies and temperature extremes across all 14 crops proved that pre-season district weather aggregations degraded forecasting accuracy or increased variance; historical autoregressive lags were universally preferred.

### 4.3 Engineering & Serving Performance
- **Sub-50ms Inference**: FastAPI endpoint with Pydantic v2 schemas and pre-inference rejection guards.
- **Cryptographic Provenance**: Every prediction receives an SHA-256 digital signature recorded in an append-only audit log.
- **Continuous Monitoring**: Live Population Stability Index (PSI) tracks feature drift; post-harvest census data evaluates directional signed bias ($\hat{y} - y$).
- **Multi-Container Infrastructure**: Nginx reverse proxy fronting FastAPI backend and React frontend with non-root security (`appuser`), fail-closed `/ready` probe (HTTP 503), and 580+ automated tests.

---

## 5. Technology Stack

| Layer | Technologies |
|---|---|
| **Backend & Serving** | Python 3.11.9, FastAPI, Uvicorn, Pydantic v2, Scikit-learn 1.6.1, SciPy (SLSQP) |
| **Data & Analytics** | Pandas 2.2.3, NumPy 2.2.3, Time-Series Econometrics, PSI Drift Tracking |
| **Frontend & UI** | React 18, TypeScript, Vite, Tailwind CSS, Lucide Icons, WCAG 2.1 AA |
| **DevOps & Testing** | Docker, Docker Compose, Nginx (Alpine), Pytest (581 tests collected), Git |

---

## 6. Project Limitations & Ethical Transparency

- **Historical Panel Ceiling**: Primary panel covers 2010–2017; forecasts beyond 2017 utilize historical lag persistence rather than post-2017 ground truth.
- **Geographic Aggregation**: Calibrated for district-level administrative planning, not farm-level precision agriculture.
- **Non-Causal Interpretations**: Model attributions (Marginal Reference Perturbation Attribution, Tree SHAP) reflect mathematical feature sensitivity within the trained manifold, not biological causality.
- **Decision Support Role**: Synthesizes evidence for human agronomic experts; strictly disclaims autonomous policy triggers.
