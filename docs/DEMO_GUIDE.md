# Demonstration & Presentation Guide

## 1. 5-7 Minute Live Demo Script

### Target Context
- **State**: `Punjab`
- **District**: `Ludhiana`
- **Year**: `2017`

---

### Step-by-Step Demo Flow

#### Minute 0:00 - 1:00 | Introduction & Platform Overview
- **Action**: Open `http://localhost:3000/`.
- **Narration**: *"Welcome to the AI Agriculture Decision Intelligence Platform. Rather than operating as an opaque point-forecasting tool, this platform combines 51 years of ICRISAT longitudinal panel records across 311 Indian districts into a multi-layered, auditable decision-support system."*

#### Minute 1:00 - 2:00 | Agricultural Intelligence & Uncertainty Spreads
- **Action**: Navigate to `/intelligence`. Select `State: Punjab`, `District: Ludhiana`.
- **Narration**: *"Here we see our pre-season Random Forest forecaster predicting yield alongside empirical uncertainty spread bands (P10 to P90). Notice that we present calibrated prediction spreads rather than false-precision point estimates."*

#### Minute 2:00 - 3:00 | Model Reliability & Out-of-Time Validation
- **Action**: Navigate to `/reliability`.
- **Narration**: *"The model was rigorously validated using a strict chronological out-of-time protocol (Train: 1966-2012, Test: 2013-2017) to prevent temporal data leakage. On unseen test seasons, it achieves R² = 0.7866, MAE = 353.01 kg/ha, and 81.3% decile calibration coverage."*

#### Minute 3:00 - 4:00 | Monitoring, CUSUM Drift & Geospatial Clusters
- **Action**: Navigate to `/monitoring` and `/geospatial`.
- **Narration**: *"In Monitoring, our two-sided CUSUM drift detection monitors 311 districts, flagging sustained negative deviations. In Geospatial Intelligence, Local Moran's I identifies spatial yield autocorrelations and regional agro-climatic clusters."*

#### Minute 4:00 - 5:00 | Explainable AI & Scenario Simulation
- **Action**: Navigate to `/explainability` and `/scenarios`.
- **Narration**: *"Tree SHAP decomposes predictions against district baselines, showing how seasonal precipitation and synthetic fertilizer contribute to predicted yield. In the Scenario Lab, bounded SLSQP optimization models hypothetical input trade-offs."*

#### Minute 5:00 - 6:30 | Decision Intelligence & Cryptographic Audit Trail
- **Action**: Navigate to `/decision`. Run analysis for `Punjab -> Ludhiana -> 2017`.
- **Narration**: *"Finally, Decision Intelligence synthesizes all evidence into an Executive Brief. Every claim is mapped via a Provenance DAG back to model versions and dataset rows, certified by a deterministic SHA-256 certificate (`DEC-8C51B090BA`)."*

---

## 2. 15-Slide Presentation Storyboard

| Slide # | Slide Title | Core Message |
| :--- | :--- | :--- |
| **1** | **Title Slide** | AI Agriculture Decision Intelligence Platform: An Auditable Decision-Support System |
| **2** | **The Core Problem** | Agricultural forecasting suffers from data leakage, black-box opacity, and disconnected risk signals |
| **3** | **Platform Vision** | Bridging machine learning with verifiable agronomic policy planning |
| **4** | **Data Engineering** | 51 years of ICRISAT district panel records (1966–2017, 311 districts, 20 states) |
| **5** | **ML Architecture** | Dual pre-season and post-harvest Random Forests with ensemble uncertainty bands |
| **6** | **Strict Validation** | Chronological out-of-time evaluation ($R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$) |
| **7** | **Calibration & Drift** | 81.3% decile spread coverage and Population Stability Index tracking |
| **8** | **Temporal Monitoring** | Two-sided CUSUM drift detection & multi-window hazard alerts across 311 districts |
| **9** | **Geospatial Intelligence**| K-Means agro-climatic clustering and Local Moran's I spatial autocorrelation |
| **10** | **Explainability (XAI)**| Local Tree SHAP attribution & non-causal interpretability boundaries |
| **11** | **Scenario Simulation** | Bounded SLSQP input trade-off exploration |
| **12** | **Decision Synthesis** | Multi-source evidence fusion ranking operational priorities |
| **13** | **Auditability & DAG** | Deterministic SHA-256 certificates (`DEC-xxxx`) and Statement $\rightarrow$ Data provenance |
| **14** | **System Architecture**| React 18 frontend + FastAPI backend + Scikit-Learn analytical engines |
| **15** | **Scientific Limits** | Statistical decision support; non-causal agronomic evaluation |
