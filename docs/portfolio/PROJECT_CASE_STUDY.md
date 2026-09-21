# Technical Case Study: Evidence-Governed Agricultural Forecasting & Decision Intelligence

**Author**: AI Agriculture Intelligence Platform Engineering Group  
**Target Roles**: Senior Data Scientist, Machine Learning Engineer, MLOps Engineer, Decision Intelligence Architect  
**Domain**: Agricultural Econometrics, Spatio-Temporal Forecasting, Model Governance, Production Serving

---

## 1. Executive Summary

In emerging agricultural economies, district-level crop yield forecasts directly inform food security reserves, buffer stock procurement, price stabilization tariffs, and localized disaster relief. However, operational machine learning deployments in this sector frequently fail. Standard ML approaches suffer from severe data leakage (e.g. using harvest production to predict yield), random cross-validation failure under spatial-temporal autocorrelation, and catastrophic tail errors during climate shocks such as droughts.

This case study documents the end-to-end design, empirical validation, operational governance, and production deployment of the **Agricultural Forecasting & Decision Intelligence Platform**. Built on a canonical panel of **71,601 district-year observations covering 29 crops across 20 Indian states and 311 districts (2010–2017)**, the platform rejects the common assumption that machine learning is universally superior to statistical baselines. 

Instead, the platform implements an **evidence-governed model selection policy** evaluated across four-fold expanding walk-forward temporal cross-validation (origins 2014–2017):
- **Unconstrained Machine Learning** (`PRODUCTION_READY`) is deployed solely for **Oilseeds**, achieving a **75.0% fold win-rate** and **+12.79% mean MAE reduction** over historical district mean persistence.
- **Conditional Machine Learning** (`CONDITIONAL_PRODUCTION`) is certified for **Sugarcane**, where raw Gradient Boosted Decision Trees degraded during the 2015 drought (-10.19% loss in Fold 2), but governed execution with mandatory 3-$\sigma$ district variance clipping restored an aggregate **+1.19% MAE gain** over baseline persistence.
- **Statistical District Mean Baselines** (`BASELINE_PRODUCTION`) are mandated for **12 commodities** (including Rice and Wheat), where simpler historical persistence outperformed complex non-linear models across sequential climate shocks.

---

## 2. Problem Statement & Operational Vulnerabilities

### 2.1 The Core Agricultural Forecasting Challenge
District agricultural yields in India exhibit extreme inter-annual variance driven by monsoon volatility, uneven irrigation infrastructure, and micro-climate fluctuations. Decision-makers (e.g., Ministry of Agriculture, state relief commissioners) require pre-season yield estimates at least 3–6 months prior to harvest to allocate grain storage, plan import/export tariffs, and prepare drought credit relief.

### 2.2 Why Conventional ML Approaches Fail in Agriculture
1. **Target Leakage via Algebraic Identities**:
   Agricultural yield is defined as:
   $$\text{Yield} = \frac{\text{Production}}{\text{Cultivated Area}} \times 1000$$
   Many published studies train models using contemporaneous harvest production figures, producing near-perfect regression scores ($R^2 > 0.98$). At pre-sowing lead times, harvest production is fundamentally unknown. Using it constitutes 100% target leakage.
2. **Random Cross-Validation Fallacy**:
   Randomly shuffling panel data splits training and testing sets across common weather years. In an El Niño year (e.g. 2015), test records share the same underlying climate shock with training records, severely overestimating out-of-sample accuracy.
3. **Catastrophic Out-of-Distribution Tail Errors**:
   Complex tree ensembles (Random Forest, GBDT) cannot extrapolate beyond their observed training feature bounds. During severe drought shocks, unconstrained trees severely over-predict yields in rainfed districts, incurring massive negative gains compared to robust historical means.

---

## 3. Data Engineering & Leakage Isolation

### 3.1 Canonical Multi-Crop Panel Overview
- **Records**: 71,601 verified observations
- **Crops**: 29 standardized commodities
- **Geography**: 20 Indian States, 311 Districts (harmonized to 1966 base boundaries)
- **Active Panel Temporal Window**: 2010–2017 (8 normalized agricultural years)
- **Historical Context**: 1966–2017 (ICRISAT District Level Database)
- **Primary Artifact**: `Datasets/processed/agricultural_panel.csv` (SHA-256: `13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b`)

### 3.2 Leakage-Free Pre-Season Feature Pipeline
To ensure 100% operational validity, all features are extracted strictly prior to planting of harvest year $t$:
1. **Lag-1 Yield ($y_{i, t-1}$)**: Prior year physical yield.
2. **Lag-2 Yield ($y_{i, t-2}$)**: Two-year prior physical yield.
3. **Three-Year Rolling Mean ($\bar{y}_{i, t-1:t-3}$)**: Medium-term productivity baseline.
4. **Cultivated Area Share**: District crop area normalized against total district cropped area.
5. **Strict Exclusion**: Harvest-year production $P_{i, t}$ is strictly masked from the feature pipeline.

---

## 4. Modeling & Validation Architecture

### 4.1 Candidate Model Classes
1. **Historical District Mean Persistence**: Expanding historical mean of district $i$ using all years $\tau < T$.
2. **District 3-Year Rolling Mean**: Moving average of preceding 3 seasons.
3. **Random Forest Regressor**: 150 estimators, max depth 12, min samples per leaf 4.
4. **Gradient Boosted Decision Trees (GBDT)**: 100 boosting stages, learning rate 0.05, max depth 4.

### 4.2 Expanding Walk-Forward Temporal Cross-Validation
Rather than relying on a single test holdout, candidate models were evaluated across four sequential expanding historical origins $T \in \{2014, 2015, 2016, 2017\}$:
- **Fold 1 (Origin 2014)**: Train on 2010–2013; Test on 2014.
- **Fold 2 (Origin 2015)**: Train on 2010–2014; Test on 2015 (Major Pan-India Drought Shock).
- **Fold 3 (Origin 2016)**: Train on 2010–2015; Test on 2016 (Post-drought recovery).
- **Fold 4 (Origin 2017)**: Train on 2010–2016; Test on 2017 (Favorable monsoon).

---

## 5. Model Governance & Strategic Certification

The platform enforces three strict governance states based on empirical walk-forward evidence:

```
+-------------------------------------------------------------------------------------------------------+
|                                  GOVERNED STRATEGY MATRIX SUMMARY                                     |
+---------------------+--------------------------+-----------------------+---------------+--------------+
| Commodity           | Governed Status          | Deployed Strategy     | Strategy MAE  | Gain vs Base |
+---------------------+--------------------------+-----------------------+---------------+--------------+
| **Oilseeds**        | `PRODUCTION_READY`       | Historical ML (RF)    | 549.67 kg/ha  | **+12.79%**  |
| **Sugarcane**       | `CONDITIONAL_PRODUCTION` | Governed GBDT (+Clip) | 1,467.97 kg/ha| **+1.19%**   |
| **Rice**            | `BASELINE_PRODUCTION`    | Hist. District Mean   | 310.28 kg/ha  | Baseline     |
| **Wheat**           | `BASELINE_PRODUCTION`    | Hist. District Mean   | 381.65 kg/ha  | Baseline     |
| **10 Other Crops**  | `BASELINE_PRODUCTION`    | Hist. District Mean   | Various       | Baseline     |
+---------------------+--------------------------+-----------------------+---------------+--------------+
```

### 5.1 Case Study: Oilseeds Unconstrained Production
- **Win Rate**: 75.0% (3 of 4 folds won against baseline).
- **MAE Improvement**: +12.79% mean gain over baseline persistence (549.67 kg/ha vs 616.60 kg/ha).
- **Regime Stability**: Worst-fold degradation during 2016 was bounded at -2.34%, passing all safety gates.

### 5.2 Case Study: Sugarcane Dual-Metric Resolution & 3-$\sigma$ Clipping
- **Raw GBDT Failure**: In Fold 2 (2015 drought), unclipped GBDT over-predicted severely (-10.19% loss), resulting in a net **-1.60% loss** across all folds.
- **Governed Clipping Rule**: If $|\hat{y} - \mu_{\text{dist}}| > 3\sigma_{\text{dist}}$, automatically fallback to $\mu_{\text{dist}}$.
- **Governed Result**: Bounded Fold 2 loss to -5.78% while preserving gains in Folds 3 (+9.75%) and 4 (+8.44%), achieving an aggregate **+1.19% MAE gain** over baseline persistence across 1,193 test observations.

### 5.3 Negative Result: Pre-Season Weather Feature Ablation
In Day 22, a 5-tier ablation tournament tested whether pre-season weather features (rainfall anomalies, temperature extremes) improved forecasting across all 14 crops. Adding pre-season weather degraded MAE or increased variance across 100% of crops; the historical-only tier ($EXP-22A$) was universally certified.

---

## 6. Production Engineering & Serving Runtime

- **FastAPI Serving Engine**: Sub-50ms $P95$ latency per request.
- **Cryptographic Provenance**: Every forecast calculates an SHA-256 digital signature:
  $$\text{Provenance Hash} = \text{SHA256}(\text{UUID} \parallel \text{Crop} \parallel \text{District} \parallel \text{Year} \parallel \text{Model Version} \parallel \text{Dataset Hash})$$
- **Explainability**: Marginal Reference Perturbation Attribution and Tree SHAP provide local feature attributions disclaimed as non-causal.
- **Uncertainty**: Empirical P10–P90 ensemble dispersion intervals extracted from 150 decision tree estimators.
- **Continuous Monitoring**: Population Stability Index (PSI) tracks feature drift ($<0.10$ Normal, $0.10\le \text{PSI}<0.25$ Moderate, $\ge 0.25$ Significant); post-harvest evaluation computes signed bias ($\hat{y} - y$).
- **Multi-Container Infrastructure**: Nginx reverse proxy routing port 80 to internal FastAPI (port 8000) and React/Vite frontend; non-root execution (`appuser`), fail-closed `/ready` probe (HTTP 503).

---

## 7. Key Lessons Learned

1. **Machine learning is not an automatic upgrade**: In 12 of 14 agricultural commodities, simple historical persistence baselines outperformed complex non-linear models. Mandating baseline deployment where appropriate is a governance victory.
2. **Single-split validation is dangerous**: Models that appear accurate on a single holdout year can fail catastrophically during climate shocks. Expanding walk-forward validation is non-negotiable in operational agricultural modeling.
3. **Operational fallbacks prevent disaster**: Adding automated variance clipping and fallback to district means prevents extreme out-of-distribution tail errors in production.
