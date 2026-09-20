# Day 19: Multi-Crop Forecasting Methodology & Zero-Leakage Architecture

## 1. Executive Summary

This document formalizes the machine learning methodology, feature timing protocols, temporal cross-validation mechanics, and model selection criteria for multi-crop yield forecasting across the **14 `MODEL_READY` crops** identified in Day 18.

All experiments strictly enforce:
1. **Zero-Leakage Shifted Features**: Pre-season covariates are restricted to past observations ($t-1, t-2$).
2. **Prohibition of Simultaneous Output**: Concurrent harvest-year production $P_t$ is strictly banned because $\text{Yield}_t = \frac{P_t}{A_t}$ constitutes mathematical target reconstruction.
3. **Temporal Partitioning**: Training period (2011–2015) is completely separated from the final held-out test period (2016–2017).
4. **Isolated Hyperparameter Tuning**: Expanding-window temporal CV is executed solely inside the training partition without touching the held-out test years.

---

## 2. Mathematical Formulation & Feature Timing

### 2.1 Target Formulation
For each crop $c$ and district $d$ in harvest year $t$:
$$y_{c, d, t} = \text{Yield (kg/ha)}$$

### 2.2 Pre-Season Covariate Matrix
All candidate predictors must be deterministically known *prior to crop planting*:

| Feature Name | Formulation | Observation Timing | Leakage Classification |
| :--- | :--- | :--- | :--- |
| `yield_lag_1` | $y_{c, d, t-1}$ | Prior Season ($t-1$) | **ZERO LEAKAGE** |
| `yield_lag_2` | $y_{c, d, t-2}$ | 2 Seasons Prior ($t-2$) | **ZERO LEAKAGE** |
| `yield_rolling_3yr_mean` | $\frac{1}{3}\sum_{k=1}^3 y_{c, d, t-k}$ | Prior 3 Seasons ($t-3$ to $t-1$) | **ZERO LEAKAGE** |
| `area_lag_1` | $\text{Area}_{c, d, t-1}$ | Prior Season ($t-1$) | **ZERO LEAKAGE** |
| `state_encoded` | $\text{OrdinalCode}(\text{State}_d)$ | Static Geographic Prior | **ZERO LEAKAGE** |
| `year` | $t$ | Current Sowing Year | **ZERO LEAKAGE** (Trend Signal) |

> [!CRITICAL]
> **Anti-Leakage Shift Rule**:
> Any rolling average must be applied *after* shifting the series by 1 lag:
> $$\text{rolling\_mean}(t) = \text{mean}\left(\{y_{t-1}, y_{t-2}, y_{t-3}\}\right)$$
> Calculating $\text{mean}\left(\{y_t, y_{t-1}, y_{t-2}\}\right)$ would introduce $y_t$ lookahead bias.

---

## 3. Candidate Algorithm Families

### 3.1 Random Forest Regressor (`RandomForestRegressor`)
- Ensemble of $B$ bootstrap-aggregated orthogonal regression trees.
- Reduces variance across volatile agricultural yield cycles.
- Empirical ensemble prediction dispersion:
  $$\hat{y}_{\text{P10}} = \text{Percentile}_{10}(\{\hat{y}_b\}_{b=1}^B), \quad \hat{y}_{\text{P90}} = \text{Percentile}_{90}(\{\hat{y}_b\}_{b=1}^B)$$
- Search space: `n_estimators` $\in [150, 200, 250]$, `max_depth` $\in [8, 12, \text{None}]$, `min_samples_leaf` $\in [1, 2]$.

### 3.2 Gradient Boosting Regressor (`GradientBoostingRegressor`)
- Sequentially builds shallow trees minimizing Mean Squared Error (MSE) loss along pseudo-residual gradients.
- Captures subtle non-linear district interactions.
- Search space: `n_estimators` $\in [100, 150, 200]$, `learning_rate` $\in [0.03, 0.04, 0.05]$, `max_depth` $\in [3, 4]$, `subsample` $\in [0.8, 0.85]$.

---

## 4. Expanding-Window Cross-Validation & Test Isolation

To select optimal hyperparameters without contaminating the held-out test set (2016–2017), an expanding-window validation scheme is executed inside 2011–2015:

```
Fold 1: Train [2011–2013]  →  Validate [2014]
Fold 2: Train [2011–2014]  →  Validate [2015]
---------------------------------------------
Final Evaluation: Fit on [2011–2015]  →  Held-Out Test [2016–2017]
```

---

## 5. Model Acceptance Decision Matrix

| Condition | Status Decision | Deployment Status | UI Action |
| :--- | :--- | :--- | :--- |
| $\text{MAE}_{\text{ML}} < \text{MAE}_{\text{baseline}}$ and $R^2 > 0$ | **`ACCEPTED`** | `VALIDATED_CANDIDATE` | Enable Forecasting Simulator with Uncertainty Bounds |
| $\text{MAE}_{\text{ML}} \ge \text{MAE}_{\text{baseline}}$ | **`BASELINE_PREFERRED`** | `BASELINE_PRODUCTION` | Recommend Statistical Baseline; Provide ML as experimental |
| High zero-inflation / localized coverage | **`EXPERIMENTAL_ONLY`** | `NOT_DEPLOYED` | Analytics supported; forecasting disabled |

This methodology guarantees that ML is only promoted when it delivers demonstrable, out-of-time predictive accuracy over transparent statistical baselines.
