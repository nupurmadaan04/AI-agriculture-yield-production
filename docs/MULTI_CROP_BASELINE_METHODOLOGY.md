# Multi-Crop Baseline Forecasting Methodology & Benchmark Evaluation

## 1. Baseline Model Formulations

To establish rigorous, zero-leakage benchmarks before training complex machine learning models, 4 deterministic statistical baselines were implemented:

### 1.1 Naive 1-Year Persistence
$$ \hat{y}_{i, t} = y_{i, t-1} $$
Assumes district yield in harvest year $t$ equals its observed yield in year $t-1$. When $y_{i, t-1}$ is missing or zero, falls back to the district training mean $\bar{y}_{i, \text{train}}$.

### 1.2 Historical District Mean
$$ \hat{y}_{i, t} = \bar{y}_{i, \text{train}} = \frac{1}{|T_{\text{train}}|} \sum_{s \in T_{\text{train}}} y_{i, s} $$
Predicts the localized 6-year historical average (2010–2015) for district $i$.

### 1.3 Historical Crop Mean
$$ \hat{y}_{i, t} = \bar{y}_{\text{crop}, \text{train}} = \frac{1}{|N_{\text{train}}|} \sum_{j \in N_{\text{train}}} y_{j, s} $$
Predicts the national unweighted crop average across all training records.

### 1.4 Linear District Trend
$$ \hat{y}_{i, t} = \hat{\alpha}_i + \hat{\beta}_i t $$
Fits an Ordinary Least Squares (OLS) linear trend on historical district observations in $T_{\text{train}}$. Falls back to district mean if $|T_{\text{train}}| < 2$.

---

## 2. Evaluation Protocol & Results Summary

- **Chronological Split**: Train on 2010–2015 ($N_{\text{train}}$), Evaluate on 2016–2017 ($N_{\text{test}}$).
- **Zero Lookahead**: All parameters ($\bar{y}_{i}, \bar{y}_{\text{crop}}, \hat{\alpha}_i, \hat{\beta}_i$) fitted strictly on $t \le 2015$.

### Baseline Comparison Across Model Families (Average over all evaluated crops)

| Baseline Model | Mean MAE (kg/ha) | Mean RMSE (kg/ha) | Mean $R^2$ | Mean MAPE (%) | Mean SMAPE (%) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical District Mean** | **398.46** | **690.84** | **0.4215** | 172.73% | **28.74%** | **Strongest overall baseline** capturing localized soil & agro-climate |
| **Naive Persistence ($t-1$)**| 399.17 | 705.82 | 0.2941 | **117.44%** | 32.43% | Competitive in stable perennial/irrigated crops |
| **Linear District Trend** | 473.99 | 790.18 | 0.1208 | 193.22% | 37.52% | Susceptible to trend extrapolation error on short series |
| **Historical Crop Mean** | 638.52 | 993.81 | -0.0520 | 196.94% | 42.69% | Poor baseline demonstrating failure of national pooling |

---

## 3. Best Baseline Model by Crop (Out-of-Time Test Set 2016–2017)

| Crop | Best Baseline Model | Test Records | MAE (kg/ha) | RMSE (kg/ha) | $R^2$ | MAPE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wheat** | Naive Persistence ($t-1$) | 557 | **396.55** | 569.21 | **0.7556** | 21.26% |
| **Rice** | Historical District Mean | 577 | **367.49** | 549.33 | **0.6735** | 17.56% |
| **Castor** | Historical District Mean | 148 | **215.43** | 338.12 | **0.7723** | 39.04% |
| **Groundnut** | Historical District Mean | 414 | **302.18** | 465.81 | **0.6220** | 24.58% |
| **Rapeseed and Mustard** | Naive Persistence ($t-1$) | 439 | **224.95** | 344.15 | **0.6039** | 43.75% |
| **Finger Millet** | Historical District Mean | 142 | **315.39** | 437.28 | **0.6101** | 31.14% |
| **Sorghum** | Historical District Mean | 391 | **305.79** | 448.91 | **0.5490** | 33.51% |
| **Rabi Sorghum** | Historical District Mean | 134 | **379.55** | 511.02 | **0.5259** | 78.67% |
| **Kharif Sorghum** | Historical District Mean | 291 | **331.18** | 467.45 | **0.4501** | 35.31% |
| **Pearl Millet** | Historical District Mean | 298 | **329.83** | 482.90 | **0.4563** | 318.08% |
| **Maize** | Historical District Mean | 546 | **763.91** | 1084.22 | **0.4697** | 35.67% |
| **Sesamum** | Historical District Mean | 397 | **117.69** | 176.45 | **0.4574** | 73.53% |
| **Sunflower** | Historical District Mean | 139 | **264.92** | 398.71 | **0.4014** | 39.03% |
| **Barley** | Naive Persistence ($t-1$) | 196 | **521.96** | 741.03 | **0.3130** | 25.25% |
| **Chickpea** | Historical District Mean | 400 | **274.42** | 402.16 | **0.1277** | 29.83% |
| **Soyabean** | Historical District Mean | 277 | **287.57** | 382.49 | **0.1958** | 35.85% |
| **Minor Pulses** | Historical District Mean | 487 | **322.30** | 459.18 | **0.1951** | 30.29% |
| **Linseed** | Historical District Mean | 200 | **183.15** | 262.33 | **0.2116** | 43.14% |
| **Safflower** | Historical District Mean | 91 | **221.92** | 308.15 | **0.1650** | 56.74% |
| **Cotton** | Historical District Mean | 209 | **262.14** | 381.04 | **0.0335** | 2847.52% |
| **Pigeonpea** | Naive Persistence ($t-1$) | 370 | **299.15** | 438.12 | **-0.0525** | 31.31% |
| **Sugarcane** | Naive Persistence ($t-1$) | 344 | **1126.31** | 1845.20 | **0.5605** | 55.82% |
| **Oilseeds** | Naive Persistence ($t-1$) | 572 | **746.72** | 1142.08 | **0.7635** | 21.54% |
