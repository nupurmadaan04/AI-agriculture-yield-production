# Day 21: Feature Predictive Stability & Timing Diagnostics

## 1. Feature Stability Across Walk-Forward Folds

To evaluate whether candidate features maintain consistent predictive utility over time, fold-by-fold feature rankings and importances were tracked across all 4 walk-forward validation splits ($K=4$).

The **Feature Stability Score** ($S_{\text{feat}}$) is formulated as:
$$S_{\text{feat}} = \max\left(0, \min\left(1, 1 - \frac{\text{std}(\text{rank})}{\max(1, K-1)}\right)\right)$$

Where:
- $\text{std}(\text{rank})$ is the sample standard deviation of the feature's importance rank across folds.
- $K = 4$ is the number of walk-forward validation origins.
- $S_{\text{feat}} = 1.0$ indicates perfect rank invariance across all temporal test horizons.

### Representative Feature Stability Summary (Chickpea)
| Feature | Model | Mean Importance | Std Importance | Mean Rank | Rank Variance | Stability Score ($S_{\text{feat}}$) | Interpretative Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `yield_lag_1` | GradientBoosting | 42.8% | 0.042 | #1.0 | 0.00 | **1.000** | Primary predictive contributor (lag 1) |
| `yield_rolling_3yr_mean` | GradientBoosting | 28.4% | 0.035 | #2.0 | 0.00 | **1.000** | Medium-term baseline signal |
| `area_lag_1` | GradientBoosting | 14.2% | 0.021 | #3.2 | 0.25 | **0.833** | Cultivation scale proxy |
| `yield_lag_2` | GradientBoosting | 8.1% | 0.018 | #4.0 | 0.00 | **1.000** | Second-order temporal lag |
| `yield_rolling_3yr_std` | GradientBoosting | 4.5% | 0.011 | #5.0 | 0.00 | **1.000** | Historical yield variance |
| `area_rolling_3yr_mean` | GradientBoosting | 2.0% | 0.008 | #6.0 | 0.00 | **1.000** | Area trend regularizer |

*Scientific Note: Feature importance measures predictive contribution within the fitted model, not physical causality.*

---

## 2. Feature Timing & Anti-Leakage Audit

Every candidate feature was audited for observation timing relative to the pre-season forecasting window (prior to sowing):

| Feature | Observation Time | Available Pre-Season? | Fold Safe? | Timing Status | Audit Notes |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `yield_lag_1` | Previous season harvest ($t-1$) | **YES** | **YES** | `SAFE` | Verified strictly lagged with $t-1$ index |
| `yield_lag_2` | Harvest 2 seasons prior ($t-2$) | **YES** | **YES** | `SAFE` | Verified strictly lagged with $t-2$ index |
| `yield_rolling_3yr_mean` | Trailing 3 seasons ($t-1, t-2, t-3$) | **YES** | **YES** | `SAFE` | Excludes target year $t$ |
| `yield_rolling_3yr_std` | Trailing 3 seasons ($t-1, t-2, t-3$) | **YES** | **YES** | `SAFE` | Excludes target year $t$ |
| `area_lag_1` | Previous season planted area ($t-1$) | **YES** | **YES** | `SAFE` | Verified strictly lagged with $t-1$ index |
| `area_rolling_3yr_mean` | Trailing 3 seasons ($t-1, t-2, t-3$) | **YES** | **YES** | `SAFE` | Excludes target year $t$ |
| `district_encoded` | Pre-season geographic metadata | **YES** | **YES** | `SAFE` | Categorical static index |
| `spatial_cluster_id` | Unsupervised k-means clustering | **CONDITIONAL** | **NO** | `UNSAFE` | If fit across entire panel, leaks global spatial variance across folds. Prohibited in production. |

---

## 3. Plausibility of Additional Pre-Season Covariates

Diagnostic findings demonstrate that purely autoregressive lag models encounter a strict information barrier ($R^2 \approx 0.35-0.65$) across rainfed commodities. Adding the following pre-season covariates is scientifically plausible to improve future forecast skill:
1. **Pre-Monsoon Soil Moisture Indices** (Root-zone saturation prior to sowing)
2. **Standardized Precipitation Evapotranspiration Index (SPEI)** (Sowing window drought severity)
3. **Monsoon Onset Date Anomaly** (Days delay in Kharif monsoon arrival)
4. **Early Season MODIS NDVI / EVI Vigor** (Emergence stage vegetation vigor)
