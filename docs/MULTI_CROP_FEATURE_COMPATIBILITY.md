# Multi-Crop Feature & Pre-Season Forecasting Compatibility Audit

## 1. Feature Compatibility Matrix

This audit evaluates all candidate predictive features across observation timing, universal availability, target reconstruction risks, and pre-season forecasting validity.

| Feature Identifier | Source Type | Timing of Observation | Pre-Season Valid? | Leakage Risk Level | Classification | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `YIELD_LAG_1` ($y_{t-1}$) | Historical Panel | Prior Crop Season ($t-1$) | **YES** | **ZERO** | `UNIVERSAL` | **Primary Feature**: 1-year persistence baseline available for all crops. |
| `YIELD_ROLLING_3YR_MEAN` | Historical Panel | Prior Crop Seasons ($t-3..t-1$) | **YES** | **ZERO** | `UNIVERSAL` | **Primary Feature**: Smooth multi-year localized historical baseline. |
| `AREA_LAG_1` ($A_{t-1}$) | Historical Panel | Prior Crop Season ($t-1$) | **YES** | **ZERO** | `UNIVERSAL` | Safe proxy for district agricultural land allocation. |
| `ANNUAL_RAINFALL` | Climate / Meteo | Post-Harvest ($t$) | **NO** | **HIGH (Simultaneity)** | `POST_HARVEST_ONLY` | Inadmissible for pre-season forecasting; post-harvest benchmarking only. |
| `JUNE_SEPTEMBER_RAINFALL` | Climate / Meteo | Kharif Season ($t$) | **PARTIAL** | **MODERATE** | `SEASONAL_WEATHER` | Kharif mid-season monitoring only; inadmissible prior to June planting. |
| `TOTAL_NPK` ($N+P+K$) | Input Survey | Annual Harvest ($t$) | **NO** | **HIGH (Simultaneity)** | `POST_HARVEST_ONLY` | Reported retrospectively on annual district basis. |
| `CURRENT_YEAR_AREA` ($A_t$) | Observed Panel | Planting Survey ($t$) | **CONDITIONAL** | **LOW** | `EARLY_SEASON` | Admissible only if early sowing area surveys are available before yield formation. |
| `CURRENT_YEAR_PRODUCTION` ($P_t$) | Observed Panel | Harvest Survey ($t$) | **NO** | **CRITICAL (100% LEAKAGE)** | `LEAKAGE_RISK` | **Strictly Prohibited**: $\text{Yield} = \frac{P_t}{A_t}$ reconstructs target mathematically. |
| `SPATIAL_CLUSTER_ID` | Agro-Climatic | Static Baseline | **YES** | **ZERO** | `UNIVERSAL` | K-Means ($k=4$) geographic zones apply universally across India. |
| `DISTRICT_LAT_LON` | Geography | Static Geographic | **YES** | **ZERO** | `UNIVERSAL` | Latitude and longitude coordinates provide continuous spatial priors. |

---

## 2. Pre-Season Feature Invariance & Anti-Leakage Rules

> [!IMPORTANT]
> **Anti-Leakage Rule 1 (Target Simultaneity)**: Under no circumstances may concurrent harvest-year production ($P_t$) or concurrent harvest-year metrics be used to predict concurrent yield ($y_t$).

> [!IMPORTANT]
> **Anti-Leakage Rule 2 (Temporal Boundary)**: A model labeled as **Pre-Season Forecasting** must strictly consume features observed at or prior to $t_{\text{sowing}}$ (e.g. historical yield lags, long-term agro-climatic clusters, pre-sowing soil moisture).

---

## 3. Recommended Universal Pre-Season Feature Set

For future multi-crop predictive model training (Day 19), the scientifically verified universal feature set comprises:
1. `yield_lag_1`: Previous year district yield ($kg/ha$)
2. `yield_lag_2`: Two-year prior district yield ($kg/ha$)
3. `yield_rolling_3yr_mean`: 3-year historical district rolling average ($kg/ha$)
4. `area_lag_1`: Previous year district cultivated area ($ha$)
5. `spatial_cluster_id`: Regional agro-ecological cluster ($0..3$)
6. `state_encoded`: Categorical state fixed effects
7. `district_latitude`, `district_longitude`: Spatial coordinates
