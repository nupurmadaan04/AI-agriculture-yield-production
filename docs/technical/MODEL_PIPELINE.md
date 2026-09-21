# Model Pipeline Architecture

## 1. Estimator Specifications

1. **Random Forest Forecaster** (`RandomForestRegressor`):
   - 150 estimators, maximum depth 12–14, minimum samples per leaf 4.
   - Deployed for Oilseeds (`PRODUCTION_READY`) and legacy single-crop Rice research.
2. **Gradient Boosted Decision Trees** (`GradientBoostingRegressor`):
   - 100 boosting stages, learning rate 0.05, maximum depth 4.
   - Deployed for Sugarcane (`CONDITIONAL_PRODUCTION`) with mandatory 3-$\sigma$ district variance clipping.
3. **Statistical Baselines**:
   - **Historical District Mean**: Expanding historical mean of the target district ($\bar{y}_{\text{dist}, <t}$).
   - Deployed as the primary production strategy for 12 major commodities (Rice, Wheat, Chickpea, Maize, etc.).

---

## 2. Serialization & Weight Integrity

- Estimator weights, preprocessing scalers, and metadata dictionaries are serialized via standard Python `pickle` with protocol 4.
- Every model artifact is verified via cryptographic SHA-256 signatures upon service startup.
