# Day 22: Agronomic Feature Engineering & Mathematical Formulations

## 1. Feature Formulations
The 10 engineered exogenous features are derived through agronomically validated transformations:

### A. Pre-Season Precipitation Features
1. **`preseason_rainfall_total`**:
   $$\text{Precip}_{\text{preseason}} = \sum_{m=\text{Jan}}^{\text{May}} R_m \quad (\text{mm})$$

2. **`preseason_rainfall_anomaly`**:
   $$\Delta R_{\text{anomaly}} = \left( \frac{\text{Precip}_{\text{preseason}} - \bar{R}_{30\text{yr}}}{\bar{R}_{30\text{yr}}} \right) \times 100 \quad (\%)$$

### B. Pre-Season Thermal Indicators
3. **`preseason_temp_mean`**:
   $$\bar{T}_{\text{preseason}} = \frac{1}{3} \sum_{m=\text{Mar}}^{\text{May}} T_{m,\text{mean}} \quad (°\text{C})$$

4. **`preseason_temp_max`**:
   $$T_{\text{max,preseason}} = \max_{m \in \{\text{Mar,Apr,May}\}} T_{m,\text{max}} \quad (°\text{C})$$

5. **`preseason_temp_anomaly`**:
   $$\Delta T = \bar{T}_{\text{preseason}} - \bar{T}_{\text{hist,baseline}} \quad (°\text{C})$$

### C. Moisture, Dry Spells & Aridity
6. **`preseason_soil_moisture`**:
   $$SM_{\text{topsoil}} \in [0.0, 1.0] \quad (\text{Normalized Saturation})$$

7. **`preseason_dry_spell_days`**:
   $$D_{\text{dry}} = \text{Consecutive rainless days } (R_d < 1.0\,\text{mm}) \text{ in Jan–May}$$

8. **`preseason_aridity_index` (SPEI Proxy)**:
   $$\text{SPEI}_{\text{proxy}} = \frac{\text{Precip}_{\text{preseason}} - \text{PET}_{\text{preseason}}}{\text{PET}_{\text{preseason}} + \epsilon}$$

### D. Carryover Hydrology & Infrastructure
9. **`rainfall_lag1_total`**: Full trailing-year annual rainfall ($t-1$).
10. **`irrigation_ratio_lag1`**: Irrigated area fraction ($\text{GIA} / \text{GCA}$) in year $t-1$.
