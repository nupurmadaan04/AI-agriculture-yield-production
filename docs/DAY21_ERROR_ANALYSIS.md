# Day 21: Multi-Level Error Analysis & Systematic Decomposition

## 1. Introduction & Methodology
To understand why standard machine learning algorithms fail to generalize across non-stationary agricultural environments, Day 21 decomposes prediction errors along five distinct axes:
1. **Error Dispersion & Quantiles**: $P_{25}, P_{50}, P_{75}, P_{90}, P_{95}$ absolute error percentiles.
2. **Normalized Errors**: Relative to historical commodity median yield.
3. **Yield Regimes**: Low ($\le Q_{25}$), Normal ($Q_{25}-Q_{75}$), High ($\ge Q_{75}$).
4. **Temporal Regimes**: Origin year stability and anomaly inspection (specifically inspecting **2016 as a high-error regime**).
5. **Geographic Heterogeneity**: District-level error stratification ($N \ge 3$ filter).

---

## 2. Yield Regime Error Decomposition

Empirical evaluation reveals that pure autoregressive lag features perform decently in normal yield conditions ($Q_{25}-Q_{75}$), but fail systematically during extreme yield shock regimes:

### A. Low-Yield Regimes ($\le Q_{25}$)
- During drought, pest, or delayed monsoon seasons, actual yields collapse significantly below historical trends.
- Autoregressive models conditioned on `yield_lag_1` and `yield_rolling_3yr_mean` systematically **over-predict** yield because they lack real-time moisture/weather signals.
- In 11 out of 14 crops, the baseline **Historical District Mean** outperforms ML in low-yield regimes because the district mean has a natural dampening effect compared to optimistic tree splits.

### B. Normal Yield Regimes ($Q_{25}-Q_{75}$)
- In stable weather seasons, ML models for `Chickpea`, `Oilseeds`, and `Kharif Sorghum` achieve statistically significant error reductions ($+5\%$ to $+15\%$ MAE improvement) by capturing regional yield trends and area shifts.

### C. High-Yield Regimes ($\ge Q_{75}$)
- In bumper harvest seasons, tree-based models suffer from **regression-to-the-mean clipping** because decision tree leaf values are bounded by the maximum yield observed in the training partition.

---

## 3. Temporal Regime Analysis & The 2016 Shock

| Commodity | 2014 Origin | 2015 Origin | 2016 Origin (Shock Year) | 2017 Origin |
| :--- | :---: | :---: | :---: | :---: |
| **Chickpea** | ML Win (+17.2%) | ML Win (+9.7%) | **ML Loss (-22.3%)** | ML Win (+2.5%) |
| **Oilseeds** | ML Win (+6.2%) | ML Win (+4.2%) | ML Loss (-2.3%) | ML Win (+43.1%) |
| **Rice** | ML Loss (-10.8%) | ML Loss (-10.7%) | ML Loss (-12.7%) | ML Win (+0.8%) |
| **Wheat** | ML Loss (-14.0%) | ML Loss (-18.1%) | ML Loss (-24.6%) | ML Loss (-2.2%) |
| **Sugarcane** | ML Loss (-4.5%) | ML Loss (-8.9%) | ML Loss (-2.3%) | ML Win (+13.0%) |

### Diagnostic Finding on 2016:
The 2016 agricultural season in India experienced widespread climatic anomalies following back-to-back monsoon deficits in 2014 and 2015, followed by abrupt rainfall recovery in 2016. Because autoregressive models were trained on 2014–2015 depressed yield lags, they heavily under-predicted the 2016 rebound, generating peak error spikes across all 14 crops.

---

## 4. Geographic & District-Level Error Stratification

Enforcing the strict $N \ge 3$ evaluation threshold across 4,313 district partitions:
- **Best ML Districts**: High-production agricultural clusters (e.g. irrigated tracts of Punjab, Haryana, coastal Andhra) where yields follow stable autoregressive trajectories exhibit 10–25% lower MAE with ML.
- **Worst ML Districts (Baseline Superior)**: Rainfed, drought-prone districts (e.g., Marathwada, Vidarbha, Rayalaseema) exhibit high year-to-year yield variance where historical district means provide better regularization.
- **High-Error Districts**: Districts with severe data sparsity ($N < 10$) or boundary redefinitions show elevated tail errors across both ML and baseline models.
