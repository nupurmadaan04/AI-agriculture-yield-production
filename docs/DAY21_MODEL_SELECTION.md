# Day 21: Deterministic Multi-Crop Model Selection Rules & Audit

## 1. Deterministic Selection Rules

Day 21 strictly enforces deterministic selection criteria. A model is **never** upgraded merely because of a higher single-split $R^2$ or average MAE.

### Rule Hierarchy
1. **Rule 1: Robust ML (`ROBUST_ML`)**
   - Condition: Multi-origin win rate $\ge 75\%$, Mean MAE gain $> 0\%$, Median MAE gain $> 0\%$, Worst-fold degradation $\le 5\%$, and Feature Timing is `SAFE`.
   - Classification: **`Oilseeds`** (Win rate: 75.0%, Mean Gain: +12.79%, Median Gain: +5.18%, Worst fold: -2.34%).
2. **Rule 2: ML With Operating Conditions (`ML_WITH_CONDITIONS`)**
   - Condition: Multi-origin win rate $\ge 50\%$, Mean MAE gain $> 0\%$, Median MAE gain $> 0\%$, but worst-fold degradation $> 5\%$ or subject to regime shocks.
   - Classification: **`Chickpea`** (Win rate: 75.0%, Mean Gain: +2.31%, Worst fold: -22.30% in 2016) and **`Kharif Sorghum`** (Win rate: 50.0%, Mean Gain: +1.30%, Worst fold: -2.28%).
3. **Rule 3: Baseline Preferred (`BASELINE_PREFERRED`)**
   - Condition: Multi-origin win rate $< 50\%$ and negative average gain, where Historical District Mean or Naive Persistence consistently equals or outperforms ML.
   - Classification: **`Rice`**, **`Sesamum`**, **`Pigeonpea`**, **`Rapeseed and Mustard`**, **`Groundnut`**, **`Sorghum`**, **`Pearl Millet`** (7 Crops).
4. **Rule 4: Research Candidate (`RESEARCH_CANDIDATE`)**
   - Condition: Models exhibiting localized district advantage or 50% win rate, but negative overall gain or requiring exogenous weather/satellite features.
   - Classification: **`Minor Pulses`**, **`Maize`**, **`Wheat`**, **`Sugarcane`** (4 Crops).

---

## 2. Model Selection Matrix & Lineage Audit

| Crop | Day 19 Status | Day 20 Status | Day 21 Final Decision | Win Rate | Mean Gain | Worst Fold | Decision Basis |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Oilseeds** | `ACCEPTED` | `ROBUST_ACCEPTED` | **`ROBUST_ML`** | 75.0% | +12.79% | -2.34% | High win rate, positive median/mean gain, robust across origins |
| **Chickpea** | `ACCEPTED` | `ROBUST_ACCEPTED` | **`ML_WITH_CONDITIONS`** | 75.0% | +2.31% | -22.30% | High win rate, positive gain, sensitive to 2016 regime shock |
| **Kharif Sorghum** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`ML_WITH_CONDITIONS`** | 50.0% | +1.30% | -2.28% | Balanced win rate, positive gain, minor degradation |
| **Minor Pulses** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`RESEARCH_CANDIDATE`** | 50.0% | -5.75% | -21.48% | Localized signal in selected districts, negative average gain |
| **Maize** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`RESEARCH_CANDIDATE`** | 25.0% | -2.63% | -6.22% | Inconsistent win rate, needs weather/satellite covariates |
| **Wheat** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`RESEARCH_CANDIDATE`** | 25.0% | -14.73% | -24.63% | Severe tail volatility in winter temperature shocks |
| **Sugarcane** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`RESEARCH_CANDIDATE`** | 25.0% | -0.66% | -8.87% | Multi-year crop cycle requires specialized cumulative covariates |
| **Rice** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`BASELINE_PREFERRED`** | 25.0% | -8.39% | -12.72% | Statistical district mean outperforms pure lag ML |
| **Sesamum** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`BASELINE_PREFERRED`** | 25.0% | -1.19% | -3.79% | Statistical district mean outperforms pure lag ML |
| **Pigeonpea** | `ACCEPTED` | `BASELINE_PREFERRED` | **`BASELINE_PREFERRED`** | 0.0% | -5.68% | -8.98% | Historical baseline strictly dominates across all folds |
| **Rapeseed & Mustard** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`BASELINE_PREFERRED`** | 25.0% | -5.62% | -10.74% | Statistical baseline superior across temporal splits |
| **Groundnut** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`BASELINE_PREFERRED`** | 25.0% | -7.87% | -12.24% | Statistical baseline superior across temporal splits |
| **Sorghum** | `ACCEPTED` | `SPLIT_SENSITIVE` | **`BASELINE_PREFERRED`** | 25.0% | -3.50% | -8.15% | Statistical baseline superior across temporal splits |
| **Pearl Millet** | `ACCEPTED` | `BASELINE_PREFERRED` | **`BASELINE_PREFERRED`** | 0.0% | -6.04% | -8.72% | Historical baseline strictly dominates across all folds |
