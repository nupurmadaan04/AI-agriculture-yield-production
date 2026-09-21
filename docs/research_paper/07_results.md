# 6. Empirical Results & Governed Strategy Analysis

### 6.1 Multi-Crop Governed Strategy Results

Table 1 summarizes the empirical performance and governed deployment status across all 14 evaluated commodities under four-fold expanding walk-forward temporal cross-validation (origins 2014–2017).

```
+---------------------------------------------------------------------------------------------------------------------------------------+
|                                    TABLE 1: MULTI-CROP EMPIRICAL RESULTS & GOVERNED STRATEGIES                                        |
+---------------------+--------------------------+-----------------------+---------------+------------+-----------+------------+--------+
| Crop Commodity      | Governed Status          | Deployed Strategy     | Strategy MAE  | Gain vs.   | Win Rate  | Worst Fold | ML CV  |
|                     |                          |                       | (kg/ha)       | Base (%)   | (%)       | Loss (%)   |        |
+---------------------+--------------------------+-----------------------+---------------+------------+-----------+------------+--------+
| **Oilseeds**        | `PRODUCTION_READY`       | Historical ML (RF)    | 549.67        | **+12.79%**| **75.0%** | -2.34%     | 0.4391 |
| **Sugarcane**       | `CONDITIONAL_PRODUCTION` | Governed GBDT (+Clip) | 1,467.97      | **+1.19%** | 50.0%     | -9.92%     | 0.1115 |
| **Chickpea**        | `BASELINE_PRODUCTION`    | Hist. District Mean   | 260.35        | 0.00%      | Baseline  | N/A        | 0.2016 |
| **Kharif Sorghum**  | `BASELINE_PRODUCTION`    | Hist. District Mean   | 294.65        | 0.00%      | Baseline  | N/A        | 0.1256 |
| **Minor Pulses**    | `BASELINE_PRODUCTION`    | Hist. District Mean   | 345.16        | 0.00%      | Baseline  | N/A        | 0.1869 |
| **Maize**           | `BASELINE_PRODUCTION`    | Hist. District Mean   | 638.38        | 0.00%      | Baseline  | N/A        | 0.1349 |
| **Wheat**           | `BASELINE_PRODUCTION`    | Hist. District Mean   | 381.65        | 0.00%      | Baseline  | N/A        | 0.2371 |
| **Rice**            | `BASELINE_PRODUCTION`    | Hist. District Mean   | 310.28        | 0.00%      | Baseline  | N/A        | 0.1412 |
| **Sesamum**         | `BASELINE_PRODUCTION`    | Hist. District Mean   | 137.44        | 0.00%      | Baseline  | N/A        | 0.1094 |
| **Pigeonpea**       | `BASELINE_PRODUCTION`    | Hist. District Mean   | 282.35        | 0.00%      | Baseline  | N/A        | 0.1531 |
| **Rapeseed/Must.**  | `BASELINE_PRODUCTION`    | Hist. District Mean   | 193.33        | 0.00%      | Baseline  | N/A        | 0.1053 |
| **Groundnut**       | `BASELINE_PRODUCTION`    | Hist. District Mean   | 287.40        | 0.00%      | Baseline  | N/A        | 0.0778 |
| **Sorghum**         | `BASELINE_PRODUCTION`    | Hist. District Mean   | 264.02        | 0.00%      | Baseline  | N/A        | 0.1343 |
| **Pearl Millet**    | `BASELINE_PRODUCTION`    | Hist. District Mean   | 260.24        | 0.00%      | Baseline  | N/A        | 0.0285 |
+---------------------+--------------------------+-----------------------+---------------+------------+-----------+------------+--------+
```

*Note: All gains reflect Mean Absolute Error (MAE) reduction relative to the optimal historical statistical baseline. Baseline-certified commodities utilize Historical District Mean persistence as their primary forecast strategy.*

---

### 6.2 Oilseeds Case Study: Unconstrained Production Machine Learning

Oilseeds emerged as the sole commodity fulfilling all statistical criteria for unconstrained machine learning deployment:
- **Algorithm**: `RandomForestRegressor` (150 trees, max depth 12).
- **Temporal Consistency**: Outperformed historical district mean persistence in **3 out of 4 walk-forward folds (75.0% win-rate)**.
- **Error Reduction**: Achieved a **+12.79% mean MAE improvement** (and +5.18% median gain) over the baseline across 1,229 test observations.
- **Regime Stability**: During the extreme 2015–2016 climate anomalies, worst-fold degradation was tightly bounded at **-2.34%**, exhibiting no catastrophic failure.
- **Residual Distribution**: Mean residual across all folds is centered close to zero with symmetric distribution, confirming absence of systematic bias.

---

### 6.3 Sugarcane Case Study: Resolving the Dual-Metric Discrepancy

Sugarcane represents an instructive case study in the necessity of operational model governance:

1. **The Raw Unclipped GBDT Failure (-1.60% Loss)**:
   When evaluated without operational constraints, Gradient Boosted Decision Trees exhibited severe split sensitivity. In Fold 2 (evaluating the severe 2015 drought), unconstrained decision trees severely over-predicted yields in drought-affected Western Maharashtra and Uttar Pradesh districts (-10.19% fold loss). Consequently, the unclipped model averaged a **-1.60% net degradation** against historical district mean persistence across all four folds, causing it to fail unconstrained production certification.
2. **Governed Strategy with 3-$\sigma$ Variance Clipping (+1.19% Gain)**:
   Under the governed production strategy, predictions exceeding $3\sigma$ of the district's historical yield distribution automatically trigger fallback to the historical district mean. This fail-safe bounded the 2015 drought tail error, enabling the governed strategy to win Folds 3 (+9.75%) and 4 (+8.44%), achieving an aggregate **+1.19% MAE gain** over baseline (Strategy MAE 1,467.97 kg/ha vs Baseline MAE 1,485.70 kg/ha).
3. **Resolution of Historical Documentation Inconsistencies**:
   Early repository notes cited `+5.62%` for Sugarcane (transposed from Rapeseed & Mustard stability tables or exploratory single-fold tests). This audit definitively establishes **+1.19%** as the authoritative, verified walk-forward metric for the governed Sugarcane strategy.

---

### 6.4 The Role of Statistical Baselines in Scientific Governance

A critical finding of this study is that machine learning is **not universally superior** to simple statistical baselines. For 12 out of 14 commodities (including staple grains such as Rice and Wheat), Historical District Mean persistence achieved lower or statistically indistinguishable error compared to complex non-linear models. 

Rather than viewing baseline selection as a failure of machine learning, the framework treats baseline certification as a **governance victory**: it prevents deploying complex, computationally intensive models where simple, transparent, and robust historical means deliver superior operational stability.

---

### 6.5 Pre-Season Exogenous Feature Ablation: A Negative Result

In Day 22, a five-tier feature ablation tournament tested whether pre-season weather features (rainfall anomalies, temperature extremes, seasonal drought indices) improved forecast accuracy across all 14 commodities.

- **Empirical Finding**: Across all 14 crops, adding pre-season weather features yielded **`NO_MEANINGFUL_GAIN`**.
- **Ablation Preference**: The historical-only feature set ($EXP-22A$) was universally preferred in all 14 commodity evaluations.
- **Scientific Interpretation**: Within the evaluated dataset, pre-season lead times, district spatial aggregations, and tree model class, pre-season weather features provided negligible incremental predictive signal over historical autoregressive lags. This does not suggest that weather has no biological impact on crop growth, but demonstrates that pre-season district weather aggregations offer insufficient signal to improve operational pre-sowing forecasts.
