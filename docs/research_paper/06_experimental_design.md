# 5. Experimental Design & Validation Protocols

To prevent the confounding of historical metrics, the platform explicitly defines and documents each empirical validation protocol separately.

---

### 5.1 Protocol Catalog & Specifications

```
+----------------------------------------------------------------------------------------------------+
|                                    VALIDATION PROTOCOL CATALOG                                     |
+----------------------------------------------------------------------------------------------------+
```

#### Protocol A: Multi-Crop Screening & Sufficiency Gate
- **Purpose**: Classify all 29 crops into modeling readiness categories.
- **Training Window**: 2010–2017 complete panel.
- **Test Window**: None (Cross-sectional and longitudinal sufficiency audit).
- **Population**: All 29 commodities across 311 districts.
- **Sufficiency Criteria**: Total observations $N \ge 100$, geographic breadth $\ge 10$ districts, temporal continuity $\ge 5$ consecutive years.
- **Result**: 14 crops classified as `MODEL_READY`; 15 crops classified as `ANALYTICS_READY` or `INSUFFICIENT_DATA`.
- **Limitation**: Evaluates data volume, not predictive signal.

#### Protocol B: Initial Crop-Specific Out-of-Time (OOT) Holdout
- **Purpose**: Initial single-split algorithm evaluation.
- **Training Window**: 2010–2015.
- **Test Window**: 2016–2017 (Fixed holdout).
- **Forecast Origin**: 2016.
- **Population**: 14 model-ready crops.
- **Models**: Random Forest Regressor, Gradient Boosting Regressor, Historical District Mean.
- **Primary Metric**: MAE, RMSE, $R^2$.
- **Limitation**: Fails to capture model vulnerability across earlier regime shocks (e.g. 2014–2015 droughts).

#### Protocol C: Multi-Origin Expanding Walk-Forward Validation
- **Purpose**: Detect temporal instability and drought regime failure across sequential historical origins.
- **Training Window**: Expanding historical slice: all years prior to origin year $T$.
- **Test Window**: Year $T$ exclusively, for $T \in \{2014, 2015, 2016, 2017\}$.
- **Forecast Origins**: 4 distinct origins ($T=2014, T=2015, T=2016, T=2017$).
- **Population**: 14 model-ready crops.
- **Models**: RandomForestRegressor, GradientBoostingRegressor, LinearTrend, HistoricalDistrictMean.
- **Primary Metric**: Fold-level MAE, fold win-rate vs. baseline (%), worst-fold degradation (%).
- **Limitation**: Restricted to four historical test origins by canonical panel temporal span.

#### Protocol D: Tournament Model Selection Gate
- **Purpose**: Authoritatively assign production governance categories.
- **Criteria**:
  - `ROBUST_ML`: Fold win-rate $\ge 75\%$, positive mean MAE gain, positive median gain, worst fold degradation $< 5\%$, feature timing verified SAFE.
  - `ML_WITH_CONDITIONS`: Win-rate $\ge 50\%$, positive mean gain, but exhibits regime sensitivity in drought folds.
  - `BASELINE_PREFERRED`: Win-rate $< 50\%$ or negative mean MAE improvement.
- **Result**: Only Oilseeds achieved `ROBUST_ML`; Sugarcane achieved `ML_WITH_CONDITIONS`; 12 crops designated `BASELINE_PREFERRED`.

#### Protocol E: Pre-Season Exogenous Feature Ablation
- **Purpose**: Evaluate incremental utility of pre-season weather features under Protocol C walk-forward testing.
- **Ablation Tiers**:
  - $EXP-22A$: Historical Only (Autoregressive Lags + Area)
  - $EXP-22B$: Historical + Pre-Season Rainfall
  - $EXP-22C$: Historical + Pre-Season Temperature
  - $EXP-22D$: Historical + Combined Pre-Season Weather
  - $EXP-22E$: Historical + All Exogenous Covariates
- **Population**: 14 model-ready crops across 4 walk-forward folds.
- **Primary Metric**: MAE change vs. $EXP-22A$, shock-year (2015/2016) resilience.
- **Result**: **NO MEANINGFUL GAIN** across all 14 crops; $EXP-22A$ remained universally preferred.

#### Protocol F: Governed Strategy Certification Audit
- **Purpose**: Evaluate production strategy router incorporating 3-$\sigma$ district variance clipping.
- **Protocol**: Protocol C walk-forward folds evaluated under active production routing logic with sparse fallback.
- **Result**: Formally certified Oilseeds (`PRODUCTION_READY`), Sugarcane (`CONDITIONAL_PRODUCTION`), and 12 baseline crops (`BASELINE_PRODUCTION`).

#### Protocol G: Legacy Single-Crop Rice Benchmark
- **Purpose**: Document original benchmark established during single-crop research phase.
- **Training Window**: 2010–2015 (1,851 records).
- **Test Window**: 2016–2017 (618 records).
- **Population**: Rice growing districts (2,469 records total).
- **Model**: Temporal Exogenous Random Forest Forecaster (150 trees, max depth 14).
- **Verified Metrics**: $R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$, $\text{RMSE} = 513.11\text{ kg/ha}$, $\text{MAPE} = 18.04\%$.
- **Artifact**: `Models/forecasting_model_metadata.json`.
