# Master Model Card: Multi-Crop Agricultural Forecast Suite

## 1. Model Details & Family

- **Model Family**: Multi-Crop Agricultural Forecast Suite (`AGRI_FORECAST_1.0`)
- **Model Version**: 1.0.0 (Governed Production Architecture)
- **Frameworks**: Scikit-learn 1.6.1, Python 3.11.9
- **Primary Artifacts**:
  - `Models/multicrop/forecast_strategy_registry.json` (SHA-256: `8974a7ee4d1ba24ddef9e1a03b46ac129fa05e2ffd1b95c624a2f0843f242e46`)
  - `Models/forecasting_pipeline.pkl` (SHA-256: `5d64b8f896f5fb9cb2deb4faa083edb3155355c049ad94eb54dc74f9c4e36086`)
  - `Datasets/metadata/final_model_certification.csv` (SHA-256: `be396e6fd112a8e68c2c7366223f1db1892cd202baea6173b3f9d24e1b596517`)

---

## 2. Certified Strategy States & Commodity Allocations

The platform operates under a strict three-tier evidence governance matrix:

```
+---------------------------------------------------------------------------------------------------------+
|                                    MODEL GOVERNANCE ALLOCATIONS                                         |
+--------------------------+-----------------------+-------------------+------------------+---------------+
| Commodity                | Governed Status       | Primary Strategy  | Deployed MAE     | Relative Gain |
+--------------------------+-----------------------+-------------------+------------------+---------------+
| **Oilseeds**             | `PRODUCTION_READY`    | RF Regressor      | 549.67 kg/ha     | **+10.85%**   |
| **Sugarcane**            | `CONDITIONAL_PROD`    | GBDT (+3σ Clip)   | 1,467.97 kg/ha   | **+1.19%**    |
| **Rice**                 | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 310.28 kg/ha     | Baseline      |
| **Wheat**                | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 381.65 kg/ha     | Baseline      |
| **Chickpea**             | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 260.35 kg/ha     | Baseline      |
| **Kharif Sorghum**       | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 294.65 kg/ha     | Baseline      |
| **Minor Pulses**         | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 345.16 kg/ha     | Baseline      |
| **Maize**                | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 638.38 kg/ha     | Baseline      |
| **Sesamum**              | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 137.44 kg/ha     | Baseline      |
| **Pigeonpea**            | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 282.35 kg/ha     | Baseline      |
| **Rapeseed & Mustard**   | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 193.33 kg/ha     | Baseline      |
| **Groundnut**            | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 287.40 kg/ha     | Baseline      |
| **Sorghum**              | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 264.02 kg/ha     | Baseline      |
| **Pearl Millet**         | `BASELINE_PRODUCTION` | Hist. Dist. Mean  | 260.24 kg/ha     | Baseline      |
+--------------------------+-----------------------+-------------------+------------------+---------------+
```

---

## 3. Key Empirical Benchmarks & Metric Distinctions

1. **Oilseeds Evidence**:
   - Primary Strategy: `RandomForestRegressor` (150 trees, max depth 12).
   - Under 4-fold expanding walk-forward validation (origins 2014–2017), achieves **+12.79% mean gain** (Day 21 unweighted fold mean) and **+10.85% aggregate gain** (Day 25/36 pooled certification table) with a **75.0% fold win-rate**.
2. **Sugarcane Dual-Metric Resolution**:
   - **Authoritative Governed Result**: **+1.19% MAE gain** over baseline (Strategy MAE 1,467.97 kg/ha vs Baseline MAE 1,485.70 kg/ha; 50% win rate).
   - **Raw Unconstrained GBDT**: Degraded during the 2015 drought (-10.19% loss in Fold 2), yielding an overall **-1.60% loss**.
   - **Historical Documentation Context**: Early draft tables cited `+5.62%` and `9,469.76 kg/ha` (transposed from Rapeseed & Mustard metrics or single-fold tests). This audit definitively establishes **+1.19%** as the governed production metric.
3. **Legacy Single-Crop Rice Benchmark**:
   - Model: Temporal Exogenous Random Forest Forecaster.
   - Evaluated on fixed 2016–2017 holdout (618 records): $R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$, $\text{RMSE} = 513.11\text{ kg/ha}$, $\text{MAPE} = 18.04\%$.
   - In multi-crop expanding walk-forward tournaments, Rice defaults to Historical District Mean due to higher stability across drought regimes.

---

## 4. Input Features & Target Outputs

- **Input Features**:
  1. `yield_lag_1` ($y_{t-1}$): Prior year district yield (kg/ha).
  2. `yield_lag_2` ($y_{t-2}$): Two-year prior district yield (kg/ha).
  3. `yield_rolling_3yr_mean` ($\bar{y}_{t-1:t-3}$): Medium-term baseline (kg/ha).
  4. `area_lag_1` ($A_{t-1}$): Prior year gross cultivated area (ha).
  5. `state_encoded`: Categorical state numerical identifier.
  6. `year`: Calendar harvest year.
- **Target Output**: `yield_kg_ha` (Forecasted crop productivity in kilograms per hectare).
- **Prohibited Features**: Contemporaneous production $P_t$, in-season harvest indices, post-sowing satellite telemetry.

---

## 5. Intended Use & Out-of-Scope Boundaries

### Intended Use
- Regional pre-season food security planning.
- District-level buffer stock and procurement logistics.
- Non-prescriptive decision support briefs for agricultural economists.

### Out-of-Scope / Prohibited Uses
- Precision farm-level micro-management or plot-scale fertilizer recommendations.
- Autonomous policy triggers or automated subsidy disbursements.
- In-season crop yield updates (the framework is strictly calibrated for pre-sowing lead times).
- Causal inference: model attributions reflect mathematical feature sensitivity, not agronomic causality.

---

## 6. Hardware & Runtime Profile

- **Inference Runtime**: Sub-50ms $P95$ latency per forecast request on single-core x86_64 CPU.
- **Memory Footprint**: ~350 MB RSS for FastAPI server with all serialized models loaded in memory.
- **Fail-Safe Mechanism**: Mandatory fallback to Historical District Mean if inputs fall outside historical bounds or lag records are missing.
