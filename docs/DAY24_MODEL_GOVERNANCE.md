# Day 24: Model Governance & Routing Directives

## 1. Governance Principles

Model governance in the Agricultural Decision Intelligence platform enforces that machine learning algorithms are only served in production when empirical temporal validation proves statistically significant superiority over historical persistence baselines.

```
       ┌─────────────────────────────────────────────────────────────┐
       │                Model Governance Hierarchy                   │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
   [Empirical Evidence Gate]                      [Pre-Inference Gate]
   • Walk-Forward Fold Win Rate >= 75%            • Crop Registration Match
   • MAE Improvement vs Baseline >= 5.0%          • District Geographical Coverage
   • Zero Lookahead Timing Compliance             • Model SHA-256 Hash Integrity
   • Non-Degenerate Bias Profile                  • Valid Observation Density
```

---

## 2. Policy Enactments by Certification Status

### `PRODUCTION_READY` (Oilseeds)
- **Primary Strategy**: Historical Pre-Season `RandomForestRegressor`
- **Operating Rule**: Serves primary ML pipeline. If district history contains fewer than 3 historical records, the router activates the **Sparse District Fallback** returning the Historical District Mean.
- **Evidence**: MAE 549.7 kg/ha vs Baseline 616.6 kg/ha (+10.8% gain), 75% fold win rate.

### `CONDITIONAL_PRODUCTION` (Sugarcane)
- **Primary Strategy**: Historical Pre-Season `GradientBoostingRegressor`
- **Operating Rule**: Serves primary ML with mandatory **3-$\sigma$ Variance Clipping**. Predictions exceeding $[\mu_{\text{district}} - 3\sigma, \mu_{\text{district}} + 3\sigma]$ are bounded to protect users against high-variance excursions.
- **Evidence**: MAE 9,469.8 kg/ha vs Baseline 10,033.4 kg/ha (+5.6% gain), 75% fold win rate.

### `BASELINE_PRODUCTION` (12 Crops)
- **Primary Strategy**: Historical District Mean Persistence
- **Operating Rule**: Served directly. If district history is sparse or missing, the router falls back to state historical averages or 3-year rolling district means.
- **Evidence**: ML models did not demonstrate temporal superiority (win rates < 50% or negative gain).
