# 4. Methodology & Pipeline Architecture

### 4.1 End-to-End System Architecture

The platform architecture implements a rigorous, leakage-free operational pipeline:

```
+---------------------------------------------------------------------------------------------------+
|                                  END-TO-END PIPELINE ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------+
  [Raw Sources]            ICRISAT DLD (1966-2017) + IMD Meteorological Grids
        |
        v
  [Data Ingestion]         Schema standardizer, unit converter, administrative name harmonizer
        |
        v
  [Canonical Panel]        agricultural_panel.csv (71,601 records across 29 crops)
        |
        v
  [Data Quality & Leakage] Zero concurrent production in features; chronological shift(1) lags
        |
        v
  [Feature Engineering]    Lag1 yield, Roll3 mean yield, district area share, pre-season weather
        |
        v
  [Temporal Validation]    4-fold expanding walk-forward tournament (origins 2014, 2015, 2016, 2017)
        |
        v
  [Model Selection]        Tournament selection (Fold win-rate >= 50%, mean MAE gain > 0)
        |
        v
  [Strategy Registry]      forecast_strategy_registry.json (PRODUCTION, CONDITIONAL, BASELINE)
        |
        v
  [Forecast Serving API]   FastAPI runtime with certification guards and 3-sigma variance clipping
        |                  |                          |                          |
        v                  v                          v                          v
  [XAI Engine]      [Uncertainty]              [Provenance]               [Monitoring]
  Marginal Ref.     Empirical P10-P90          SHA-256 Request            PSI Covariate Drift &
  Perturbation      Tree Dispersion            Lineage Fingerprint        Post-Harvest Bias
        |                  |                          |                          |
        +------------------+--------------------------+--------------------------+
                                       |
                                       v
                             [Decision Workspace]
             Inspectable evidence briefs & bounded what-if scenario simulations
```

### 4.2 Feature Construction & Mathematical Formulations

To satisfy the pre-season forecasting constraint, all predictive features are constructed strictly using observations available prior to the planting of harvest year $t$:

1. **Autoregressive Lag-1 Yield ($y_{i, t-1}$)**:
   $$y_{i, t-1} = \text{Yield of district } i \text{ in year } t-1$$
   Captures recent local soil productivity, technological adoption, and base agronomic performance.
2. **Three-Year Rolling Mean Yield ($\bar{y}_{i, t-1:t-3}$)**:
   $$\bar{y}_{i, t-1:t-3} = \frac{1}{3} \sum_{k=1}^{3} y_{i, t-k}$$
   Smooths inter-annual climate volatility to establish the medium-term district productivity baseline.
3. **District Historical Long-Term Mean ($\mu_{i, <t}$)**:
   $$\mu_{i, <t} = \frac{1}{|H_{i, <t}|} \sum_{\tau \in H_{i, <t}} y_{i, \tau}$$
   Computed strictly over historical training years $\tau < t$.
4. **Cultivated Crop Area ($A_{i, t}$)**:
   Pre-season recorded or intention area in hectares, normalized relative to total district cropped area:
   $$\text{Area Share}_{i, t} = \frac{A_{i, t}}{\sum_{c \in \text{Crops}} A_{i, c, t}}$$

### 4.3 Zero-Leakage Invariants

To eliminate data leakage, three mathematical invariants are enforced across the pipeline:
- **Invariant 1 (Strict Temporal Masking)**: No record with year $\tau \ge t_{\text{forecast}}$ is accessible to feature scalers, imputers, or model training sets.
- **Invariant 2 (Algebraic Identity Exclusion)**: Contemporaneous harvest production $P_{i, t}$ is strictly prohibited from entering any feature vector.
- **Invariant 3 (Out-of-Sample Preprocessing)**: All scaling transformations:
  $$z = \frac{x - \hat{\mu}_{\text{train}}}{\hat{\sigma}_{\text{train}}}$$
  derive statistical parameters ($\hat{\mu}, \hat{\sigma}$) strictly from the training slice and apply them statically to test folds.
