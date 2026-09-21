# Figure 1: End-to-End System Architecture

```
+---------------------------------------------------------------------------------------------------------+
|                                FIGURE 1: END-TO-END SYSTEM ARCHITECTURE                                 |
+---------------------------------------------------------------------------------------------------------+

  [RAW SOURCES]
     |-- ICRISAT District Level Database (1966-2017) [Directly Ingested]
     |-- IMD District Weather Grids [Directly Ingested]
     +-- Directorate of Economics & Statistics (DES) [Directly Ingested]
        |
        v
  [DATA HARMONIZATION & INGESTION]
     |-- Standardize district spelling across 1966 administrative boundaries
     |-- Convert units to gross hectares (ha), metric tonnes (t), yield (kg/ha)
     +-- Zero duplicate keys under (crop, state, district, year)
        |
        v
  [CANONICAL PANEL: agricultural_panel.csv]
     +-- 71,601 records | 29 crops | 20 states | 311 districts | Years: 2010-2017
        |
        v
  [LEAKAGE-SAFE FEATURE EXTRACTION]
     |-- Shift(1) autoregressive yield lag (y_{t-1})
     |-- 3-year rolling mean yield (Roll3)
     |-- Pre-season cultivated crop area share
     +-- Strictly zero concurrent harvest production in features
        |
        v
  [TEMPORAL WALK-FORWARD TOURNAMENT]
     |-- 4 Expanding origins: T in {2014, 2015, 2016, 2017}
     |-- Train on all years < T; Test strictly on year T
     +-- Evaluate RandomForest, GradientBoosting, and HistoricalDistrictMean
        |
        v
  [STRATEGY GOVERNANCE GATE]
     |-- Fold win rate >= 75% & mean gain > 0   --> PRODUCTION_READY (Oilseeds)
     |-- Fold win rate >= 50% & drought risk    --> CONDITIONAL_PRODUCTION (Sugarcane + 3-sigma Clip)
     +-- Fold win rate < 50% or baseline superior --> BASELINE_PRODUCTION (12 crops, Rice, Wheat)
        |
        v
  [FORECAST SERVING RUNTIME (FastAPI)]
     |-- Certification Guard: Rejects uncertified query combinations
     |-- 3-Sigma Variance Clipping: Mitigates out-of-distribution extremes
     +-- Provenance Service: Generates SHA-256 digital execution signature
        |
        v
  [DECISION WORKSPACE & OBSERVABILITY]
     |-- Multi-evidence Decision Briefs (Forecast + Baseline + Attributions)
     |-- Empirical P10-P90 ensemble dispersion spread
     |-- PSI covariate drift monitoring & post-harvest error decomposition
     +-- Bounded [SCENARIO] what-if parameter simulations
```

**Interpretation**: This figure traces the complete data and execution flow from raw public agricultural records to governed decision support, emphasizing leakage-safe feature transformations, temporal walk-forward tournament selection, and fail-safe strategy routing.
