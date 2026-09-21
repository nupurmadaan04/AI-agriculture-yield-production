# Day 31: Decision Brief Specification & 9-Dimension Architecture

## 1. Document Objective

The **Agricultural Decision Brief** serves as the primary artifact delivered to agricultural economists, district planning officers, policy analysts, and commodity traders. It translates raw multi-stage analytical outputs into an inspectable, transparent, and non-causal decision document.

---

## 2. The 9 Core Brief Dimensions

### Dimension 1: Executive Summary
- **Context**: Geographic scope, administrative entity, and planning horizon.
- **Forecast**: Certified pre-season point estimate with confidence categorization.
- **Certainty Tier**: `HIGH_CONFIDENCE`, `MODERATE_CONFIDENCE`, or `BASELINE_BENCHMARK`.
- **Top Signals**: Key empirical and monitoring indicators synthesized into concise bullet points.

### Dimension 2: Context & Scope
- **Geographic Unit**: State and District.
- **Commodity**: Selected agricultural crop.
- **Planning Horizon**: e.g., Pre-Season ($t$), Next Harvest ($t+1$).
- **Scale of Operation**: Regional cultivated land area (in $1,000\text{ ha}$).

### Dimension 3: Governed Forecast Summary
- **Predicted Yield**: Point estimate in $\text{kg/ha}$.
- **Strategy Tier**: `PRODUCTION_READY`, `CONDITIONAL_PRODUCTION`, or `BASELINE_PRODUCTION`.
- **Model Name & Version**: Crop-specific certified regressor or persistence baseline.
- **Cryptographic Lineage**: SHA-256 provenance hash verifying input data and pipeline weights.

### Dimension 4: Historical Baseline Comparison
- **Empirical Moments**: Sample count ($N$), mean, median, standard deviation, minimum, and maximum yields.
- **Historical Trajectory**: Longitudinal trend slope ($\text{kg/ha/year}$) computed via linear regression over authentic panel observations.
- **Recent Harvest Points**: Table of the last 5 observed harvest records.

### Dimension 5: Model & Validation Evidence
- **Validation Protocol**: 4-Fold Expanding Walk-Forward Cross-Validation (2014–2017).
- **Error Metrics**: Walk-forward Test MAE, RMSE, and $R^2$ scores.
- **Win Rate & Relative Gain**: Fold win rate (%) and percentage improvement over historical persistence baseline.
- **Legacy Benchmarks**: For Rice, includes the certified academic benchmark note ($R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$).

### Dimension 6: Uncertainty & Model Dispersion
- **Empirical P10–P90 Spread**: Tree-ensemble dispersion across estimator predictions.
- **Relative Spread**: Spread percentage relative to predicted yield.
- **Mandatory Disclaimer**:
  > *"This range represents empirical ensemble spread across walk-forward estimator predictions and is not a formal distribution-free confidence interval."*

### Dimension 7: Explainability & Model Attribution
- **Feature Importance / Tree SHAP**: Top positive and negative feature anchors decomposed via Shapley values.
- **Mathematical Framing**: Explicitly stated as model sensitivity and dependency within trained feature space, avoiding biological causal claims.

### Dimension 8: Operational Monitoring & Health
- **Population Stability Index (PSI)**: Quantifies feature and prediction distribution drift relative to reference baseline.
- **Runtime Health**: Operational error rate and request success telemetry.
- **Post-Outcome Evaluation**: Directional signed bias ($\text{forecast} - \text{observed}$) when post-harvest data is published. Marked `EVALUATION_UNAVAILABLE` for unharvested future horizons.

### Dimension 9: Decision Considerations, Trade-Offs & Limitations
- **Scenario Options**: Counterfactual simulations clearly labeled with `is_simulated: true` and `semantic_classification: DERIVED`.
- **Trade-Off Analysis**: Friction, resource efficiency, and sensitivity trade-offs.
- **Assumptions & Limitations**: Explicit enumeration of dataset scope, temporal boundaries, and non-causal constraints.

---

## 3. Export Formats

The brief is natively rendered in the web dashboard and exportable in two standard formats:
1. **JSON (`/api/decision/brief`)**: Full machine-readable payload validated against Pydantic schema `DecisionBrief`.
2. **Markdown (`/api/reports/generate`)**: Formatted GitHub-style Markdown report with alerts, tables, and provenance hashes for archival.
