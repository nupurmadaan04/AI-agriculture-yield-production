# Agricultural Decision Evidence Reports

## Overview
The **Agricultural Decision Evidence Report** is an exportable, multi-section decision-support artifact summarizing all empirical, predictive, and scenario evidence synthesized for a given agricultural region and year.

---

## 16-Section Structure
1. **Decision Context**: Entity, Crop, Year, Decision Horizon.
2. **Executive Summary**: Deterministic factual synthesis.
3. **Current Agricultural State**: Empirical panel yield and acreage.
4. **Historical Evidence**: Multi-year linear trajectory slope.
5. **Forecast Outlook**: Exogenous Random Forest forecast and prediction spread.
6. **Risk & Early Warning Signals**: Fused multi-window severity signals.
7. **Geospatial Evidence**: Spatial peer comparisons and within-state z-scores.
8. **Model Reliability**: Chronological out-of-time benchmark metrics ($R^2$, $\text{MAE}$, $\text{RMSE}$, $\text{MAPE}$).
9. **What Influenced the Model Prediction**: Day 13 XAI feature contribution breakdown.
10. **Available Scenario Options**: Pre-season allocation options.
11. **Scenario Trade-offs**: Acreage vs yield trade-offs.
12. **Recommended Analytical Priority**: Ranked analytical issues.
13. **Alternative Options & Robustness**: Sensitivity elasticity tiers (`ROBUST`, `MODERATELY ROBUST`, `SENSITIVE`).
14. **Key Limitations**: Domain boundaries and non-causal interpretation.
15. **Data & Model Provenance**: Lineage DAG connecting statements to raw dataset records.
16. **Audit Certificate**: Immutable SHA-256 decision certificate (`DEC-xxxx`).

---

## Non-Causal Notice
> **Decision-Support Artifact Notice**:
> Results are model- and data-dependent and should not be interpreted as causal or guaranteed agricultural recommendations. Predictions, scenario simulations, and analytical priorities provide decision-support guidance under empirical historical constraints.
