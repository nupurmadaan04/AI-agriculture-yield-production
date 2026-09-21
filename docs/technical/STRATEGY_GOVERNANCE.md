# Strategy Governance & Certification Engine

## 1. Governance State Definitions

The platform enforces three distinct production governance states across all commodities:

```
+---------------------------------------------------------------------------------------------------------+
|                                    GOVERNED STRATEGY CATEGORIES                                         |
+--------------------------+------------------------------------------------------------------------------+
| Status                   | Governance Criteria & Deployment Policy                                      |
+--------------------------+------------------------------------------------------------------------------+
| `PRODUCTION_READY`       | Walk-forward win-rate >= 75%, positive mean MAE gain, worst-fold loss < 5%. |
|                          | Deploys unconstrained machine learning with fallback for missing lags.       |
|                          | **Assigned to**: Oilseeds (+12.79% mean gain over baseline).                 |
+--------------------------+------------------------------------------------------------------------------+
| `CONDITIONAL_PRODUCTION` | Walk-forward win-rate >= 50%, positive gain, but exhibits drought volatility.|
|                          | Deploys machine learning with mandatory 3-sigma variance clipping fallback.  |
|                          | **Assigned to**: Sugarcane (+1.19% governed gain over baseline).              |
+--------------------------+------------------------------------------------------------------------------+
| `BASELINE_PRODUCTION`    | Win-rate < 50% or baseline MAE <= ML MAE across walk-forward folds.          |
|                          | Mandates Historical District Mean persistence as primary operational strategy|
|                          | **Assigned to**: 12 crops (Rice, Wheat, Chickpea, Maize, Sorghum, etc.).    |
+--------------------------+------------------------------------------------------------------------------+
```

---

## 2. Dynamic Fallback Execution

If an inference request targets a district with unrecorded previous-year lags, the runtime triggers a graceful, transparent fallback sequence:
$$\text{Primary ML} \longrightarrow \text{Historical District Mean} \longrightarrow \text{State Agro-Climatic Mean}$$
All fallback occurrences are explicitly flagged in the API response metadata (`strategy_used: "FALLBACK_DISTRICT_MEAN"`).
