# Day 32: Production Decision Workspace & Scenario Comparison

## 1. Overview & Architecture

The **Production Decision Workspace** is an evidence-first synthesis and what-if exploration layer designed for agricultural economists, regional planners, and commodity analysts. It integrates governed pre-season forecasting, authentic longitudinal panel history, out-of-time walk-forward validation benchmarks, runtime operational drift monitoring, and comparative what-if scenario simulations.

```
Frontend (/decision-workspace)
      │
      ▼
Decision Workspace API (/api/workspace/analyze)
      │
      ├───────────────────────┬──────────────────────┬──────────────────────┐
      ▼                       ▼                      ▼                      ▼
Forecast Router         Scenario Engine        Decision Intelligence  Monitoring Service
(PredictionService)     (ScenarioService)      (Validation Evidence)  (PSI Drift & Outcomes)
      │                       │                      │                      │
      ▼                       ▼                      ▼                      ▼
Governed Point          What-If Archetypes     Walk-Forward Metrics   Live Telemetry &
Forecast & SHA-256      & Bounded Deltas       & Tree SHAP Attrib.    Post-Harvest Bias
```

---

## 2. Reused Governed Subsystems (No Duplication)

The Day 32 implementation does **not** modify or rebuild existing scientific layers. Instead, it acts as a thin, auditable orchestration layer:

1. **Governed Forecast Router (`PredictionService`)**:
   Routes crops across the certified hierarchy (Oilseeds → Random Forest, Sugarcane → Gradient Boosting, Rice/Wheat → Historical District Mean / Persistence). Obtains pre-season point estimates, strategy names, and cryptographic SHA-256 hashes.
2. **Scenario Simulation Engine (`ScenarioService` & `ScenarioEngine`)**:
   Reuses supported what-if archetypes (`conservative_improvement`, `moderate_improvement`, `stress_scenario`, `custom`) and parameter bounds.
3. **Walk-Forward Validation (`StrategyRegistry` & `CertificationGuard`)**:
   Extracts out-of-time evaluation metrics over 2014–2017 expanding folds (MAE, Win Rate %, relative improvement vs persistence baseline).
4. **Forecast Monitoring & Drift (`ForecastMonitoringService`)**:
   Pulls Population Stability Index (PSI) drift scores, active alerts, and post-outcome directional bias ($\text{predicted} - \text{observed}$).
5. **Model Explainability (`explainability_service`)**:
   Extracts local Tree SHAP feature attributions for ML strategies, framing them as statistical dependencies rather than biological causation.

---

## 3. Core Functional Cards

### A. Governed Baseline Forecast Card
Displays the authoritative pre-season prediction:
- Predicted yield ($\text{kg/ha}$)
- Strategy tier (`PRODUCTION_READY`, `CONDITIONAL_PRODUCTION`, `BASELINE_PRODUCTION`)
- Model artifact and pipeline version
- Cryptographic SHA-256 provenance fingerprint and unique Request ID

### B. Historical Context & Trajectory Card
- Empirical moments ($N$, mean, median, min, max, standard deviation) from `agricultural_panel.csv` (1966–2017).
- Covariance-based linear regression trend slope ($\text{kg/ha/yr}$) over recent harvest observations.
- Strict temporal isolation: observations are bounded strictly to $\text{Year} < \text{forecast\_year}$ to prevent lookahead leakage.

### C. Validation & Empirical Uncertainty Panel
- 4-Fold expanding walk-forward out-of-time MAE, fold win rate %, and relative improvement.
- Academic legacy benchmark note for Rice ($R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$).
- Strategy-aware uncertainty: P10–P90 tree dispersion bounds for certified ML models; explicitly marked unavailable for deterministic baselines.
- Mandatory disclaimer that empirical spread does not constitute a formal distribution-free confidence interval.

### D. What-If Scenario Comparison Matrix
- Displays Baseline vs Conservative vs Moderate vs Stress vs Custom scenarios side-by-side.
- Compares Projected Yield, Yield Delta ($\text{kg/ha}$), Relative Change (%), Uncertainty, Assumptions, and Evidence Classification.
- **Strict Non-Autonomous Principle**: Zero subjective ranking words (`BEST`, `WORST`, `WINNER`, `RECOMMENDED`). All trade-offs are presented purely quantitatively.

### E. Model Attribution (Tree SHAP) & Monitoring Health
- Feature contribution bars for ML models (prior year yield lag, 3-year rolling mean, land area).
- Persistence baseline explanation for baseline commodities.
- Prediction distribution PSI drift and post-harvest evaluation availability.

---

## 4. Governance Sign-Off
- **Models Changed**: NO
- **Weights Changed**: NO
- **Validation Changed**: NO
- **Scenarios Changed**: NO
- **Prescriptive Directives**: ZERO
