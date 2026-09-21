# Day 32: Scientific Validation & Verification Report

## 1. Scientific Freeze Attestation

As mandated for Day 32 integration, all scientific layers remain strictly frozen:

- **Trained Model Weights**: Unchanged (`Models/*.pkl` unchanged)
- **Model Hyperparameters**: Unchanged
- **Strategy Classifications**: Unchanged (`StrategyRegistry` unchanged)
- **Certification Thresholds**: Unchanged (`CertificationGuard` unchanged)
- **Validation Methodology**: Unchanged (4-Fold Expanding Walk-Forward 2014–2017)
- **Scenario Mathematics**: Unchanged (`ScenarioEngine` unchanged)
- **Canonical Panel Dataset**: Unchanged (`data/canonical/agricultural_panel.csv` unchanged)
- **Synthetic Data**: ZERO added
- **Fabricated Outcomes**: ZERO added

---

## 2. Golden Case Verification

Four commodity archetypes were rigorously tested across the workspace pipeline:

### 1. Oilseeds (Certified Governed ML Production Strategy)
- **Strategy Tier**: `PRODUCTION_READY`
- **Model Engine**: Random Forest Regressor (`oilseeds_rf_production.pkl`)
- **Validation Metrics**: 4-Fold Walk-Forward $\text{MAE} \approx 206.3\text{ kg/ha}$, Fold Win Rate $= 50.0\%$, relative gain vs baseline $= 10.85\%$.
- **Evidence Rendered**: Tree SHAP attribution values, empirical P10–P90 tree dispersion uncertainty, PSI drift status.
- **Verification Status**: **PASS**

### 2. Sugarcane (Conditional ML Production Strategy)
- **Strategy Tier**: `CONDITIONAL_PRODUCTION`
- **Model Engine**: Gradient Boosting Regressor (`Sugarcane_GradientBoosting_v1.pkl`)
- **Validation Metrics**: 4-Fold Walk-Forward $\text{MAE} \approx 6265.1\text{ kg/ha}$, Fold Win Rate $= 75.0\%$, relative gain vs baseline $= 19.34\%$.
- **Evidence Rendered**: Model feature attributions, empirical uncertainty interval, conditional deployment caveats.
- **Verification Status**: **PASS**

### 3. Rice (Certified Baseline Production Strategy)
- **Strategy Tier**: `BASELINE_PRODUCTION`
- **Model Engine**: Historical District Mean / Persistence Benchmark (`rice_district_mean_production`)
- **Validation Evidence**: Walk-forward benchmark metrics; academic historical baseline reference ($R^2 = 0.7866$, $\text{MAE} = 353.01\text{ kg/ha}$).
- **Evidence Rendered**: Deterministic baseline attribution (`PERSISTENCE_BASELINE`), explicit indicator that ML tree uncertainty is unavailable.
- **Verification Status**: **PASS**

### 4. Wheat (Certified Baseline Production Strategy)
- **Strategy Tier**: `BASELINE_PRODUCTION`
- **Model Engine**: Historical District Mean / Persistence Benchmark (`wheat_district_mean_production`)
- **Validation Evidence**: Historical persistence benchmark metrics.
- **Evidence Rendered**: Deterministic baseline attribution, ML-only evidence suppressed.
- **Verification Status**: **PASS**

---

## 3. Edge Case Handling

The workspace router and engine implement robust structured status handling:

| Scenario / Edge Case | Expected System Behavior | Verified Result |
| :--- | :--- | :--- |
| **Unsupported Commodity** | 400 Bad Request with supported crop list | **PASS** (`UNSUPPORTED_CROP`) |
| **Unsupported State / District** | Graceful fallback to state-level statistics; clear data coverage warning | **PASS** |
| **Future Harvest Year (e.g. 2026)** | Outcome evaluation marked `EVALUATION_UNAVAILABLE` | **PASS** (Zero hallucinated outcomes) |
| **Historical Year with Ground Truth (e.g. 2017)** | Displays verified observed yield, forecast, and prediction error | **PASS** |
| **Empty Scenario Request** | Generates default archetypes (Conservative, Moderate, Stress) | **PASS** |
| **Non-Ranking Comparison Matrix** | Zero occurrences of subjective ranking words (`BEST`, `WORST`, `WINNER`, `RECOMMENDED`) | **PASS** (Strictly validated by regex) |

---

## 4. Analytical Determinism

Bitwise determinism was verified via `tests/test_workspace_determinism.py`:
- Two identical workspace requests executed independently with the same scope (`crop='oilseeds'`, `state='Madhya Pradesh'`, `district='Indore'`, `year=2026`) yield:
  - Identical baseline forecast yield ($\text{predicted\_yield}_1 \equiv \text{predicted\_yield}_2$)
  - Identical scenario simulation values across all archetypes
  - Identical delta calculations ($\Delta_{\text{abs}}$ and $\Delta\%$)
  - Identical validation metrics and attribution vectors
  - Stable cryptographic SHA-256 fingerprint
- Operational fields (`request_id`, `generated_at_utc`) vary per invocation as designed for trace logging.

---

## 5. Non-Autonomous Decision Support Declaration

> **Mandatory Disclaimer**:
> The Decision Workspace is a decision-support interface. It does not autonomously select an action. It presents governed predictions, historical context, out-of-time validation, live drift monitoring, and what-if simulation deltas. The responsibility and authority for agricultural and economic decisions rest entirely with human stakeholders.
