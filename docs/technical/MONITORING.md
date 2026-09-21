# Continuous Monitoring & Observability Engine

## 1. Population Stability Index (PSI) Drift Detection

The monitoring service (`src/monitoring_service.py`) calculates PSI across 10 empirical quantile bins for live operational feature vectors:
$$\text{PSI} = \sum_{b=1}^{10} (P_b - Q_b) \times \ln\left(\frac{P_b}{Q_b}\right)$$
- **Alert Levels**: Green ($\text{PSI} < 0.10$), Amber ($0.10 \le \text{PSI} < 0.25$), Red ($\text{PSI} \ge 0.25$).
- **Telemetry Log**: All runtime metrics append to `Datasets/metadata/operational_telemetry.jsonl`.

---

## 2. Post-Harvest Outcome Evaluation

Executed asynchronously via `src/outcome_evaluation.py` once actual agricultural census data is ingested:
- Computes signed bias ($\frac{1}{N}\sum (\hat{y}-y)$), MAE, RMSE, and MAPE.
- Years without harvested ground truth (e.g. 2026) return `EVALUATION_UNAVAILABLE`.
