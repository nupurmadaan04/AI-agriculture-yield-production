# Day 31: Evidence Synthesis Framework & Taxonomies

## 1. Principle of Governed Evidence

Decision intelligence systems in high-stakes agricultural planning fail when they hallucinate certainty or conflate statistical predictions with ground-truth facts. The Day 31 Evidence Synthesis Framework enforces strict semantic typings and audit trails across all analytical entities.

Every claim in the decision brief must trace to a unique, immutable evidence object:
`EV-[CATEGORY]-[INDEX]`

---

## 2. Semantic Typology & Badging

| Semantic Classification | Color Code | Description | Example Entity |
|:---|:---|:---|:---|
| `OBSERVED` | Emerald Green | Empirical historical harvest observations recorded in panel dataset. | District actual yield recorded in `agricultural_panel.csv`. |
| `PREDICTED` | Blue | Governed pre-season estimate from certified forecast strategy. | Point forecast from `PredictionService`. |
| `SIMULATED` | Violet | Hypothetical scenario projection under controlled input assumptions. | Counterfactual cropland allocation projection. |
| `DERIVED` | Amber | Empirical statistical moments, trends, ranges, and ensemble spread. | Historical trend slope or P10–P90 tree dispersion. |
| `MODEL_ATTRIBUTION` | Indigo | Tree SHAP feature contributions decomposing model adjustments. | Local Shapley value for prior year yield lag. |
| `VALIDATION` | Teal | 4-Fold expanding walk-forward out-of-time error metrics. | Fold win rate %, test MAE, and baseline comparison. |
| `MONITORING` | Rose Red | Runtime Population Stability Index (PSI) drift, bias, and operational health. | Prediction PSI score vs reference baseline. |
| `PROVENANCE` | Slate Grey | Cryptographic SHA-256 lineage hash and audit trails. | Pipeline checksum and input feature hash. |
| `ASSUMPTION` | Light Blue | Explicit operating assumptions required for interpretation. | Constant technology trend assumption. |
| `LIMITATION` | Orange | Known data, temporal, or non-causal boundaries. | Observational panel vs controlled field trials. |

---

## 3. Evidence Completeness Evaluation

The evidence synthesis engine computes an objective evidence completeness score ($0-100$) and assigns an evidence tier:

```
+-----------------------------------------------------------------------------------+
| EVIDENCE TIER          | REQUIREMENTS                                             |
+-----------------------------------------------------------------------------------+
| STRONG_EVIDENCE        | 1. Certified ML Model (PRODUCTION_READY)                 |
| (Score >= 80)          | 2. Out-of-time walk-forward validation gain               |
|                        | 3. Historical sample depth >= 10 observations            |
|                        | 4. Empirical uncertainty bounds calculated               |
|                        | 5. Explainability feature attribution available          |
+-----------------------------------------------------------------------------------+
| PARTIAL_EVIDENCE       | 1. Certified Baseline (BASELINE_PRODUCTION) or           |
| (Score 60 - 79)        |    Conditional ML (CONDITIONAL_PRODUCTION)               |
|                        | 2. Walk-forward baseline benchmarks present              |
|                        | 3. Historical sample depth >= 5 observations             |
+-----------------------------------------------------------------------------------+
| LIMITED_EVIDENCE       | 1. Historical sample depth < 5 observations              |
| (Score 40 - 59)        | 2. Strategy fallback invoked                             |
|                        | 3. High monitoring drift or missing features             |
+-----------------------------------------------------------------------------------+
| INSUFFICIENT_EVIDENCE  | 1. Unsupported crop commodity                            |
| (Score < 40)           | 2. Geographic entity not found in panel                  |
+-----------------------------------------------------------------------------------+
```

---

## 4. Evidence Structure Schema

```json
{
  "evidence_id": "EV-HIST-0001",
  "category": "historical",
  "statement": "Historical yield records across 52 observations (1966–2017) report empirical mean of 954.2 kg/ha.",
  "value": 954.2,
  "unit": "kg/ha",
  "source_module": "data_loader",
  "source_method": "empirical_panel_aggregation",
  "evidence_type": "OBSERVED",
  "confidence_status": "HIGH",
  "timestamp": "2026-09-21T07:45:00Z",
  "model_version": "1.0.0",
  "dataset_version": "AGRI_PANEL_1.0 (ICRISAT 1966-2017)",
  "period": "1966-2017",
  "population": "Oilseeds in Punjab, Ludhiana",
  "interpretation": "Longitudinal baseline level for the regional commodity.",
  "limitation": "Reflects observational district panel averages, not experimental plots."
}
```

Every decision summary, recommendation option, and trade-off in the system explicitly links to one or more of these structured evidence records via their `supporting_evidence` ID list.
