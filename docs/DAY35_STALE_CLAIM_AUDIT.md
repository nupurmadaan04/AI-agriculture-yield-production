# DAY 35 — Stale Claim & Dataset Scale Audit

## 1. Executive Summary

Day 35 audited all numerical claims, historical counts, and dataset references across the codebase, documentation, and frontend components. The primary objective was to reconcile legacy references to the initial single-crop Rice dataset with the platform's multi-crop canonical panel scale.

---

## 2. Platform Scale Reconciliation

| Metric | Legacy / Stale Reference | Current Verified Canonical Platform Scale | Resolution |
|---|---|---|---|
| **Total Panel Records** | 2,469 records | **71,601 records** | Clarified: 2,469 represents the historical single-crop Rice panel (311 districts × ~8 years); 71,601 represents the unified 29-crop agricultural panel (1966–2017). |
| **Commodity Coverage** | Single-crop (Rice only) | **29 canonical crops (14 model-ready)** | Verified across tournament evaluation and strategy registries. |
| **Geographic Scope** | 20 States, 311 Districts | **20 States, 311 Districts** | Confirmed: Full pan-India representation across major agricultural zones. |
| **Validation Period** | Single train/test split | **Chronological Expanding Walk-Forward (2014–2017)** | Confirmed: Four distinct out-of-sample origins testing multi-origin stability. |
| **Model Scope** | Single Random Forest | **Multi-strategy tournament (RF, GBDT, Historical Mean)** | Governed strategy registry selecting certified ML or statistical persistence baselines. |

---

## 3. External Feed & Sensor Claims Audit

- **Satellite Remote Sensing**: The canonical panel is grounded strictly in historical district statistical records (ICRISAT/DES). Any references implying real-time high-resolution satellite radar or daily NDVI streaming have been clarified as historical district aggregations.
- **Micro-Climate / IoT Sensors**: No live IoT soil probe or on-farm weather station feeds are connected. Clarified in [ObservabilityCenter.tsx](file:///c:/Users/devin/AI-agriculture-yield-production/frontend/src/pages/ObservabilityCenter.tsx) and [MonitoringOverview.tsx](file:///c:/Users/devin/AI-agriculture-yield-production/frontend/src/components/monitoring/MonitoringOverview.tsx) that monitoring denotes statistical covariate drift (PSI) and error diagnostics over evaluated batches.
- **National Portal Integration**: Clarified that data models are structured for alignment with National APY/UPAg schema definitions, but do not claim live government cloud synchronizations unless tested.

---

## 4. Verification

The automated test `test_zero_ungrounded_marketing_claims` in [tests/test_day35_ui_contracts.py](file:///c:/Users/devin/AI-agriculture-yield-production/tests/test_day35_ui_contracts.py) verifies zero occurrences of ungrounded accuracy claims or inflated capabilities across all frontend pages.
