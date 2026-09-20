# Day 24: Safety Guards & Rejection Policies

## 1. Pre-Inference Governance Pipeline

The pre-inference validation engine in `src/certification_guard.py` enforces five non-negotiable gates:

```
[Request Input]
      │
      ▼
 Gate 1: Input Completeness
      ├─► Missing crop / state / district ──────────► [REJECT: INPUT_INCOMPLETE]
      ▼
 Gate 2: Crop Strategy Registration
      ├─► Crop not in forecast_strategy_registry ───► [REJECT: UNSUPPORTED_CROP]
      ▼
 Gate 3: Geographic Coverage Match
      ├─► District not in forecast_coverage.csv ────► [REJECT: DISTRICT_UNSUPPORTED]
      ▼
 Gate 4: Model Artifact Availability & Hash Match
      ├─► Pickle missing from Models/multicrop/ ────► [REJECT: MODEL_ARTIFACT_MISSING]
      ▼
 Gate 5: Execution Safety & Variance Bounds
      └─► Valid ───────────────────────────────────► [ALLOW: CERTIFIED_ALLOW]
```

---

## 2. Rejection Code Taxonomy

| Rejection Code | Trigger Condition | System Behavior |
| :--- | :--- | :--- |
| `INPUT_INCOMPLETE` | Blank or null crop, state, or district | Returns descriptive 400 error; logs event in audit log. |
| `UNSUPPORTED_CROP` | Crop is not certified in Day 23 registry (e.g. Potato) | Rejects request; prevents uncertified ML or fabricated baselines. |
| `DISTRICT_UNSUPPORTED` | District does not exist in 1966–2017 training panel | Rejects request; prevents spatial extrapolation without historical records. |
| `MODEL_ARTIFACT_MISSING` | Model `.pkl` missing from disk | Prevents serving unverified code; fails safely. |
| `SPARSE_DISTRICT_FALLBACK` | District records $< 3$ | Automatically activates statistical fallback instead of failing. |
