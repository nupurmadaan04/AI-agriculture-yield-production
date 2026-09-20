# Day 29: Scientific Validation & Determinism Verification

## 1. Scope & Scientific Boundary

The Day 29 implementation establishes strict empirical controls across the forecasting platform:

1. **Pre-Season Forecasting vs Post-Harvest Yield Calculation**:
   - Pre-season yield forecasts use strictly lag features available before the crop sowing window ($t-1$ yield, 3-year rolling average, $t-1$ area, spatial encodings).
   - Post-harvest yield reconstruction ($\text{Production} / \text{Area} \times 1000$) is never presented as an ML model or predictive forecast.
2. **Frozen Modeling Layer**:
   - Zero models retrained.
   - Zero model weights or hyperparameters modified.
   - Validated walk-forward metrics across all 14 crops preserved exactly.

---

## 2. Temporal Correctness & Historical Window

For any forecast request targeting harvest horizon $Y$ (e.g. $Y = 2018$):
- **Immediate Preceding Season ($t-1$)**: $\text{Observation Year } Y - 1 = 2017$.
- **3-Year Rolling Reference Window**: $\text{Observation Years } [Y-3, Y-1] = [2015, 2017]$.
- **Longitudinal District Distribution**: Sourced exclusively from $\text{Observations with } \text{Year} < Y$.
- **Leakage Prevention**: No observations from Year $\ge Y$ are ever incorporated into context calculations, baseline references, or inference features.

---

## 3. Dual-Run Bitwise Invariance (Determinism)

Identical forecast requests submitted sequentially must return identical outputs:

$$\text{Prediction}_1 \equiv \text{Prediction}_2$$
$$\text{Strategy}_1 \equiv \text{Strategy}_2$$
$$\text{ModelVersion}_1 \equiv \text{ModelVersion}_2$$
$$\text{DatasetVersion}_1 \equiv \text{DatasetVersion}_2$$
$$\text{ProvenanceFingerprint}_1 \equiv \text{ProvenanceFingerprint}_2$$

This deterministic invariance has been verified across test cases in `tests/test_prediction_consistency.py`.

---

## 4. Cryptographic Provenance Integrity

Every forecast execution generates a SHA-256 fingerprint binding:
- Target crop, state, and district
- Forecast horizon year
- Governed strategy and model version
- Exact input feature vector
- Model artifact hash
- Output prediction value and canonical unit (`kg/ha`)

This record is stored in memory and persisted to `Datasets/metadata/prediction_audit_log.csv` for full traceability.
