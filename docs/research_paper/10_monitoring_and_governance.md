# 9. Continuous Monitoring, Drift Detection & Provenance Governance

### 9.1 Conceptual Partitioning in Operations

To maintain scientific integrity during live serving, the platform enforces an explicit separation across operational concepts:

```
+----------------------------------------------------------------------------------------------------+
|                                  OPERATIONAL CONCEPTS PARTITION                                    |
+--------------------------+-------------------------------------------------------------------------+
| Concept                  | Scope and Definition                                                    |
+--------------------------+-------------------------------------------------------------------------+
| **Model Validation**     | Historical out-of-sample benchmark against verified ground truth.       |
| **Runtime Monitoring**   | Operational telemetry (latency, memory, throughput, distribution drift).|
| **Data Quality**         | Pre-inference schema validation, missingness checks, zero-leakage audit.|
| **Post-Harvest Outcome** | Retrospective evaluation once actual census yields become available.     |
+--------------------------+-------------------------------------------------------------------------+
```

Runtime monitoring events (e.g. inference requests, covariate shifts) must never be conflated with historical model validation benchmarks.

---

### 9.2 Covariate Drift Monitoring via Population Stability Index (PSI)

To detect distribution shifts in operational feature inputs prior to harvest, the monitoring service (`src/monitoring_service.py`) calculates the Population Stability Index (PSI) for each incoming feature batch relative to the reference training distribution:

$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \times \ln\left( \frac{P_b}{Q_b} \right)$$

where $P_b$ represents the proportion of live observations falling into quantile bin $b$, and $Q_b$ represents the historical training baseline proportion. Operational actions are governed by established industry standards:
- **$\text{PSI} < 0.10$ (Normal / Green)**: Feature distribution is stable; standard model inference proceeds without intervention.
- **$0.10 \le \text{PSI} < 0.25$ (Moderate Drift / Amber)**: Moderate covariate shift detected; system issues a supervisory telemetry flag.
- **$\text{PSI} \ge 0.25$ (Significant Drift / Red)**: Substantial distribution departure; system flags high-uncertainty warnings and alerts operators to inspect potential regional climate or reporting anomalies.

---

### 9.3 Retrospective Post-Harvest Outcome Evaluation

Once final agricultural harvest records are published (typically 6–12 months post-harvest), the outcome evaluation pipeline (`src/outcome_evaluation.py`) decomposes operational errors:
- **Signed Bias**: $e_t = \frac{1}{N} \sum_{i=1}^N (\hat{y}_{i, t} - y_{i, t})$.
- **Root Mean Squared Error (RMSE)** and **Mean Absolute Percentage Error (MAPE)**.
- Any unharvested or future forecast year (e.g. Year 2026) is strictly flagged as **`EVALUATION_UNAVAILABLE`**, preventing premature evaluation against missing ground truth.

---

### 9.4 Cryptographic Provenance & Append-Oriented Audit Trails

Every served forecast executes within an immutable digital provenance chain:

```
CANONICAL DATASET (SHA-256 Hash)
      ↓
FEATURE PIPELINE (Shift-1 Lag Checksum)
      ↓
TRAINED MODEL ARTIFACT (Serialized Pickle Hash)
      ↓
CERTIFIED STRATEGY (Registry Governance Hash)
      ↓
PREDICTION INFERENCE RUNTIME
      ↓
CRYPTOGRAPHIC EXECUTION SIGNATURE (SHA-256)
```

For every request, the engine generates an immutable cryptographic signature:
$$\text{Provenance Fingerprint} = \text{SHA256}(\text{Request UUID} \parallel \text{Crop} \parallel \text{District} \parallel \text{Year} \parallel \text{Model Version} \parallel \text{Dataset Hash})$$

This fingerprint is appended to an immutable audit log (`Datasets/metadata/prediction_audit_log.csv`) alongside execution timestamps, input feature snapshots, and strategy certification tags, enabling bitwise post-hoc reproduction of any historical forecast.
