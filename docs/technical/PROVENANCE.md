# Cryptographic Provenance Architecture

## 1. Digital Provenance Hashing

Implemented in `src/prediction_service.py` to establish an immutable audit trail for every served prediction:
$$\text{Provenance Fingerprint} = \text{SHA256}(\text{Request UUID} \parallel \text{Crop} \parallel \text{District} \parallel \text{Year} \parallel \text{Model Version} \parallel \text{Dataset Hash})$$

---

## 2. Append-Oriented Audit Log

- Audit records are synchronously written to `Datasets/metadata/prediction_audit_log.csv`.
- Each record captures request timestamp, request UUID, strategy used, model version, predicted yield, uncertainty bounds, and provenance hash.
- **Bitwise Determinism**: Verified via `tests/test_provenance_chain.py`; identical inputs produce bitwise identical SHA-256 signatures across process restarts.
