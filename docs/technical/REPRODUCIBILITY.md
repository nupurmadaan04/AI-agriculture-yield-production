# Technical Reproducibility Specification

## 1. Environment Determinism

All random seeds are statically anchored across training and validation pipelines:
- `numpy.random.seed(42)`
- `random.seed(42)`
- Scikit-learn estimators initialized with `random_state=42`.

---

## 2. Artifact Verification Protocol

To independently verify that your local environment matches the certified reference environment:
1. Verify Git commit: `git rev-parse HEAD == 84464c9588628592f49a5da44f9c24c3433900f7`.
2. Verify Python version: `python --version` (Python 3.11.9).
3. Compute SHA-256 hash of `Datasets/processed/agricultural_panel.csv`:
   - Must match `13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b`.
4. Run `pytest tests/test_day36_reproducibility.py -v`.
   - All 7 tests must pass within 2.0 seconds.
