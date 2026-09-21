# Master Reproducibility Guide

## 1. Quick Reproduction Steps

To verify all claims, metrics, and models reported in this study from scratch:

```bash
# 1. Verify Git commit state
git rev-parse HEAD
# Expected: 84464c9588628592f49a5da44f9c24c3433900f7

# 2. Verify Python environment
python --version
# Expected: Python 3.11.9

# 3. Verify canonical dataset hash
python -c "import hashlib; print(hashlib.sha256(open('Datasets/processed/agricultural_panel.csv', 'rb').read()).hexdigest())"
# Expected: 13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b

# 4. Run automated reproducibility test suite
pytest tests/test_day36_reproducibility.py -v
# Expected: 7 passed in ~1s

# 5. Run full deployment and UI contracts verification suites
pytest tests/test_day35_ui_contracts.py tests/test_deployment_verification.py -v
# Expected: 25 passed in ~9s

# 6. Verify frontend compilation
cd frontend
npm run build
# Expected: Clean build with 0 TypeScript errors
```

---

## 2. Directory Layout & Artifact Map

- `docs/reproducibility/EXPERIMENT_LINEAGE.md`: Chronological experiment lineage tracing all 37 days of research.
- `docs/reproducibility/CLAIM_EVIDENCE_MATRIX.md`: Formal mapping of every quantitative claim to its underlying physical artifact.
- `docs/reproducibility/REPRODUCIBILITY_MANIFEST.md`: Complete SHA-256 inventory of all repository models and datasets.
