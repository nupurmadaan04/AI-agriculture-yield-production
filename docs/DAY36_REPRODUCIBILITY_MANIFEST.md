# DAY 36 — Platform Scientific Reproducibility Manifest

This manifest documents the exact environment, dependencies, cryptographic artifact hashes, execution commands, and verification protocols required to independently reproduce all verified scientific findings of the platform.

---

## 1. Repository & Runtime Environment

- **Repository**: `AI-agriculture-yield-production`
- **Current Git Commit**: `84464c9588628592f49a5da44f9c24c3433900f7`
- **Branch**: `main`
- **Host Operating System**: Microsoft Windows 11 Enterprise (Build 26100) / x86_64
- **Python Runtime**: Python 3.11.9
- **Node.js Runtime**: Node.js v24.14.0 (npm v11.9.0)
- **Containerization Target**: Docker Engine / Docker Compose (Nginx reverse proxy + Python 3.11-slim)

---

## 2. Core Python Dependencies

| Package | Documented Version | Purpose |
|---|---|---|
| `fastapi` | `>=0.100.0` | Production forecast serving & observability API |
| `uvicorn` | `>=0.23.0` | ASGI application server |
| `scikit-learn` | `1.6.1` (or `>=1.3.0`) | Model training, tree estimators, metrics |
| `pandas` | `2.2.3` (or `>=2.0.0`) | Longitudinal panel data handling |
| `numpy` | `2.2.3` (or `>=1.24.0`) | Array operations, linear algebra |
| `scipy` | `>=1.11.0` | Optimization (SLSQP), statistical distributions |
| `pytest` | `8.4.0` | Automated testing & contract verification |
| `httpx` | `0.28.1` | FastAPI asynchronous test client |

---

## 3. Cryptographic Artifact Hashes (SHA-256)

All primary datasets, trained models, and registries have been verified with immutable SHA-256 digital signatures:

### 3.1 Primary Datasets
- **Canonical Agricultural Panel** (`Datasets/processed/agricultural_panel.csv`):
  - **Size**: 11,563,368 bytes (71,601 rows × 10 columns)
  - **SHA-256**: `13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b`
- **Single-Crop Rice Panel** (`Datasets/rice_data_outlier_removed.csv`):
  - **Size**: 1,048,000 bytes (2,469 rows × 81 columns)
  - **SHA-256**: `87387d5ddc6e681539245f857154aabe75011732bbd10d24b2083976373d15a0`
- **Dataset Manifest** (`Datasets/metadata/dataset_manifest.json`):
  - **Size**: 1,610 bytes
  - **SHA-256**: `70e8b6967d75a9636ebfa9cb77b1aa5177ebb58694d4be5293a2aae670b590ad`

### 3.2 Core Model Artifacts & Registries
- **Rice Legacy Pipeline** (`Models/forecasting_pipeline.pkl`):
  - **Size**: 10,274,922 bytes
  - **SHA-256**: `5d64b8f896f5fb9cb2deb4faa083edb3155355c049ad94eb54dc74f9c4e36086`
- **Rice Model Metadata** (`Models/forecasting_model_metadata.json`):
  - **Size**: 1,999 bytes
  - **SHA-256**: `a003f3e215e3730e5bce42ad06e6c642bd1860261d5f221525b37b4fd0628247`
- **Forecast Strategy Registry** (`Models/multicrop/forecast_strategy_registry.json`):
  - **Size**: 15,415 bytes
  - **SHA-256**: `8974a7ee4d1ba24ddef9e1a03b46ac129fa05e2ffd1b95c624a2f0843f242e46`
- **Final Model Certification** (`Datasets/metadata/final_model_certification.csv`):
  - **Size**: 6,273 bytes
  - **SHA-256**: `be396e6fd112a8e68c2c7366223f1db1892cd202baea6173b3f9d24e1b596517`
- **Multicrop Model Selection** (`Datasets/metadata/multicrop_model_selection.csv`):
  - **Size**: 4,651 bytes
  - **SHA-256**: `a46eca3558530f095be43bf674fb548b7f9f0648b8892ef1b569c157db689588`
- **Exogenous Model Selection** (`Datasets/metadata/exogenous_model_selection.csv`):
  - **Size**: 3,314 bytes
  - **SHA-256**: `81f51efce2fe6e165db91f522fd529828dd25f4a1ccfc9489e526a615e1a024a`

---

## 4. Execution Commands for Reproduction

To independently reproduce and verify all scientific claims and contract invariants from the repository:

### 4.1 Backend Scientific Reproducibility Verification
```bash
# Execute Day 36 dedicated scientific reproducibility test suite
pytest tests/test_day36_reproducibility.py -v

# Execute Day 34 & Day 35 operational and UI contract suites
pytest tests/test_day35_ui_contracts.py tests/test_deployment_verification.py -v

# Execute full acceptance and resilience suite (Days 33-35)
pytest tests/test_end_to_end_acceptance.py tests/test_security_acceptance.py tests/test_failure_resilience.py tests/test_api_contracts.py tests/test_provenance_chain.py tests/test_concurrency_safety.py -v
```

### 4.2 Frontend Production Bundle Verification
```bash
cd frontend
npm run build
```

---

## 5. Defined Numerical Tolerances

| Metric Type | Allowed Absolute Tolerance | Allowed Relative Tolerance | Rationale |
|---|---|---|---|
| **Record Counts / Counts** | `0` (Exact integer equality) | `0%` | Discrete row, state, district, and fold counts must match exactly. |
| **Model R² Scores** | `1e-3` (0.001) | `0.1%` | Accounts for minor floating-point divergence across compiler flags. |
| **MAE / RMSE (kg/ha)** | `0.1 kg/ha` | `0.05%` | Yield metrics are measured in integer-level kg/ha; 0.1 kg/ha is negligible. |
| **MAPE (%) / Improvement (%)** | `0.05%` | `0.1%` | Percentage improvements require high precision to prevent misleading claims. |
| **SHA-256 Hashes** | Exact match | Exact match | File integrity and digital provenance are bitwise immutable. |
