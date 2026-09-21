# 13. Reproducibility & Research Artifact Audit

### 13.1 Computational Environment & Dependencies

To enable full independent verification of all reported metrics, the platform's exact runtime configuration is specified below:

- **Git Commit Hash**: `84464c9588628592f49a5da44f9c24c3433900f7`
- **Operating System**: Microsoft Windows 11 Enterprise (Build 26100) / x86_64
- **Python Environment**: Python 3.11.9
- **Node.js Environment**: Node.js v24.14.0 (npm v11.9.0)
- **Container Environment**: Docker Engine 27.x / Docker Compose v2.x (Nginx reverse proxy + Python 3.11-slim)

---

### 13.2 Cryptographic Verification Hashes (SHA-256)

Every primary dataset, trained estimator, and governance registry is anchored by an immutable SHA-256 cryptographic hash:

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                    CRYPTOGRAPHIC ARTIFACT SIGNATURES (SHA-256)                                                |
+---------------------------------------------------+---------------+-----------------------------------------------------------+
| Relative File Path                                | Size (Bytes)  | SHA-256 Checksum                                          |
+---------------------------------------------------+---------------+-----------------------------------------------------------+
| Datasets/processed/agricultural_panel.csv         | 11,563,368    | 13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd|
| Datasets/rice_data_outlier_removed.csv            | 1,048,000     | 87387d5ddc6e681539245f857154aabe75011732bbd10d24b208397637|
| Datasets/metadata/dataset_manifest.json           | 1,610         | 70e8b6967d75a9636ebfa9cb77b1aa5177ebb58694d4be5293a2aae670|
| Models/forecasting_pipeline.pkl                   | 10,274,922    | 5d64b8f896f5fb9cb2deb4faa083edb3155355c049ad94eb54dc74f9c4|
| Models/forecasting_model_metadata.json            | 1,999         | a003f3e215e3730e5bce42ad06e6c642bd1860261d5f221525b37b4fd0|
| Models/multicrop/forecast_strategy_registry.json  | 15,415        | 8974a7ee4d1ba24ddef9e1a03b46ac129fa05e2ffd1b95c624a2f0843f|
| Datasets/metadata/final_model_certification.csv   | 6,273         | be396e6fd112a8e68c2c7366223f1db1892cd202baea6173b3f9d24e1b|
| Datasets/metadata/multicrop_model_selection.csv   | 4,651         | a46eca3558530f095be43bf674fb548b7f9f0648b8892ef1b569c157db|
| Datasets/metadata/exogenous_model_selection.csv   | 3,314         | 81f51efce2fe6e165db91f522fd529828dd25f4a1ccfc9489e526a615e|
+---------------------------------------------------+---------------+-----------------------------------------------------------+
```

---

### 13.3 Independent Reproduction Commands

To reproduce all quantitative validation tables, contract invariants, and test assertions:

```bash
# 1. Execute automated scientific reproducibility suite
pytest tests/test_day36_reproducibility.py -v

# 2. Execute UI contracts and deployment verification suites
pytest tests/test_day35_ui_contracts.py tests/test_deployment_verification.py -v

# 3. Verify frontend production bundle
cd frontend
npm run build
```

The test suite executes 32 regression tests in ~10 seconds with zero failures, confirming identical metrics, file shapes, and governance states across environments.
