# DAY 34 — Environment Reproducibility Audit

## Purpose

Ensure the platform can be deployed on any machine without machine-specific path dependencies, and that predictions are deterministic across environments.

## Path Portability Scan

### Scope

All Python source files in:
- `backend/` (API, services, utilities)
- `src/` (core ML, data, analytics modules)

### Forbidden Patterns

| Pattern | Risk | Found |
|---------|------|-------|
| `C:/` | Windows absolute path | ❌ Not Found |
| `C:\\` | Windows absolute path (escaped) | ❌ Not Found |
| `/home/` | Unix user home directory | ❌ Not Found |
| `/Users/` | macOS user home directory | ❌ Not Found |

### Allowed Relative Path Patterns

The codebase correctly uses:

```python
# Relative to module file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, '..', 'Datasets', 'your_file.csv')

# Environment variable overrides (for CI/CD)
model_dir = os.environ.get('MODEL_DIR', 'Models/')
```

## Determinism Guarantees

### Random State

All scikit-learn models are trained with explicit `random_state=42`. Verified in:
- `RandomForestRegressor(random_state=42)`
- `GradientBoostingRegressor(random_state=42)`

Consequence: Predictions for identical inputs are bitwise identical across restarts.

### Data Loading

The canonical dataset at `Datasets/final_agricultural_data_cleaned.csv` is:
- Read-only (never modified by the API)
- Version-controlled in git
- Loaded once at startup into `data_loader.dataframe`

### Model Serialization

Models serialized via `joblib` with protocol pinned:
- `.pkl` artifacts stored in `Models/multicrop/`
- Loaded at startup into `PredictionService`
- No re-training occurs at runtime

## Docker Reproducibility

### Volume Mounts

```yaml
volumes:
  - ./Datasets:/app/Datasets      # Read-mostly: agricultural data
  - ./Models:/app/Models          # Read-only: serialized model artifacts
  - ./Datasets/metadata:/app/Datasets/metadata  # Write: audit logs, telemetry
```

All volume paths are relative to the compose file location — portable across any machine.

### Image Build Reproducibility

Frontend:
- `node:18-alpine` base (pinned minor version)
- `npm ci` for exact dependency installation from `package-lock.json`

Backend:
- `python:3.11-slim` base
- `pip install -r requirements.txt` with pinned versions

## Audit Result

| Check | Status |
|-------|--------|
| Zero absolute paths in production code | ✅ PASS |
| All relative paths use `os.path` or `__file__` | ✅ PASS |
| Random state pinned for ML models | ✅ PASS |
| Dataset read-only at runtime | ✅ PASS |
| Volume mounts use relative paths | ✅ PASS |
| Dependencies pinned in requirements | ✅ PASS |
| Cross-restart prediction parity | ✅ PASS |
