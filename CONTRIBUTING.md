# Contributing to Agricultural Intelligence Platform

Thank you for your interest in contributing to the **Agricultural Intelligence Platform**! We welcome contributions from data scientists, software engineers, agronomists, and open-source enthusiasts.

---

## Code of Conduct

Please be respectful, collaborative, and constructive. We adhere to standard open-source community standards to ensure a welcoming environment for everyone.

---

## Development Setup

### Prerequisites
- **Python**: Version 3.10 or higher
- **Node.js**: Version 18.0 or higher (with `npm`)
- **Git**: Installed and configured

### 1. Fork and Clone the Repository
```bash
git clone https://github.com/<your-username>/AI-agriculture-yield-production.git
cd AI-agriculture-yield-production
```

### 2. Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat

# Install dependencies
pip install -r Requirements.txt
```

### 3. Set Up Frontend Web App
```bash
cd frontend
npm install
cd ..
```

---

## Running the Application Locally

### Start Backend API Server
```bash
# Run FastAPI server with hot reloading on port 8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI Swagger docs will be available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Start Frontend Development Server
```bash
cd frontend
npm run dev
```
The React frontend dashboard will be available at [http://localhost:5173](http://localhost:5173) (or `http://localhost:3000`).

---

## Testing & Quality Assurance

Before submitting any code changes or pull requests, ensure all automated tests pass:

### Run Python Test Suite
```bash
# Run all unit, integration, and governance tests
pytest tests/ -v

# Run determinism and reproducibility verification
python -m src.forecast_validation
```

### Run Frontend Type Check & Build
```bash
cd frontend
npm run build
```

---

## Repository Structure

```
AI-agriculture-yield-production/
├── backend/                  # FastAPI analytical backend with REST endpoints
│   ├── main.py               # Master API router & middleware
│   ├── routers/              # Modular endpoint routers
│   ├── schemas/              # Pydantic data schemas & response contracts
│   └── services/             # Analytical query & domain service handlers
├── frontend/                 # React 18 + Vite + TypeScript dashboard
│   ├── src/components/       # Modular UI sections & design system
│   ├── src/pages/            # 15+ analytical dashboards & tools
│   ├── src/services/         # Type-safe API clients
│   └── src/types/            # Shared TypeScript domain interfaces
├── src/                      # Scientific analytical & modeling engines
│   ├── data_ingestion/       # Multi-source ingestion & unified panel builder
│   ├── exogenous/            # Weather, NDVI & leakage auditing
│   ├── data_quality/         # Integrity audits & boundary validators
│   ├── strategy_registry.py  # Multi-crop strategy registry
│   ├── certification_guard.py# Pre-inference governance guards
│   ├── forecast_router.py    # Governed forecast dispatcher
│   └── ...
├── Datasets/                 # Unified longitudinal panel & metadata
├── Models/                   # Certified crop model pipelines & registries
├── tests/                    # Pytest test suite (100% passing)
├── docs/                     # Scientific whitepapers, model cards & audit logs
├── Dockerfile                # Production container definition
├── docker-compose.yml        # Multi-container local orchestration
├── pyproject.toml            # Project dependencies & pytest configuration
├── Requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Scientific & architectural platform documentation
```

---

## Contribution Workflow

1. **Create a Feature Branch**:
   ```bash
   git checkout -b feat/your-feature-name
   # or for bug fixes:
   git checkout -b fix/issue-description
   ```
2. **Make Changes**:
   - Write clean, well-documented, modular code.
   - Follow scientific integrity principles: never fabricate metrics, avoid data leakage, and validate temporally.
3. **Add Tests**:
   - Add unit tests in `tests/` for any new backend functionality.
   - Verify all tests pass with `pytest tests/`.
4. **Commit Changes**:
   ```bash
   git commit -m "feat(module): concise description of changes"
   ```
5. **Push and Open a Pull Request**:
   ```bash
   git push origin feat/your-feature-name
   ```
   Open a PR against the `main` branch with a clear summary of your changes.

---

## License

By contributing to this repository, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
