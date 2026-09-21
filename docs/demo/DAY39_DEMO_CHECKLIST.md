# DAY 39 — DEMO PREPARATION, FAILOVER & DISASTER RECOVERY CHECKLIST
## AI Agriculture Intelligence Platform

> **Target Audience:** Presenter, Technical Operations, SRE Team  
> **Purpose:** Zero-downtime operational preparedness for live presentations, thesis defenses, and high-stakes technical reviews.

---

## 1. Pre-Demo Setup Timeline Checklist

### T - 30 Minutes: Infrastructure & Environment Warmup
- [ ] **System Resources**: Verify host machine has at least 4 GB RAM and < 50% CPU utilization. Close heavy background apps (Slack, Discord, Zoom background effects).
- [ ] **Git State**: Confirm working directory is clean on the target release branch (`git status`). Ensure no uncommitted debug prints or broken test mocks exist.
- [ ] **Python Environment**: Confirm virtual environment is activated (`which python` / `Get-Command python`) and dependencies match `requirements.txt`.
- [ ] **Node Environment**: Confirm Node v18+ and npm are functional (`node -v`, `npm -v`).

### T - 15 Minutes: Service Boot & Smoke Testing
- [ ] **Backend Launch**: Start FastAPI backend in dedicated terminal:
  ```bash
  uvicorn backend.main:app --host 127.0.0.1 --port 8000
  ```
- [ ] **Liveness & Readiness Verification**:
  ```bash
  curl -s http://127.0.0.1:8000/health
  # Expected: {"status": "ok", "service": "agricultural-intelligence-api"}
  curl -s http://127.0.0.1:8000/ready
  # Expected: {"status": "ready", ...}
  ```
- [ ] **Frontend Launch**: Start Vite development server in separate terminal:
  ```bash
  cd frontend && npm run dev
  ```
- [ ] **Frontend Smoke Check**: Confirm `http://localhost:5173` is serving HTTP 200 OK.

### T - 5 Minutes: Browser Environment Preparation
- [ ] **Clean Profile**: Open a clean browser window (Incognito / Dedicated Demo Profile) with zero distracting browser extensions.
- [ ] **Resolution & Zoom**: Set browser zoom to 100% or 110% on a 1080p display for optimal legibility.
- [ ] **Pre-Load Demo Tabs in Exact Sequence**:
  1. **Tab 1:** `http://localhost:5173/` (Executive Homepage)
  2. **Tab 2:** `http://localhost:5173/portal` (National Agricultural Panel Explorer)
  3. **Tab 3:** `http://localhost:5173/prediction-explorer` (Governed Prediction Explorer)
  4. **Tab 4:** `http://localhost:5173/monitoring` (Forecast Monitoring & Drift)
  5. **Tab 5:** `http://localhost:5173/decision-intelligence` (Evidence-Based Decision Briefs)
- [ ] **Warm In-Memory Cache**: Click through Tab 3 once (execute Oilseeds and Rice predictions) so in-memory artifacts and React Query caches are fully primed for sub-millisecond tab switching.

### T - 0: Showtime
- [ ] Silence cell phone and operating system notifications.
- [ ] Full screen browser (`F11` if appropriate).
- [ ] Deep breath, maintain clear and measured pacing (~130 wpm).

---

## 2. Live Demo Execution Checklist

| Segment | Action | Expected Visual Verification | Fallback Action if Sluggish |
|---|---|---|---|
| **0:00–0:30** | Open Tab 1 (Homepage) | Hero banner, Governance Badges active | Hard refresh `Ctrl+F5` |
| **0:30–1:00** | Architecture overview | Health indicators green (`200 OK`) | Point to pre-saved architecture diagram in Tab 6 |
| **1:00–2:00** | Open Tab 2 (`/portal`) | Select MP -> Ujjain, 51-yr panel renders | Switch to Punjab -> Ludhiana if already cached |
| **2:00–3:00** | Open Tab 3 (`/prediction-explorer`) | **Case 1 (Oilseeds):** Green ML Badge, 487.65 kg/ha | Use offline curl command if UI stalls |
| | | **Case 2 (Rice):** Neutral Baseline Badge, 4,512.40 kg/ha | |
| **3:00–3:45** | Expand XAI Accordion | Marginal attribution bars render (+64.2 lag1) | Show `docs/demo/DAY39_XAI_VIVA.md` Table |
| **3:45–4:30** | View Uncertainty & Provenance | `[412.3 - 568.1 kg/ha]`, SHA-256 copied | Display terminal verification card |
| **4:30–5:15** | Open Tab 4 (`/monitoring`) | PSI gauge stable (< 0.1), Bias diagnostics render | Show pre-rendered monitoring report |
| **5:15–6:00** | Open Tab 5 (`/decision-intelligence`) | Decision brief with `[OBSERVED]`, `[PREDICTED]` | Display cached decision brief card |
| **6:00–7:00** | Summary Slide / Modeling Readiness | Results scorecard table displayed | Refer directly to `docs/demo/DAY39_RESULTS.md` |

---

## 3. Failover & Disaster Recovery Procedures

### Scenario A: Backend API Fails / Dies During Demo
1. **Immediate Reaction**: Do not panic or stare at the error. Calmly announce:  
   *"Our frontend error boundary has gracefully caught the connection drop. Let me restart the stateless backend service."*
2. **Action**:
   - In the backend terminal, press `Ctrl+C` and re-run:
     ```bash
     uvicorn backend.main:app --port 8000
     ```
   - Service boots in $< 3\text{ seconds}$ because dataset and models are local.
   - Click "Retry" in the frontend UI.
3. **If Python process is locked**:
   ```powershell
   Stop-Process -Name "python" -Force
   uvicorn backend.main:app --port 8000
   ```

### Scenario B: Frontend Vite Server Stalls
1. **Immediate Action**:
   - Open a pre-compiled production build served via Python's built-in HTTP server or Nginx:
     ```bash
     cd frontend/dist && python -m http.server 5174
     ```
   - Navigate browser to `http://localhost:5174`.

### Scenario C: Complete Machine / Network Failure
1. **Backup Artifacts**: Maintain an offline USB or separate device with:
   - `docs/demo/DAY39_PRESENTATION.md` (Slide deck)
   - `docs/demo/DAY39_RESULTS.md` (Audited numbers)
   - Verified pre-recorded 5-minute screencast in MP4/WebP.
2. **Emergency Offline Pitch**:
   - Transition to the **Emergency Offline Demo Script** below, walking the audience through the architecture, methodologies, and exact results with zero disruption.

---

## 4. Emergency Offline Demo Script (Zero Live System Requirement)

> *"If a live screen share or container fails, deliver this verbal technical defense:"*

> *"Examiners/Interviewers, while the network reconnects, let me walk you directly through the architectural blueprint and empirical results of the platform:*
>
> *1. **The Core Thesis**: We proved that agricultural machine learning must be governed by temporal walk-forward evaluation. On our 51-year panel of 311 districts, machine learning achieved production readiness on **Oilseeds** (out-of-time MAE 549.67 kg/ha vs 616.60 kg/ha baseline, +10.85% gain, $p < 0.001$).*
>
> *2. **The Negative Result**: On **Rice and Wheat**, ML lost to historical district averages by up to 17.5% due to irrigation buffering. Our Certification Guard automatically rejects ML and routes those crops to certified statistical baselines.*
>
> *3. **Safety & Provenance**: Every forecast is bounded by an empirical P10–P90 tree dispersion interval and locked with an immutable SHA-256 cryptographic provenance hash linking the dataset version (`v2.1`) and model artifact hash.*
>
> *4. **Monitoring**: In production, the system tracks the Population Stability Index (PSI) to catch distribution drift before models degrade in the field.*
>
> *The entire codebase is verified by 581 automated pytest tests and ready for immediate GitHub inspection."*

---

## 5. Verified Screenshot & Backup Asset Index

All pre-rendered backup visual assets are cataloged in `docs/assets/` and `docs/demo/`:
- **Architecture Diagram**: `docs/demo/DAY39_ARCHITECTURE_DEFENSE.md` (Section 1)
- **Results Scorecard**: `docs/demo/DAY39_RESULTS.md` (Tables 1, 2, 3)
- **12-Slide Deck**: `docs/demo/DAY39_PRESENTATION.md`
- **Full Viva Q&A**: `docs/demo/DAY39_ML_VIVA.md`, `docs/demo/DAY39_AGRICULTURE_VIVA.md`, `docs/demo/DAY39_XAI_VIVA.md`, `docs/demo/DAY39_SYSTEM_DESIGN_VIVA.md`, `docs/demo/DAY39_HARD_QUESTIONS.md`.
