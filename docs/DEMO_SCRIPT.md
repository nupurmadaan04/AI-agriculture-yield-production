# Live Demonstration Script (5–7 Minute Walkthrough)

## 1. Overview & Objective

This demonstration showcases the **AI Agriculture Decision Intelligence Platform**'s key technical strengths:
1. Multi-crop panel scale (29 crops, 71,601 records).
2. **Selective model deployment**: Showing that ML is only deployed where empirically superior to statistical baselines.
3. Governed production forecast serving with cryptographic provenance and append-oriented audit logging.

---

## 2. Step-by-Step Demonstration Flow

### Step 1: Platform Overview & High-Level Architecture (0:00 – 1:00)
- **Screen**: Navigate to Home / Overview (`/`).
- **Talking Points**:
  > *"Welcome to the Agricultural Intelligence Platform. This system addresses a major flaw in agricultural AI: universal, unverified machine learning forcing. We built an evidence-first governance framework on top of a longitudinal panel of 71,601 verified records across 29 crops and 311 Indian districts."*

---

### Step 2: Multi-Crop Panel Scale & Data Quality (1:00 – 2:00)
- **Screen**: Navigate to **National Portal** (`/portal`) or **Modeling Readiness** (`/modeling-readiness`).
- **Talking Points**:
  > *"Here is our standardized AGRI_PANEL_1.0 dataset. We cover 20 states and 311 districts over 52 consecutive years (1966–2017). Every single record has passed 14 automated data quality checks to guarantee physical boundary validity and zero future leakage."*

---

### Step 3: Model Certification — Production-Ready ML (Oilseeds) (2:00 – 3:15)
- **Screen**: Open **Modeling Readiness** (`/modeling-readiness`) $\rightarrow$ Tab **Model Certification** $\rightarrow$ Select **Oilseeds**.
- **Talking Points**:
  > *"Let's look at Oilseeds. Under 4-fold expanding walk-forward validation (2014–2017 origins), Random Forest achieved an MAE of 549.67 kg/ha compared to 616.60 kg/ha for the baseline. That represents a +10.85% error reduction and a 75% fold win rate. Because it met both certification gates, the governance engine granted it PRODUCTION_READY status."*

---

### Step 4: Model Governance in Action — Baseline Preferred (Rice) (3:15 – 4:30)
- **Screen**: In **Model Certification**, select **Rice**.
- **Talking Points**:
  > *"Now, look at Rice. Many systems would force a complex neural network or tree model here. But under temporal walk-forward validation, evaluated pre-season ML models failed to beat historical persistence averages. 
  > Rather than hallucinating complex predictions, our platform certifies the simpler **Historical District Mean Persistence Baseline**. Choosing the baseline when ML does not prove superior is a core feature of responsible AI engineering."*

---

### Step 5: Governed Forecast Serving Wizard (4:30 – 5:45)
- **Screen**: Navigate to **Forecast Serving** (`/forecast`).
- **Action**: 
  - Select Crop: `Oilseeds`
  - Select State: `Punjab`
  - Select District: `Ludhiana`
  - Target Year: `2018`
  - Click **Execute Governed Forecast**.
- **Talking Points**:
  > *"When we submit a forecast, the request passes through our Certification Guard. It verifies crop registration, geographic coverage in our historical panel, and model artifact integrity before routing.
  > Here is our result: 817.06 kg/ha. Notice that the system transparently displays the certification status, the primary strategy, and whether any fallback policy was applied."*

---

### Step 6: Cryptographic Provenance & Lineage (5:45 – 6:30)
- **Screen**: Expand the **Why This Prediction?** and **Cryptographic Provenance Lineage** panel on the result card.
- **Talking Points**:
  > *"Every forecast produces a full provenance record answering exactly where the number came from, the model artifact SHA-256 hash, historical validation MAE, and an SHA-256 lineage fingerprint. Users can copy this fingerprint to verify that the forecast was not altered after generation."*

---

### Step 7: Append-Oriented Prediction Audit Trail (6:30 – 7:00)
- **Screen**: Click on Tab **Prediction Audit Trail** on `/forecast`.
- **Talking Points**:
  > *"Finally, every inference event—both successful predictions and governance rejections—is written to an append-oriented audit log. This provides enterprise-grade observability and auditability for agricultural policy and risk underwriting."*
