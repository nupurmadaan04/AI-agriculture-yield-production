# Multi-Crop Agricultural Data Governance Specification

## 1. Source Hierarchy & Ingestion Protocol
1. **Primary Source**: ICRISAT District Level Database (DLD). Ingested locally with SHA-256 integrity verification.
2. **Secondary Source**: Government of India Open Government Data (OGD) Directorate of Economics & Statistics (DES).
3. **Tertiary / Benchmark Source**: FAOSTAT Crop and Livestock Products.

---

## 2. Standardization & Harmonization Rules

### 2.1 Crop Taxonomy
- Raw survey column prefixes are standardized into canonical capitalized crop names (`RICE` $\rightarrow$ `Rice`, `WHEAT` $\rightarrow$ `Wheat`, `CHICKPEA` $\rightarrow$ `Chickpea`).
- Any unmapped crop remains `UNMAPPED` rather than silently merged.

### 2.2 Geography
- Standardized to 1966 baseline district boundaries across 20 Indian states to guarantee longitudinal panel continuity.
- State and District names are sanitized to Title Case without synthetic boundary extrapolation.

### 2.3 Unit Conversions
- **Area**: Converted from `1000 ha` to hectares (`ha`) via exact $\times 1000$ multiplication.
- **Production**: Converted from `1000 tons` to metric tonnes (`tonnes`) via exact $\times 1000$ multiplication.
- **Yield**: Expressed in kilograms per hectare (`kg/ha`). Computed as $(\text{Production in kg}) / (\text{Area in ha})$ where yield is missing and area $> 0$.

---

## 3. Data Quality & Anomaly Handling Policy
- 14 automated quality checks are executed prior to panel release.
- Survey anomalies where area is rounded down to $0.0$ but production $> 0$ are sanitized to avoid division-by-zero errors.
- Extreme historical observations are preserved for empirical validity rather than silently trimmed.

---

## 4. Dataset Versioning & Provenance
- Active Dataset Version: **`AGRI_PANEL_1.0`**
- Total Unified Panel Records: **71,601**
- Verified Crop Commodities: **29**
- Quality Audit Status: **`PASS` (14/14 checks passed)**
