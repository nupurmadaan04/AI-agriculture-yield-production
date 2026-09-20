# Agricultural Data Methodology & Preprocessing Pipeline

## 1. Primary Dataset Overview
The platform is grounded in the verified **ICRISAT District-Level Agricultural Panel Dataset**, covering agricultural observations across India from **1966 to 2017**:
- **Geographic Breadth:** 20 major agricultural states and 311 distinct agricultural districts.
- **Temporal Span:** 52 consecutive annual seasons.
- **Completeness:** 100% verified complete records for key agronomic variables.

---

## 2. Data Cleaning & Outlier Treatment

### 2.1 Outlier Removal Strategy
Agricultural crop yields exhibit empirical physiological boundaries. An empirical IQR (Interquartile Range) filtering strategy was executed on historical records:
- Lower Bound: $Q_1 - 1.5 \times \text{IQR}$
- Upper Bound: $Q_3 + 1.5 \times \text{IQR}$
- Extreme physical outliers (e.g. negative yields, typographical errors exceeding 10,000 kg/ha) were removed to yield `Datasets/rice_data_outlier_removed.csv`.

---

## 3. Feature Engineering & Temporal Split Protocols

### 3.1 Strict Anti-Leakage Protocol
- **Target Leakage Prohibition:** Features derived from post-harvest measurements (e.g. $\text{RICE PRODUCTION} / \text{RICE AREA}$) are strictly banned from model inputs.
- **Temporal Holdout Evaluation:** Out-of-time evaluation splits train models on records $\le 2015$ and test exclusively on unseen future years ($2016–2017$) to eliminate lookahead bias.
- **Scaler Fitting:** Standard scalers are fitted strictly on training partitions and applied to testing/inference partitions.

---

## 4. State Code Mapping Standard

| State Name | State Code | State Name | State Code |
|---|---|---|---|
| Andhra Pradesh | 1 | Madhya Pradesh | 9 |
| Assam | 2 | Maharashtra | 10 |
| Bihar | 3 | Orissa | 11 |
| Gujarat | 4 | Punjab | 12 |
| Haryana | 5 | Rajasthan | 13 |
| Himachal Pradesh | 6 | Chhattisgarh | 14 |
| Karnataka | 7 | Jharkhand | 15 |
| Kerala | 8 | Tamil Nadu | 16 |
| Uttar Pradesh | 17 | Uttarakhand | 18 |
| West Bengal | 19 | Telangana | 20 |
