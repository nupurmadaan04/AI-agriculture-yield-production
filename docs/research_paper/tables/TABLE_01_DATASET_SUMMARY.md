# Table 1: Canonical Agricultural Dataset Summary

| Attribute | Parameter Value | Source / Verification Method |
|---|---|---|
| **Total Physical Observations** | 71,601 rows | `Datasets/processed/agricultural_panel.csv` (`len(df) == 71601`) |
| **Number of Distinct Crops** | 29 verified commodities | `df['crop'].nunique() == 29` |
| **State Administrative Units** | 20 Indian States | `df['state'].nunique() == 20` |
| **District Administrative Units** | 311 Districts (1966 Base Boundaries) | `df['district'].nunique() == 311` |
| **Temporal Coverage (Active Panel)** | 2010–2017 (8 crop years) | Continuous annual records |
| **Historical Baseline Context** | 1966–2017 | ICRISAT long-term agricultural repository |
| **Duplicate Keys** | 0 duplicate entries | Primary key: `(crop, state, district, year)` |
| **Physical File Size** | 11,563,368 bytes | Host filesystem inspection |
| **Cryptographic Hash (SHA-256)** | `13f882d7d4617e77b6ded31c7febb55e599f3f6a13e981f23b94c4cecd47f13b` | Bitwise verification hash |
