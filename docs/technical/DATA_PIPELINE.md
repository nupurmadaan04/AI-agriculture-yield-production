# Data Pipeline Architecture

## 1. Ingestion & Preprocessing

The ingestion pipeline converts raw district ICRISAT tables and government statistics into the canonical long panel:
1. **Administrative Alignment**: Standardizes historical state and district names across 1966 base boundaries, resolving district bifurcations and renamings (e.g. `Bijapur / Vijayapura`).
2. **Unit Conversion**: Converts area to gross hectares ($\text{ha}$) and production to metric tonnes ($\text{t}$).
3. **Algebraic Consistency**: Verifies physical yield calculation $\text{Yield} = \frac{\text{Production}}{\text{Area}} \times 1000$. Zero production with positive area is permitted; zero area with positive production is rejected as corrupt.

---

## 2. Feature Pipeline & Leakage Isolation

Predictive features are extracted strictly using pre-season information:
- **Autoregressive Lags**: $y_{t-1} = \text{Yield}(\text{year}=t-1)$.
- **Rolling Windows**: $\bar{y}_{t-1:t-3} = \frac{1}{3}\sum_{k=1}^3 y_{t-k}$.
- **Cultivated Area Share**: District crop area divided by total district cropped area.
- **Strict Leakage Barrier**: Contemporaneous harvest-year production $P_t$ is strictly masked from the feature pipeline.
