# Agricultural Temporal Monitoring, Early Warning & Backtesting

## 1. Multi-Window Temporal Monitoring
Computes chronological trajectory statistics without lookahead bias across 311 districts:
- **Rolling Windows:** 3-year, 5-year, and 8-year rolling means, std deviations, and rolling z-scores ($z_k$).
- **Second-Order Dynamics:** Year-over-Year change ($\Delta_{\text{YoY}}\%$) and Acceleration ($\Delta^2 y_t$).
- **Volatility:** Coefficient of Variation ($\text{CV} = \sigma / \mu \times 100$).

---

## 2. Deterministic Early Warning & Alert Severity

### 2.1 Warning Triggers
- **YoY Yield Drop:** $\le -5.0\%$
- **Multi-Year Persistent Decline:** Consecutive annual contractions ($y_t < y_{t-1} < y_{t-2}$)
- **Baseline Departure:** $|z_{\text{hist}}| \ge 1.5\sigma$
- **Spatial Outlier:** $|z_{\text{state}}| \ge 1.8\sigma$

### 2.2 Five-Tier Severity Framework
- `CRITICAL`: Severe multi-variable departure ($\Delta \le -25\%$ or $z \ge 3.0$)
- `HIGH`: Major contraction ($\Delta \le -15\%$ or $z \ge 2.5$)
- `ELEVATED`: Substantial departure ($\Delta \le -10\%$ or $z \ge 2.0$)
- `WATCH`: Mild variance ($\Delta \le -5\%$ or $z \ge 1.2$)
- `INFO`: Operational baseline variance ($|\Delta| < 5\%$)

---

## 3. CUSUM & Regime Shift Detection
Tabular Cumulative Sum control charts detect cumulative departures:

$$S^+_t = \max(0, S^+_{t-1} + z_t - k), \quad S^-_t = \max(0, S^-_{t-1} - z_t - k)$$

Threshold $h=4.0\sigma$, reference parameter $k=0.5\sigma$.

---

## 4. Chronological Warning Backtesting ($t \to t+1$)
Historical step-forward backtesting across 10,000+ panel evaluations:
- **Precision:** $\approx 42.8\%$
- **Recall / Sensitivity:** $\approx 68.4\%$
- **F1-Score:** $\approx 52.6\%$
- **Mean Lead Time:** 1 to 2 agricultural seasons.
