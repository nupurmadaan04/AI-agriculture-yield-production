# Day 20 — Scientific Validation & Hypothesis Testing

## 1. Scientific Principles & Null Hypotheses

### Primary Hypothesis $H_0$:
$$H_0: \text{Candidate ML models do NOT provide statistically significant or temporally robust yield forecast improvements over historical district baselines across expanding validation origins.}$$

### Alternative Hypothesis $H_1$:
$$H_1: \text{Candidate ML models provide consistent outperformance (\text{Win Rate} \ge 75\%, \text{Mean Gain} > 0\%) across multiple temporal forecasting origins.}$$

---

## 2. Hypothesis Test Outcomes

| Commodity | $H_0$ Decision | Validated Status | Scientific Evidence |
| :--- | :--- | :--- | :--- |
| **Chickpea** | **REJECT $H_0$** | `ROBUST_ACCEPTED` | Outperformed baseline in 3/4 test origins (+6.11% median MAE improvement). |
| **Oilseeds** | **REJECT $H_0$** | `ROBUST_ACCEPTED` | Outperformed baseline in 3/4 test origins (+12.79% mean MAE improvement). |
| **Sugarcane** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in only 2/4 origins; outperformance is split-sensitive. |
| **Kharif Sorghum** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in only 2/4 origins; outperformance is split-sensitive. |
| **Minor Pulses** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in only 2/4 origins; baseline preferred. |
| **Wheat** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in only 2/4 origins; district mean baseline is more robust. |
| **Sesamum** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Day 19 acceptance invalidated: won 1/4 folds, lost 3/4 folds under walk-forward. |
| **Maize** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Day 19 acceptance invalidated: won 1/4 folds, lost 3/4 folds under walk-forward. |
| **Groundnut** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in 1/4 folds; baseline dominant. |
| **Rapeseed & Mustard** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in 1/4 folds; baseline dominant. |
| **Sorghum** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in 1/4 folds; baseline dominant. |
| **Rice (Multi-Crop)** | **FAIL TO REJECT $H_0$** | `SPLIT_SENSITIVE` | Won in 1/4 folds; baseline dominant. |
| **Pearl Millet** | **FAIL TO REJECT $H_0$** | `BASELINE_PREFERRED` | Won 0/4 folds; baseline strictly dominates. |
| **Pigeonpea** | **FAIL TO REJECT $H_0$** | `BASELINE_PREFERRED` | Day 19 acceptance invalidated: won 0/4 folds; baseline strictly dominates. |

---

## 3. Scientific Integrity Conclusion

The platform enforces absolute empirical honesty:
- **No confirmation bias**: Models that looked acceptable on a single split are downgraded without hesitation when multi-origin evidence fails.
- **Production Safety**: Only models meeting the strict $75\%$ win rate threshold with positive multi-fold gain are deployed for production ML forecasting (`Chickpea`, `Oilseeds`).
- **Baseline Transparency**: For crops where statistical baselines dominate, the platform presents statistical baselines with pride rather than forcing fragile ML models.
