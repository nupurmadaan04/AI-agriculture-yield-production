# Day 20 — Model Stability & Lineage Audit

## 1. Lineage Audit & Status Transition

A primary goal of Day 20 is to audit the stability of Day 19 models and explicitly track the transition from single-split validation to walk-forward cross-validation.

| Commodity | Day 19 Single Split Status (2016–2017) | Day 19 MAE Gain (%) | Day 20 Walk-Forward Status (2014–2017) | Fold Win Rate (%) | Mean MAE Gain (%) | Lineage Action & Scientific Rationale |
| :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| **Chickpea** | `BASELINE_PREFERRED` | -0.73% | `ROBUST_ACCEPTED` | 75.0% (3/4) | +2.32% | **UPGRADED**: Consistent outperformance across 2014, 2015, 2017 test origins (+6.11% median gain). |
| **Oilseeds** | `BASELINE_PREFERRED` | -4.80% | `ROBUST_ACCEPTED` | 75.0% (3/4) | +12.79% | **UPGRADED**: Strong multi-year signal (+12.79% mean gain, wins 3/4 folds). |
| **Sesamum** | `ACCEPTED` | +2.61% | `SPLIT_SENSITIVE` | 25.0% (1/4) | -1.19% | **DOWNGRADED**: Failed in 3/4 test origins (2014, 2015, 2017); won only in 2016. |
| **Maize** | `ACCEPTED` | +2.51% | `SPLIT_SENSITIVE` | 25.0% (1/4) | -2.63% | **DOWNGRADED**: Failed in 3/4 test origins; split-sensitive to drought year anomalies. |
| **Pigeonpea** | `ACCEPTED` | +1.25% | `BASELINE_PREFERRED` | 0.0% (0/4) | -5.68% | **DOWNGRADED**: Failed in 4/4 walk-forward folds; historical district mean dominates. |
| **Sugarcane** | `ACCEPTED` | +0.45% | `SPLIT_SENSITIVE` | 50.0% (2/4) | +1.17% | **DOWNGRADED**: Won in 2/4 folds, but inconsistent across test origins. |
| **Minor Pulses** | `BASELINE_PREFERRED` | -9.71% | `SPLIT_SENSITIVE` | 50.0% (2/4) | -5.76% | **MAINTAINED**: 50% win rate insufficient for production acceptance. |
| **Wheat** | `BASELINE_PREFERRED` | -8.76% | `SPLIT_SENSITIVE` | 50.0% (2/4) | -13.21% | **MAINTAINED**: High spatial inertia favors district mean baseline. |
| **Kharif Sorghum** | `BASELINE_PREFERRED` | -0.71% | `SPLIT_SENSITIVE` | 50.0% (2/4) | +1.30% | **MAINTAINED**: Inconsistent multi-origin win rate (2/4). |
| **Groundnut** | `BASELINE_PREFERRED` | -5.35% | `SPLIT_SENSITIVE` | 25.0% (1/4) | -7.87% | **MAINTAINED**: Statistical baseline superior in 3/4 folds. |
| **Sorghum** | `BASELINE_PREFERRED` | -1.89% | `SPLIT_SENSITIVE` | 25.0% (1/4) | -3.50% | **MAINTAINED**: Statistical baseline superior in 3/4 folds. |
| **Rapeseed & Mustard** | `BASELINE_PREFERRED` | -5.87% | `SPLIT_SENSITIVE` | 25.0% (1/4) | -5.62% | **MAINTAINED**: Statistical baseline superior in 3/4 folds. |
| **Rice (Multi-Crop)** | `BASELINE_PREFERRED` | -6.44% | `SPLIT_SENSITIVE` | 25.0% (1/4) | -8.39% | **MAINTAINED**: Note: Standalone Rice production model remains untouched. |
| **Pearl Millet** | `BASELINE_PREFERRED` | -12.39% | `BASELINE_PREFERRED` | 0.0% (0/4) | -6.04% | **MAINTAINED**: Baseline strictly dominates across all 4 folds. |

---

## 2. Stability Scoring Formula

The Composite Robustness Score ($S_{\text{rob}} \in [0, 100]$) is computed as follows:

$$S_{\text{rob}} = 0.40 \cdot \text{WinRate} + 0.35 \cdot \text{GainScore} + 0.25 \cdot \text{ConsistencyScore}$$

Where:
- $\text{WinRate} = \frac{\text{Wins}}{4} \times 100$
- $\text{GainScore} = \text{Clip}(50 + 2.5 \cdot \text{MeanGain}_{\%}, 0, 100)$
- $\text{ConsistencyScore} = \text{Clip}(100 - \frac{\sigma_{\text{MAE}}}{\mu_{\text{MAE}}} \times 100, 0, 100)$

### Classification Thresholds:
- **`ROBUST_ACCEPTED`**: $S_{\text{rob}} \ge 70.0$, $\text{Win Rate} \ge 75\%$, $\text{Mean Gain} > 0\%$.
- **`SPLIT_SENSITIVE`**: $45.0 \le S_{\text{rob}} < 70.0$, $25\% \le \text{Win Rate} < 75\%$.
- **`BASELINE_PREFERRED`**: $S_{\text{rob}} < 45.0$ or $\text{Win Rate} < 25\%$.
