# 11. Discussion

### 11.1 Addressing the Research Questions

The empirical findings of this research address five foundational questions in operational agricultural intelligence:

- **RQ1: Does crop-specific modeling improve predictive reliability over shared modeling and statistical baselines?**  
  *Finding*: Yes, but selectively. Crop-specific modeling is critical because commodities possess vastly different agro-ecological dynamics and response curves. However, machine learning improved over simple statistical baselines in only 2 of 14 evaluated crops under temporal walk-forward testing. In the remaining 12 crops, historical district mean persistence demonstrated superior or indistinguishable stability.
- **RQ2: How stable are model improvements across temporal walk-forward validation?**  
  *Finding*: Performance degrades significantly under climate shocks. Models that appeared highly accurate under single out-of-time holdouts (e.g. Sugarcane in Day 19) failed during drought shock years (Fold 2, 2015), exhibiting negative aggregate improvements unless bounded by operational variance clipping.
- **RQ3: Do tested pre-season exogenous variables improve forecasting under evaluated conditions?**  
  *Finding*: No. Under four-fold walk-forward validation across all 14 commodities, pre-season district weather aggregations produced no meaningful gain over autoregressive historical lags.
- **RQ4: Can forecast strategy selection be governed using empirical validation evidence?**  
  *Finding*: Yes. Implementing an automated strategy registry based on statistical gates (75% win-rate for unconstrained ML; 3-$\sigma$ fallback for conditional ML; historical mean for baselines) successfully prevents catastrophic out-of-distribution deployments.
- **RQ5: Can prediction provenance, explainability, monitoring, and decision evidence be integrated into an inspectable system?**  
  *Finding*: Yes. The platform demonstrates that sub-50ms inference can be served alongside full SHA-256 digital provenance, local perturbation attributions, PSI drift tracking, and bounded scenario simulations.

---

### 11.2 Scientific Contributions vs. Engineering Contributions

To maintain research integrity, scientific findings are explicitly separated from software engineering features:

```
+----------------------------------------------------------------------------------------------------+
|                                    CONTRIBUTION TAXONOMY                                           |
+----------------------------------------------------------------------------------------------------+
| SCIENTIFIC CONTRIBUTIONS:                                                                          |
| 1. Demonstration of the walk-forward evaluation necessity in multi-crop district forecasting.      |
| 2. Empirical proof of statistical baseline superiority in 12 of 14 major agricultural commodities. |
| 3. Resolution of Sugarcane volatility via operational 3-sigma variance clipping (+1.19% gain).    |
| 4. Comprehensive negative-result ablation demonstrating pre-season weather feature limitations.     |
| 5. Rigorous quantification of empirical P10-P90 tree dispersion intervals.                         |
+----------------------------------------------------------------------------------------------------+
| ENGINEERING CONTRIBUTIONS:                                                                         |
| 1. High-throughput, sub-50ms FastAPI forecast serving runtime with certification guards.           |
| 2. Cryptographic SHA-256 digital provenance hashing appended to immutable audit logs.              |
| 3. Population Stability Index (PSI) covariate drift monitoring pipeline.                           |
| 4. Production multi-container Docker Compose architecture with Nginx reverse proxy.              |
| 5. Accessible, WCAG 2.1 AA compliant Decision Workspace interface.                                |
+----------------------------------------------------------------------------------------------------+
```

---

### 11.3 Policy Implications

These findings have direct implications for public agricultural policy:
- **Avoid Uncritical ML Mandates**: Procurement and planning agencies should not mandate machine learning universally. Where historical record variance is low or data is sparse, simple statistical district persistence offers superior resilience against severe weather anomalies.
- **Mandate Operational Fallbacks**: Where machine learning models are deployed, production systems must enforce automated sanity bounds and variance clipping to catch catastrophic tail failures before forecasts reach human decision-makers.
