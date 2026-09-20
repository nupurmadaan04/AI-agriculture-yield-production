# Day 22: Scientific Validation & Statistical Proofs

## 1. Experimental Integrity Verification
1. **Zero Future Target Contamination**: Verified that no June–September precipitation, post-sowing NDVI, or concurrent harvest records entered pre-season models.
2. **Expanding Walk-Forward Splitting**: All 5 ablation tiers and 3 model families evaluated on identical temporal partitions (Folds 1–4, Origins 2014–2017).
3. **Data Provenance Preservation**: All raw weather data linked to documented public registries (IMD, NASA POWER/ERA5, ICRISAT).
4. **Historical Day 9 Rice Benchmark**: Intact ($R^2 = 0.7866$, $\text{MAE} = 353.01\,\text{kg/ha}$).

---

## 2. Statistical Breakdown of Exogenous Ineffectiveness
- **Information Decay Hypothesis**: Pre-monsoon showers (Jan–May) exhibit low mutual information ($I(X; Y) < 0.04$) with final Kharif yields, which depend overwhelmingly on August–September monsoon distribution.
- **Dimensionality Penalty**: Increasing feature count from 7 to 17 without strong signal leads to overfitting on the training partition and increased out-of-fold generalization variance.
