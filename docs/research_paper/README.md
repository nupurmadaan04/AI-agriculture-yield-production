# Research Paper Package: Evidence-Governed Agricultural Forecasting

This directory contains the complete research paper manuscript, modular chapters, figures, tables, and bibliographic references for:

**"An Evidence-Governed Agricultural Forecasting and Decision Intelligence Framework for District-Level Multi-Crop Analysis"**

---

## Directory Structure

```
docs/research_paper/
├── README.md                          # Package overview and build instructions
├── 01_abstract.md                     # Abstract and key findings
├── 02_introduction.md                 # Context, challenges, analytical task taxonomy
├── 03_related_work.md                 # Grounded academic literature review
├── 04_dataset.md                      # Canonical panel properties, provenance, schema
├── 05_methodology.md                  # Leakage-safe pipeline and feature definitions
├── 06_experimental_design.md          # Expanding walk-forward protocols (A-G)
├── 07_results.md                      # Multi-crop tournament, Rice, Oilseeds, Sugarcane
├── 08_error_analysis.md               # Quantile errors, regime breakdown, drought shocks
├── 09_explainability.md               # Marginal Reference Perturbation Attribution
├── 10_monitoring_and_governance.md    # PSI covariate drift and certification routing
├── 11_decision_intelligence.md        # Evidence synthesis and scenario boundaries
├── 12_discussion.md                   # Statistical baselines vs ML; policy implications
├── 13_limitations.md                  # Temporal, spatial, causal, and telemetry limits
├── 14_reproducibility.md              # Cryptographic hashes and test verification
├── 15_conclusion.md                   # Summary and future census recommendations
├── references.bib                     # Formal BibTeX citations
├── paper.md                           # Fully assembled monolithic manuscript
├── figures/                           # Publication-grade Markdown/ASCII diagrams
└── tables/                            # Publication-ready Markdown empirical tables
```

---

## Artifact Mapping

Every chapter and table directly maps to immutable project artifacts:

| Chapter / Table | Supporting Repository Artifact | Verification Test |
|---|---|---|
| `04_dataset.md` / `TABLE_01` | `Datasets/processed/agricultural_panel.csv` | `test_canonical_agricultural_panel_physical_counts` |
| `06_experimental_design.md` / `TABLE_02` | `Datasets/metadata/multicrop_fold_results.csv` | `test_oilseeds_governed_ml_evidence_chain` |
| `07_results.md` (Rice) / `TABLE_07` | `Models/forecasting_model_metadata.json` | `test_rice_legacy_benchmark_metrics` |
| `07_results.md` (Oilseeds) / `TABLE_05` | `Datasets/metadata/multicrop_model_selection.csv` | `test_oilseeds_governed_ml_evidence_chain` |
| `07_results.md` (Sugarcane) / `TABLE_06` | `Datasets/metadata/final_model_certification.csv` | `test_sugarcane_dual_metrics_resolution` |
| `07_results.md` (Exogenous) / `TABLE_08` | `Datasets/metadata/exogenous_model_selection.csv` | `test_day22_exogenous_negative_result_invariants` |
| `09_explainability.md` | `src/explainability_engine.py` | `test_explainability_engine_structure` |
| `10_monitoring_and_governance.md` | `src/monitoring_service.py` | `test_drift_psi_computation` |
| `14_reproducibility.md` / `TABLE_10` | `docs/DAY36_REPRODUCIBILITY_MANIFEST.md` | `test_model_artifacts_and_dataset_manifest_integrity` |

---

## Compilation & Verification

The monolithic manuscript `paper.md` is compiled sequentially from chapters `01` through `15` and references `references.bib`. All quantitative claims within the text have been verified against the master claim register in `docs/DAY36_CLAIM_REGISTER.md`.
