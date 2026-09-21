# Master Documentation Index & Repository Guide

Welcome to the comprehensive documentation index for the **AI Agriculture Yield Prediction & Decision Intelligence Platform**. This index guides researchers, machine learning engineers, DevOps operators, policy evaluators, and auditors through all operational and scientific assets of the repository.

---

## 1. Documentation Map by Target Audience

```
+---------------------------------------------------------------------------------------------------------+
|                                    AUDIENCE-DRIVEN NAVIGATION GUIDE                                     |
+--------------------------+------------------------------------------------------------------------------+
| Audience Role            | Recommended Starting Points & Core Documents                                 |
+--------------------------+------------------------------------------------------------------------------+
| **Scientific Researcher**| - [Research Paper Manuscript](file:///docs/research_paper/paper.md)          |
|                          | - [Dataset Card](file:///docs/DATASET_CARD.md)                               |
|                          | - [Model Card](file:///docs/MODEL_CARD.md)                                   |
|                          | - [Experimental Design](file:///docs/research_paper/06_experimental_design.md)|
+--------------------------+------------------------------------------------------------------------------+
| **ML / Software Engineer**| - [Technical Overview](file:///docs/technical/TECHNICAL_OVERVIEW.md)         |
|                          | - [System Architecture](file:///docs/technical/SYSTEM_ARCHITECTURE.md)       |
|                          | - [Model Pipeline](file:///docs/technical/MODEL_PIPELINE.md)                 |
|                          | - [Forecast Serving](file:///docs/technical/FORECAST_SERVING.md)             |
|                          | - [XAI Implementation](file:///docs/technical/XAI.md)                        |
+--------------------------+------------------------------------------------------------------------------+
| **DevOps / SRE**         | - [Deployment Guide](file:///docs/technical/DEPLOYMENT.md)                   |
|                          | - [Day 34 Runbook](file:///docs/DAY34_RUNBOOK.md)                            |
|                          | - [Security Architecture](file:///docs/technical/SECURITY.md)               |
|                          | - [Testing Suites](file:///docs/technical/TESTING.md)                        |
+--------------------------+------------------------------------------------------------------------------+
| **Auditor / Evaluator**  | - [Master Claim Register](file:///docs/DAY36_CLAIM_REGISTER.md)              |
|                          | - [Reproducibility Guide](file:///docs/reproducibility/REPRODUCIBILITY_GUIDE.md)|
|                          | - [Claim-Evidence Matrix](file:///docs/reproducibility/CLAIM_EVIDENCE_MATRIX.md)|
|                          | - [Reproducibility Manifest](file:///docs/reproducibility/REPRODUCIBILITY_MANIFEST.md)|
+--------------------------+------------------------------------------------------------------------------+
| **Policy / Decision-Maker| - [Decision Intelligence](file:///docs/technical/DECISION_INTELLIGENCE.md)   |
|                          | - [Discussion & Policy](file:///docs/research_paper/12_discussion.md)        |
|                          | - [Research Limitations](file:///docs/research_paper/13_limitations.md)      |
+--------------------------+------------------------------------------------------------------------------+
```

---

## 2. Directory Structure & Core Modules

### 2.1 Research Paper Package (`docs/research_paper/`)
- [Package README](file:///docs/research_paper/README.md)
- [Monolithic Paper Manuscript](file:///docs/research_paper/paper.md)
- [BibTeX References](file:///docs/research_paper/references.bib)
- Modular Chapters:
  - [01: Abstract](file:///docs/research_paper/01_abstract.md)
  - [02: Introduction & Problem Framing](file:///docs/research_paper/02_introduction.md)
  - [03: Related Work Review](file:///docs/research_paper/03_related_work.md)
  - [04: Dataset Properties & Provenance](file:///docs/research_paper/04_dataset.md)
  - [05: Methodology & Pipeline](file:///docs/research_paper/05_methodology.md)
  - [06: Experimental Design & Protocols](file:///docs/research_paper/06_experimental_design.md)
  - [07: Empirical Results & Analysis](file:///docs/research_paper/07_results.md)
  - [08: Error Diagnostics & Regimes](file:///docs/research_paper/08_error_analysis.md)
  - [09: Explainability & Attribution](file:///docs/research_paper/09_explainability.md)
  - [10: Monitoring & Provenance](file:///docs/research_paper/10_monitoring_and_governance.md)
  - [11: Decision Intelligence](file:///docs/research_paper/11_decision_intelligence.md)
  - [12: Discussion & Findings](file:///docs/research_paper/12_discussion.md)
  - [13: Research Limitations](file:///docs/research_paper/13_limitations.md)
  - [14: Reproducibility & Audit](file:///docs/research_paper/14_reproducibility.md)
  - [15: Conclusion & Directions](file:///docs/research_paper/15_conclusion.md)
- Publication Figures (`docs/research_paper/figures/`):
  - [Figure 1: End-to-End Architecture](file:///docs/research_paper/figures/FIGURE_01_ARCHITECTURE.md)
  - [Figure 2: Dataset Coverage](file:///docs/research_paper/figures/FIGURE_02_DATASET_COVERAGE.md)
  - [Figure 3: Temporal Validation Design](file:///docs/research_paper/figures/FIGURE_03_TEMPORAL_VALIDATION.md)
  - [Figure 4: Strategy Decision Matrix](file:///docs/research_paper/figures/FIGURE_04_STRATEGY_MATRIX.md)
  - [Figure 5: Exogenous Ablation](file:///docs/research_paper/figures/FIGURE_05_EXOGENOUS_ABLATION.md)
  - [Figure 6: Empirical Uncertainty](file:///docs/research_paper/figures/FIGURE_06_UNCERTAINTY_COVERAGE.md)
- Publication Tables (`docs/research_paper/tables/`):
  - [Table 1: Dataset Summary](file:///docs/research_paper/tables/TABLE_01_DATASET_SUMMARY.md)
  - [Table 2: Validation Protocols](file:///docs/research_paper/tables/TABLE_02_VALIDATION_PROTOCOLS.md)
  - [Table 3: Multi-Crop Strategies](file:///docs/research_paper/tables/TABLE_03_MULTICROP_STRATEGIES.md)
  - [Table 4: Walk-Forward Results](file:///docs/research_paper/tables/TABLE_04_WALK_FORWARD_RESULTS.md)
  - [Table 5: Oilseeds Evidence](file:///docs/research_paper/tables/TABLE_05_OILSEEDS_EVIDENCE.md)
  - [Table 6: Sugarcane Evidence](file:///docs/research_paper/tables/TABLE_06_SUGARCANE_EVIDENCE.md)
  - [Table 7: Rice Benchmark](file:///docs/research_paper/tables/TABLE_07_RICE_BENCHMARK.md)
  - [Table 8: Exogenous Ablation](file:///docs/research_paper/tables/TABLE_08_EXOGENOUS_ABLATION.md)
  - [Table 9: Error Analysis](file:///docs/research_paper/tables/TABLE_09_ERROR_ANALYSIS.md)
  - [Table 10: Reproducibility Manifest](file:///docs/research_paper/tables/TABLE_10_REPRODUCIBILITY.md)
  - [Table 11: Claim-to-Evidence Matrix](file:///docs/research_paper/tables/TABLE_11_CLAIM_EVIDENCE.md)

### 2.2 Technical Documentation Package (`docs/technical/`)
- [Technical Overview](file:///docs/technical/TECHNICAL_OVERVIEW.md)
- [System Architecture](file:///docs/technical/SYSTEM_ARCHITECTURE.md)
- [Data Pipeline](file:///docs/technical/DATA_PIPELINE.md)
- [Model Pipeline](file:///docs/technical/MODEL_PIPELINE.md)
- [Forecast Serving Runtime](file:///docs/technical/FORECAST_SERVING.md)
- [Strategy Governance](file:///docs/technical/STRATEGY_GOVERNANCE.md)
- [Explainable AI (XAI)](file:///docs/technical/XAI.md)
- [Empirical Uncertainty](file:///docs/technical/UNCERTAINTY.md)
- [Observability & Monitoring](file:///docs/technical/MONITORING.md)
- [Decision Intelligence & Scenarios](file:///docs/technical/DECISION_INTELLIGENCE.md)
- [Cryptographic Provenance](file:///docs/technical/PROVENANCE.md)
- [Security & Hardening](file:///docs/technical/SECURITY.md)
- [Deployment & Disaster Readiness](file:///docs/technical/DEPLOYMENT.md)
- [Testing Suites & Contracts](file:///docs/technical/TESTING.md)
- [Reproducibility Specifications](file:///docs/technical/REPRODUCIBILITY.md)

### 2.3 Reproducibility & Audit Assets (`docs/reproducibility/`)
- [Master Reproducibility Guide](file:///docs/reproducibility/REPRODUCIBILITY_GUIDE.md)
- [Master Experiment Lineage](file:///docs/reproducibility/EXPERIMENT_LINEAGE.md)
- [Master Claim-Evidence Matrix](file:///docs/reproducibility/CLAIM_EVIDENCE_MATRIX.md)
- [Platform Reproducibility Manifest](file:///docs/reproducibility/REPRODUCIBILITY_MANIFEST.md)

### 2.4 Governance Cards & Reports
- [Dataset Card](file:///docs/DATASET_CARD.md)
- [Master Model Card](file:///docs/MODEL_CARD.md)
- [Day 36 Scientific Claim Register](file:///docs/DAY36_CLAIM_REGISTER.md)
- [Day 37 Research Documentation Report](file:///docs/DAY37_RESEARCH_DOCUMENTATION.md)
- [Day 38 Presentation & Portfolio Report](file:///docs/DAY38_GITHUB_PORTFOLIO.md)

### 2.5 Portfolio & Recruiter Package (`docs/portfolio/`)
- [Technical Project Case Study](file:///docs/portfolio/PROJECT_CASE_STUDY.md)
- [Recruiter One-Page Brief (<60s)](file:///docs/portfolio/PROJECT_ONE_PAGE.md)
- [Verified Resume Evidence](file:///docs/portfolio/RESUME_EVIDENCE.md)
- [Role-Specific Resume Bullets (DS / MLE / DA)](file:///docs/portfolio/RESUME_BULLETS.md)
- [LinkedIn Technical Post](file:///docs/portfolio/LINKEDIN_PROJECT_DESCRIPTION.md)
- [Portfolio Showcase Page](file:///docs/portfolio/PORTFOLIO_PAGE.md)
- [Project Chronological Story](file:///docs/portfolio/PROJECT_STORY.md)
- [Technical Interview Story & Q&A](file:///docs/portfolio/INTERVIEW_STORY.md)
- [Architecture System Design Interview Guide](file:///docs/portfolio/ARCHITECTURE_INTERVIEW.md)
- [Design Decisions: "Why Not Just X?"](file:///docs/portfolio/DESIGN_DECISIONS.md)
- [GitHub Presentation Metadata](file:///docs/portfolio/GITHUB_METADATA.md)

### 2.6 Demonstration Assets & Screenshots (`docs/assets/screenshots/`)
- [Screenshot Package & Golden Demo Checklist](file:///docs/assets/screenshots/README.md)

### 2.7 Technical Demonstration & Viva Defense Package (`docs/demo/`)
- [Live Technical Demonstration Script (0:00–7:00)](file:///docs/demo/DAY39_DEMO_SCRIPT.md)
- [60-Second Executive Pitch](file:///docs/demo/DAY39_60_SECOND_PITCH.md)
- [2-Minute Comprehensive Technical Explanation](file:///docs/demo/DAY39_2_MINUTE_EXPLANATION.md)
- [Layer-by-Layer Architecture & Data Flow Defense](file:///docs/demo/DAY39_ARCHITECTURE_DEFENSE.md)
- [Machine Learning & Data Science Viva Defense (36 Q&A)](file:///docs/demo/DAY39_ML_VIVA.md)
- [Agricultural Science & Agronomy Viva Defense (14 Q&A)](file:///docs/demo/DAY39_AGRICULTURE_VIVA.md)
- [Explainable AI (XAI) Viva Defense (10 Q&A)](file:///docs/demo/DAY39_XAI_VIVA.md)
- [System Design & Production Engineering Viva Defense (14 Q&A)](file:///docs/demo/DAY39_SYSTEM_DESIGN_VIVA.md)
- [Tough Skeptical Questions & Viva Defense (13 Q&A)](file:///docs/demo/DAY39_HARD_QUESTIONS.md)
- [Certified Scientific Results & Benchmark Audit](file:///docs/demo/DAY39_RESULTS.md)
- [12-Slide Technical Presentation & Speaker Notes](file:///docs/demo/DAY39_PRESENTATION.md)
- [Demo Preparation, Failover & Disaster Recovery Checklist](file:///docs/demo/DAY39_DEMO_CHECKLIST.md)
- [Day 39 Final Demonstration Audit & Status Report](file:///docs/DAY39_FINAL_DEMO_STATUS.md)

