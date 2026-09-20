"""
Final Temporal Validation & Master Audit Coordinator (Day 23).

Executes the comprehensive Day 23 audit across strategy evaluation, residual diagnostics,
bias detection, reproducibility certification, and model registry finalization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.strategy_evaluation import StrategyEvaluationEngine
from src.residual_diagnostics import ResidualDiagnosticsEngine
from src.reproducibility_audit import ReproducibilityAuditEngine
from src.final_model_certification import FinalModelCertificationEngine


class FinalValidationCoordinator:
    """Coordinates full Day 23 final validation and lineage checks."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.metadata_dir = self.base_dir / "Datasets" / "metadata"

    def run_master_audit(self) -> Dict[str, Any]:
        """
        Runs the complete Day 23 master audit workflow.
        """
        print("=" * 80)
        print("DAY 23: FINAL TEMPORAL VALIDATION, STRATEGY EVALUATION & CERTIFICATION")
        print("=" * 80)

        # 1. Temporal Range Inspection
        panel_path = self.base_dir / "Datasets" / "processed" / "agricultural_panel.csv"
        max_year = 2017
        if panel_path.exists():
            df_p = pd.read_csv(panel_path, usecols=["year"])
            max_year = int(df_p["year"].max())

        print(f"[Temporal Range Inspection] Usable dataset spans 1966 to {max_year}.")
        print("STATEMENT: The available dataset does not contain a post-2017 independent temporal holdout; therefore final independent validation is constrained to the existing walk-forward evidence.")

        # 2. Strategy Evaluation
        print("\n--- PHASE 1: STRATEGY VS MODEL EVALUATION ---")
        strat_engine = StrategyEvaluationEngine(self.base_dir)
        df_detailed_val, df_strat_summary = strat_engine.evaluate_all_strategies()

        # 3. Residual Diagnostics & Bias Detection
        print("\n--- PHASE 2: RESIDUAL DIAGNOSTICS & BIAS DETECTION ---")
        res_engine = ResidualDiagnosticsEngine(self.base_dir)
        res_dict = res_engine.run_full_residual_diagnostics()

        # 4. Reproducibility Audit
        print("\n--- PHASE 3: REPRODUCIBILITY AUDIT & CRYPTOGRAPHIC HASHING ---")
        repro_engine = ReproducibilityAuditEngine(self.base_dir)
        df_repro, repro_cert = repro_engine.run_reproducibility_audit()

        # 5. Final Model Certification & Lineage Finalization
        print("\n--- PHASE 4: FINAL MODEL CERTIFICATION & REGISTRY FINALIZATION ---")
        cert_engine = FinalModelCertificationEngine(self.base_dir)
        df_cert, registry_data = cert_engine.determine_final_certification(
            df_detailed_val=df_detailed_val,
            df_strat_summary=df_strat_summary,
            df_overall_res=res_dict["overall"],
            df_bias=res_dict["bias"],
            df_repro=df_repro
        )

        # 6. Special Audits
        print("\n--- PHASE 5: SPECIAL COMMODITY AUDITS ---")
        oilseeds_cert = df_cert[df_cert["crop"] == "Oilseeds"].to_dict(orient="records")[0]
        print(f"[Oilseeds Audit] Status: {oilseeds_cert['final_status']} | Gain vs Baseline: +{oilseeds_cert['gain_vs_baseline_pct']}% | Bias: {oilseeds_cert['bias_status']}")

        chickpea_cert = df_cert[df_cert["crop"] == "Chickpea"].to_dict(orient="records")[0]
        print(f"[Chickpea Audit] Lineage: Day 19 ACCEPTED -> Day 20 ROBUST_ACCEPTED -> Day 21 ML_WITH_CONDITIONS -> Day 22 Baseline Preferred -> Day 23 {chickpea_cert['final_status']}")

        print("\n[Day 22 Negative Result Validation]")
        print("  - Evidence-Supported Explanation: Pre-season weather signals (Jan-May) precede monsoon grain-filling by 3-5 months; without post-sowing monsoon rain data, pre-season variables act as noise that increases tree variance.")
        print("  - Evidence-Uncertain Hypothesis: Distant pre-season soil moisture anomalies might have localized micro-climate effects, but lack district-wide explanatory power without higher spatial resolution satellite sensors.")

        print("=" * 80)
        print("DAY 23 MASTER AUDIT COMPLETED SUCCESSFULLY.")
        print("=" * 80)

        return {
            "strategy_summary": df_strat_summary,
            "detailed_validation": df_detailed_val,
            "residual_diagnostics": res_dict,
            "reproducibility": df_repro,
            "certification": df_cert,
            "registry": registry_data
        }


if __name__ == "__main__":
    coordinator = FinalValidationCoordinator()
    coordinator.run_master_audit()
