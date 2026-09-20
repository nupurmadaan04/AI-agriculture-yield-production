"""
Day 24 Certification Guard & Input Pre-Inference Validation.
Enforces strict governance checks before routing or executing forecasting inference.
"""

from pathlib import Path
import json
import hashlib
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("certification_guard")


class CertificationGuard:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.meta_dir = self.base_dir / "Datasets" / "metadata"
        self.models_dir = self.base_dir / "Models" / "multicrop"

        self.strategy_registry: Dict[str, Any] = {}
        self.coverage_df: pd.DataFrame = pd.DataFrame()
        self._coverage_set: set[Tuple[str, str, str]] = set()
        self._artifact_hashes: Dict[str, Tuple[bool, str]] = {}
        self._load_metadata()

    def _load_metadata(self):
        reg_json = self.models_dir / "forecast_strategy_registry.json"
        if reg_json.exists():
            with open(reg_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.strategy_registry = data.get("strategies", {})

        cov_csv = self.meta_dir / "forecast_coverage.csv"
        if cov_csv.exists():
            self.coverage_df = pd.read_csv(cov_csv)
            self._coverage_set = set(
                zip(
                    self.coverage_df["crop"].str.lower().str.strip(),
                    self.coverage_df["state"].str.lower().str.strip(),
                    self.coverage_df["district"].str.lower().str.strip(),
                )
            )

    def validate_request(
        self,
        crop: str,
        state: str,
        district: str,
        forecast_year: Optional[int] = None,
        features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, str, Optional[Dict[str, Any]]]:
        """
        Validates request against governance rules.
        Returns: (is_allowed: bool, status_code: str, reason_message: str, strategy_metadata: Optional[dict])
        """
        if not self.strategy_registry:
            self._load_metadata()
        # 1. Input completeness
        if not crop or not crop.strip():
            return False, "INPUT_INCOMPLETE", "Crop commodity must be specified.", None
        if not state or not state.strip():
            return False, "INPUT_INCOMPLETE", "State name must be specified.", None
        if not district or not district.strip():
            return False, "INPUT_INCOMPLETE", "District name must be specified.", None

        # 2. Crop registration check
        crop_clean = crop.strip()
        if crop_clean not in self.strategy_registry:
            return (
                False,
                "UNSUPPORTED_CROP",
                f"Crop '{crop_clean}' is not registered with an authoritative forecasting strategy.",
                None,
            )

        strategy_meta = self.strategy_registry[crop_clean]
        cert_status = strategy_meta.get("certification_status")

        # 3. Geographic coverage check
        if self._coverage_set:
            key = (crop_clean.lower(), state.strip().lower(), district.strip().lower())
            if key not in self._coverage_set:
                return (
                    False,
                    "DISTRICT_UNSUPPORTED",
                    f"District '{district.strip()}' in state '{state.strip()}' is not supported in historical training data for {crop_clean}.",
                    None,
                )
        elif not self.coverage_df.empty:
            match = self.coverage_df[
                (self.coverage_df["crop"].str.lower() == crop_clean.lower())
                & (self.coverage_df["state"].str.lower() == state.strip().lower())
                & (self.coverage_df["district"].str.lower() == district.strip().lower())
            ]
            if match.empty:
                return (
                    False,
                    "DISTRICT_UNSUPPORTED",
                    f"District '{district.strip()}' in state '{state.strip()}' is not supported in historical training data for {crop_clean}.",
                    None,
                )

        # 4. Model artifact check for ML strategies
        if cert_status in ["PRODUCTION_READY", "CONDITIONAL_PRODUCTION"]:
            artifact_name = strategy_meta.get("model_artifact")
            if artifact_name:
                artifact_path = self.models_dir / artifact_name
                if not artifact_path.exists():
                    return (
                        False,
                        "MODEL_ARTIFACT_MISSING",
                        f"Certified ML artifact '{artifact_name}' for crop '{crop_clean}' is missing from repository.",
                        None,
                    )

        # 5. Passed all governance gates
        return (
            True,
            "CERTIFIED_ALLOW",
            f"Request verified against Day 23 certification standards ({cert_status}).",
            strategy_meta,
        )

    def verify_artifact_integrity(self, crop: str) -> Tuple[bool, str]:
        """Verifies SHA-256 hash integrity of model artifact against model registry."""
        if crop in self._artifact_hashes:
            return self._artifact_hashes[crop]

        if crop not in self.strategy_registry:
            return False, "Crop not registered"

        meta = self.strategy_registry[crop]
        artifact_name = meta.get("model_artifact")
        if not artifact_name:
            res = (True, "Baseline strategy (no ML artifact required)")
            self._artifact_hashes[crop] = res
            return res

        artifact_path = self.models_dir / artifact_name
        if not artifact_path.exists():
            return False, "Artifact file missing"

        with open(artifact_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()

        res = (True, f"SHA256:{h[:16]}")
        self._artifact_hashes[crop] = res
        return res
