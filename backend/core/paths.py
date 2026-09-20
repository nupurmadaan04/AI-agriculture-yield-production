"""
Centralized Path Management.

Provides machine-independent, canonical Path objects for project root, models,
datasets, documents, reports, and scratch directories using pathlib.
"""

from pathlib import Path
import os


class ProjectPaths:
    """
    Encapsulates all filesystem paths for the platform.
    """
    def __init__(self, root: Path | None = None):
        if root is not None:
            self.root = root
        else:
            # Anchor to repository root (2 levels up from backend/core/paths.py)
            self.root = Path(__file__).resolve().parent.parent.parent

        self.model_dir = Path(os.getenv("MODEL_DIR", str(self.root / "Models"))).resolve()
        self.dataset_dir = Path(os.getenv("DATASET_DIR", str(self.root / "Datasets"))).resolve()
        self.docs_dir = (self.root / "docs").resolve()
        self.reports_dir = Path(os.getenv("REPORTS_DIR", str(self.root / "reports"))).resolve()
        self.backend_dir = (self.root / "backend").resolve()
        self.src_dir = (self.root / "src").resolve()
        self.tests_dir = (self.root / "tests").resolve()
        self.frontend_dir = (self.root / "frontend").resolve()

        # Ensure dynamic output directories exist safely
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    @property
    def canonical_dataset_path(self) -> Path:
        """Returns the primary cleaned panel dataset file path."""
        outlier_cleaned = self.dataset_dir / "rice_data_outlier_removed.csv"
        if outlier_cleaned.exists():
            return outlier_cleaned
        return self.dataset_dir / "rice_data.csv"

    @property
    def crops_dataset_path(self) -> Path:
        """Returns the multi-crop reference dataset file path."""
        return self.dataset_dir / "Crops_data.csv"

    def get_model_path(self, filename: str) -> Path:
        """Resolves path for a specific model artifact."""
        return self.model_dir / filename

    def get_report_path(self, filename: str) -> Path:
        """Resolves sanitized report path within controlled reports directory."""
        clean_name = Path(filename).name
        return self.reports_dir / clean_name


# Canonical singleton instance
paths = ProjectPaths()
