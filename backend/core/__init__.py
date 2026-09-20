"""
Core production infrastructure package for Agricultural Decision Intelligence Platform.
"""

from backend.core.paths import paths
from backend.core.config import settings
from backend.core.version import (
    APPLICATION_VERSION,
    DATASET_VERSION,
    MODEL_VERSION,
    METHODOLOGY_VERSION,
    API_VERSION
)

__all__ = [
    "paths",
    "settings",
    "APPLICATION_VERSION",
    "DATASET_VERSION",
    "MODEL_VERSION",
    "METHODOLOGY_VERSION",
    "API_VERSION"
]
