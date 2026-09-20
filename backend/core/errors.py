"""
Structured API Errors and Exceptions.

Defines standardized error responses and custom domain exceptions.
Guarantees zero leakage of stack traces, local paths, or internal credentials.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Structured error payload schema."""
    code: str = Field(..., description="Machine-readable uppercase error code")
    message: str = Field(..., description="Human-readable explanation of error")
    details: Dict[str, Any] = Field(default_factory=dict, description="Contextual validation details")
    request_id: Optional[str] = Field(None, description="Unique trace request ID")


class StructuredErrorResponse(BaseModel):
    """Standardized top-level API error envelope."""
    error: ErrorDetail


class BasePlatformError(Exception):
    """Base class for all structured platform exceptions."""
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ModelUnavailableError(BasePlatformError):
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Model artifact '{model_name}' is currently unavailable or uninitialized.",
            code="MODEL_UNAVAILABLE",
            status_code=503,
            details=details
        )


class DataNotFoundError(BasePlatformError):
    def __init__(self, resource: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Requested agricultural resource '{resource}' was not found.",
            code="DATA_NOT_FOUND",
            status_code=404,
            details=details
        )


class FeatureSchemaMismatchError(BasePlatformError):
    def __init__(self, missing_features: list, extra_features: list = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Input feature vector does not match registered model schema. Missing: {missing_features}.",
            code="FEATURE_SCHEMA_MISMATCH",
            status_code=422,
            details={"missing": missing_features, "extra": extra_features or [], **(details or {})}
        )


class UnsupportedContextError(BasePlatformError):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Requested agricultural context is unsupported: {reason}",
            code="UNSUPPORTED_CONTEXT",
            status_code=400,
            details=details
        )


class ReportGenerationError(BasePlatformError):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Failed to generate decision evidence report: {reason}",
            code="REPORT_GENERATION_ERROR",
            status_code=500,
            details=details
        )
