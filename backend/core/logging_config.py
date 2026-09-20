"""
Structured Logging Configuration.

Provides formatted logging with timestamp, level, request_id, endpoint,
duration, status, service, and error codes.
"""

import logging
import sys
import json
from typing import Any, Dict
from backend.core.config import settings


class StructuredJsonFormatter(logging.Formatter):
    """
    Formats log records into structured JSON.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach custom contextual fields if present
        for field in ("request_id", "endpoint", "method", "duration_ms", "status_code", "service", "error_code"):
            if hasattr(record, field):
                log_obj[field] = getattr(record, field)

        if record.exc_info and not settings.is_production:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logging():
    """Configures root application logger with structured formatting."""
    log_level = getattr(logging, settings.log_level, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.is_production:
        formatter = StructuredJsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Quiet external noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return root_logger


logger = logging.getLogger("agricultural_platform")
