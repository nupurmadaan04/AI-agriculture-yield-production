"""
Environment and Production Configuration.

Manages configuration parameters from environment variables with strong typing,
sensible development defaults, and strict production security.
"""

import os
from typing import List
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """
    Application configuration model.
    """
    app_name: str = "Agricultural Decision Intelligence Platform API"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    api_prefix: str = "/api"
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"))

    # CORS Configuration
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
            ).split(",")
            if origin.strip()
        ]
    )

    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    # Cache Configuration
    cache_ttl_seconds: int = Field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))
    enable_cache: bool = Field(default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1", "yes"))

    # Optional External Services
    gemini_api_key: str | None = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.5-flash"))

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in ("production", "prod")

    @property
    def is_testing(self) -> bool:
        return self.app_env.lower() in ("test", "testing")


settings = Settings()
