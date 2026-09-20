# =============================================================================
# Agricultural Decision Intelligence Platform — Production Backend Dockerfile
# =============================================================================

FROM python:3.11-slim AS base

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Install Python dependencies
COPY Requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r Requirements.txt

# Copy application source and assets
COPY backend/ ./backend/
COPY src/ ./src/
COPY Models/ ./Models/
COPY Datasets/ ./Datasets/
COPY docs/ ./docs/

# Create runtime directories and set ownership
RUN mkdir -p /app/reports /app/Datasets/metadata && \
    chown -R appuser:appgroup /app

# Default production environment settings
ENV APP_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
