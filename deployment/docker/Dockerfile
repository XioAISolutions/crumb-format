# Multi-stage production Dockerfile for Wave Field LLM
# Optimized for size, security, and performance

# ============================================================================
# Stage 1: Builder - Compile dependencies and prepare application
# ============================================================================
FROM python:3.11-slim as builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for layer caching
COPY requirements.txt /tmp/requirements.txt
COPY pyproject.toml /tmp/pyproject.toml

# Install Python dependencies
WORKDIR /tmp
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for smaller image, override for GPU)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY crumb_llm/ /app/crumb_llm/
COPY api/ /app/api/
COPY cli/ /app/cli/
COPY setup.py /app/
COPY README.md /app/

# Install the application
WORKDIR /app
RUN pip install --no-cache-dir -e .

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Metadata
LABEL maintainer="Wave Field LLM Team"
LABEL description="Production deployment of Wave Field LLM"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    WAVE_FIELD_ENV=production \
    WAVE_FIELD_LOG_LEVEL=INFO

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r wavefield && \
    useradd -r -g wavefield -u 1000 -m -s /bin/bash wavefield

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application from builder
COPY --from=builder /app /app

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/cache && \
    chown -R wavefield:wavefield /app

# Set working directory
WORKDIR /app

# Copy entrypoint script
COPY deployment/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown wavefield:wavefield /entrypoint.sh

# Switch to non-root user
USER wavefield

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Volume mounts for models and data
VOLUME ["/app/models", "/app/data", "/app/logs"]

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run API server
CMD ["python", "-m", "api.server", "--host", "0.0.0.0", "--port", "8000"]

# Made with Bob
