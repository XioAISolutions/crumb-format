#!/bin/bash
set -e

# Wave Field LLM Container Entrypoint Script
# Handles initialization, health checks, and graceful shutdown

echo "=========================================="
echo "Wave Field LLM Container Starting"
echo "=========================================="

# Environment variables with defaults
export WAVE_FIELD_ENV="${WAVE_FIELD_ENV:-production}"
export WAVE_FIELD_LOG_LEVEL="${WAVE_FIELD_LOG_LEVEL:-INFO}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/app/models}"
export DATA_DIR="${DATA_DIR:-/app/data}"
export LOG_DIR="${LOG_DIR:-/app/logs}"

# Print environment info
echo "Environment: ${WAVE_FIELD_ENV}"
echo "Log Level: ${WAVE_FIELD_LOG_LEVEL}"
echo "Model Cache: ${MODEL_CACHE_DIR}"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    export CUDA_AVAILABLE=true
else
    echo "No GPU detected, running in CPU mode"
    export CUDA_AVAILABLE=false
fi

# Create necessary directories
mkdir -p "${MODEL_CACHE_DIR}" "${DATA_DIR}" "${LOG_DIR}"

# Check if models directory is empty
if [ -z "$(ls -A ${MODEL_CACHE_DIR})" ]; then
    echo "WARNING: Models directory is empty. Please mount models volume."
fi

# Wait for dependent services
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=0

    echo "Waiting for ${service} at ${host}:${port}..."
    
    while ! nc -z "${host}" "${port}" 2>/dev/null; do
        attempt=$((attempt + 1))
        if [ ${attempt} -ge ${max_attempts} ]; then
            echo "ERROR: ${service} not available after ${max_attempts} attempts"
            return 1
        fi
        echo "Attempt ${attempt}/${max_attempts}: ${service} not ready, waiting..."
        sleep 2
    done
    
    echo "${service} is ready!"
    return 0
}

# Wait for Redis if configured
if [ -n "${REDIS_URL}" ]; then
    REDIS_HOST=$(echo ${REDIS_URL} | sed -n 's/.*:\/\/\([^:]*\).*/\1/p')
    REDIS_PORT=$(echo ${REDIS_URL} | sed -n 's/.*:\([0-9]*\).*/\1/p')
    wait_for_service "${REDIS_HOST}" "${REDIS_PORT}" "Redis" || echo "WARNING: Redis not available"
fi

# Wait for PostgreSQL if configured
if [ -n "${POSTGRES_URL}" ]; then
    POSTGRES_HOST=$(echo ${POSTGRES_URL} | sed -n 's/.*@\([^:]*\).*/\1/p')
    POSTGRES_PORT=$(echo ${POSTGRES_URL} | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    wait_for_service "${POSTGRES_HOST}" "${POSTGRES_PORT}" "PostgreSQL" || echo "WARNING: PostgreSQL not available"
fi

# Run database migrations if needed
if [ "${RUN_MIGRATIONS}" = "true" ]; then
    echo "Running database migrations..."
    python -m alembic upgrade head || echo "WARNING: Migration failed"
fi

# Warm up model cache if configured
if [ "${WARMUP_MODEL}" = "true" ]; then
    echo "Warming up model cache..."
    python -c "
from crumb_llm.inference import WaveFieldInference
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading model on {device}...')
inference = WaveFieldInference(device=device)
print('Model loaded successfully!')
" || echo "WARNING: Model warmup failed"
fi

# Set up signal handlers for graceful shutdown
shutdown() {
    echo "Received shutdown signal, gracefully stopping..."
    # Kill the main process
    kill -TERM "$child" 2>/dev/null
    wait "$child"
    echo "Shutdown complete"
    exit 0
}

trap shutdown SIGTERM SIGINT

# Execute the main command
echo "=========================================="
echo "Starting application: $@"
echo "=========================================="

# Run the command in background so we can trap signals
"$@" &
child=$!

# Wait for the child process
wait "$child"
exit_code=$?

echo "Application exited with code: ${exit_code}"
exit ${exit_code}

# Made with Bob
