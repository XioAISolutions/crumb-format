#!/bin/bash
set -euo pipefail

# Wave Field LLM - Quick Start Script
# Rapid local deployment for testing and development

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
MODEL_NAME="wavefield-small"
PORT=8000
SKIP_MODEL_DOWNLOAD=false
USE_GPU=false
CLEAN_START=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}\n"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Quick start script for local Wave Field LLM deployment

Options:
    --model MODEL           Model to use (default: wavefield-small)
    --port PORT             Port to expose (default: 8000)
    --gpu                   Use GPU acceleration
    --skip-download         Skip model download (use existing)
    --clean                 Clean start (remove existing containers)
    -h, --help              Show this help message

Examples:
    # Basic quick start
    $0

    # Use specific model with GPU
    $0 --model wavefield-medium --gpu

    # Clean start on custom port
    $0 --clean --port 8080

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL_NAME="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --gpu)
                USE_GPU=true
                shift
                ;;
            --skip-download)
                SKIP_MODEL_DOWNLOAD=true
                shift
                ;;
            --clean)
                CLEAN_START=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done
}

check_prerequisites() {
    log_step "Checking Prerequisites"

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    if [[ "$USE_GPU" == true ]]; then
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            log_warning "GPU support requested but not available. Falling back to CPU."
            USE_GPU=false
        fi
    fi

    log_success "Prerequisites check passed"
}

clean_existing() {
    if [[ "$CLEAN_START" == true ]]; then
        log_step "Cleaning Existing Deployment"

        cd "$PROJECT_ROOT/deployment/docker"

        log_info "Stopping existing containers..."
        docker-compose down -v 2>/dev/null || true

        log_info "Removing old images..."
        docker images | grep wavefield-llm | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true

        log_success "Cleanup complete"
    fi
}

download_model() {
    if [[ "$SKIP_MODEL_DOWNLOAD" == true ]]; then
        log_warning "Skipping model download"
        return 0
    fi

    log_step "Downloading Model"

    local model_dir="${PROJECT_ROOT}/models/${MODEL_NAME}"

    if [[ -d "$model_dir" ]] && [[ -f "${model_dir}/config.json" ]]; then
        log_info "Model already exists at ${model_dir}"
        return 0
    fi

    mkdir -p "$model_dir"

    log_info "Downloading ${MODEL_NAME}..."
    
    # Use model loader to download
    python3 << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from model_zoo.loader import ModelLoader

loader = ModelLoader()
try:
    model_path = loader.download_model('${MODEL_NAME}')
    print(f"Model downloaded to: {model_path}")
except Exception as e:
    print(f"Error downloading model: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    if [[ $? -eq 0 ]]; then
        log_success "Model downloaded successfully"
    else
        log_error "Failed to download model"
        exit 1
    fi
}

setup_environment() {
    log_step "Setting Up Environment"

    local env_file="${PROJECT_ROOT}/deployment/docker/.env.local"

    cat > "$env_file" << EOF
# Wave Field LLM Local Environment Configuration
MODEL_NAME=${MODEL_NAME}
PORT=${PORT}
USE_GPU=${USE_GPU}
LOG_LEVEL=INFO
MAX_WORKERS=4
BATCH_SIZE=8
MAX_LENGTH=2048

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379

# PostgreSQL Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=wavefield
POSTGRES_USER=wavefield
POSTGRES_PASSWORD=wavefield_local_password

# API Configuration
API_KEY=local_development_key
ENABLE_CORS=true
CORS_ORIGINS=*

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
EOF

    log_success "Environment configured"
}

build_images() {
    log_step "Building Docker Images"

    cd "$PROJECT_ROOT/deployment/docker"

    local dockerfile="Dockerfile"
    if [[ "$USE_GPU" == true ]]; then
        dockerfile="Dockerfile.gpu"
    else
        dockerfile="Dockerfile.cpu"
    fi

    log_info "Building image using ${dockerfile}..."
    docker build -f "$dockerfile" -t wavefield-llm:local "$PROJECT_ROOT"

    log_success "Docker images built"
}

start_services() {
    log_step "Starting Services"

    cd "$PROJECT_ROOT/deployment/docker"

    # Create docker-compose override for local development
    cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  wavefield-llm:
    image: wavefield-llm:local
    ports:
      - "${PORT}:8000"
    environment:
      - MODEL_NAME=${MODEL_NAME}
      - USE_GPU=${USE_GPU}
    volumes:
      - ${PROJECT_ROOT}/models:/app/models
      - ${PROJECT_ROOT}/logs:/app/logs
EOF

    if [[ "$USE_GPU" == true ]]; then
        cat >> docker-compose.override.yml << EOF
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
    fi

    log_info "Starting services with Docker Compose..."
    docker-compose up -d

    log_success "Services started"
}

wait_for_services() {
    log_step "Waiting for Services"

    local max_attempts=60
    local attempt=0

    log_info "Waiting for API to be ready..."

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            log_success "API is ready!"
            break
        fi

        attempt=$((attempt + 1))
        sleep 2
        echo -n "."
    done

    echo ""

    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Services failed to start within timeout"
        log_info "Check logs with: docker-compose logs -f"
        exit 1
    fi
}

run_basic_tests() {
    log_step "Running Basic Tests"

    log_info "Testing health endpoint..."
    if curl -s "http://localhost:${PORT}/health" | jq . > /dev/null 2>&1; then
        log_success "Health check passed"
    else
        log_warning "Health check failed"
    fi

    log_info "Testing inference endpoint..."
    local response=$(curl -s -X POST "http://localhost:${PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Hello, world!",
            "max_tokens": 50
        }')

    if echo "$response" | jq -e '.choices[0].text' > /dev/null 2>&1; then
        log_success "Inference test passed"
        echo "Response: $(echo "$response" | jq -r '.choices[0].text' | head -c 100)..."
    else
        log_warning "Inference test failed"
    fi
}

print_summary() {
    log_step "Quick Start Complete!"

    cat << EOF
${GREEN}Wave Field LLM is now running locally!${NC}

Service Information:
  API URL:        http://localhost:${PORT}
  Model:          ${MODEL_NAME}
  GPU Enabled:    ${USE_GPU}
  Health Check:   http://localhost:${PORT}/health
  API Docs:       http://localhost:${PORT}/docs

Quick Commands:
  # View logs
  cd ${PROJECT_ROOT}/deployment/docker && docker-compose logs -f

  # Stop services
  cd ${PROJECT_ROOT}/deployment/docker && docker-compose down

  # Restart services
  cd ${PROJECT_ROOT}/deployment/docker && docker-compose restart

  # Test inference
  curl -X POST http://localhost:${PORT}/v1/completions \\
    -H "Content-Type: application/json" \\
    -d '{"prompt": "Hello, world!", "max_tokens": 50}'

Example Python Usage:
  import requests
  
  response = requests.post(
      'http://localhost:${PORT}/v1/completions',
      json={'prompt': 'Hello, world!', 'max_tokens': 50}
  )
  print(response.json()['choices'][0]['text'])

For more information, see: deployment/guides/LOCAL_DEPLOYMENT.md
EOF
}

main() {
    log_info "Starting Wave Field LLM Quick Start"

    parse_args "$@"
    check_prerequisites
    clean_existing
    download_model
    setup_environment
    build_images
    start_services
    wait_for_services
    run_basic_tests
    print_summary

    log_success "Quick start completed successfully!"
}

main "$@"

# Made with Bob
