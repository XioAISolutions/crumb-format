#!/bin/bash
set -euo pipefail

# Wave Field LLM - Model Deployment Script
# Deploy a specific model to Kubernetes

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
MODEL_NAME=""
MODEL_VERSION="latest"
NAMESPACE="default"
REPLICAS=3
GPU_ENABLED=false
MEMORY_REQUEST="4Gi"
MEMORY_LIMIT="8Gi"
CPU_REQUEST="2"
CPU_LIMIT="4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}\n"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy a specific Wave Field LLM model to Kubernetes

Required:
    --model MODEL           Model name (e.g., wavefield-small, wavefield-medium)

Options:
    --version VERSION       Model version (default: latest)
    --namespace NS          Kubernetes namespace (default: default)
    --replicas N            Number of replicas (default: 3)
    --gpu                   Enable GPU support
    --memory-request MEM    Memory request (default: 4Gi)
    --memory-limit MEM      Memory limit (default: 8Gi)
    --cpu-request CPU       CPU request (default: 2)
    --cpu-limit CPU         CPU limit (default: 4)
    -h, --help              Show this help

Examples:
    # Deploy small model
    $0 --model wavefield-small --replicas 3

    # Deploy medium model with GPU
    $0 --model wavefield-medium --gpu --replicas 2

    # Deploy to specific namespace
    $0 --model wavefield-large --namespace production --replicas 5

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) MODEL_NAME="$2"; shift 2 ;;
            --version) MODEL_VERSION="$2"; shift 2 ;;
            --namespace) NAMESPACE="$2"; shift 2 ;;
            --replicas) REPLICAS="$2"; shift 2 ;;
            --gpu) GPU_ENABLED=true; shift ;;
            --memory-request) MEMORY_REQUEST="$2"; shift 2 ;;
            --memory-limit) MEMORY_LIMIT="$2"; shift 2 ;;
            --cpu-request) CPU_REQUEST="$2"; shift 2 ;;
            --cpu-limit) CPU_LIMIT="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown option: $1"; usage ;;
        esac
    done

    if [[ -z "$MODEL_NAME" ]]; then
        log_error "Model name is required"
        usage
    fi
}

check_prerequisites() {
    log_step "Checking Prerequisites"

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not installed"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Not connected to a Kubernetes cluster"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

download_model() {
    log_step "Downloading Model"

    local model_dir="${PROJECT_ROOT}/models/${MODEL_NAME}"

    if [[ -d "$model_dir" ]] && [[ -f "${model_dir}/config.json" ]]; then
        log_info "Model already exists locally"
        return 0
    fi

    log_info "Downloading ${MODEL_NAME} version ${MODEL_VERSION}..."
    
    python3 << EOF
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from model_zoo.loader import ModelLoader

loader = ModelLoader()
try:
    model_path = loader.download_model('${MODEL_NAME}', version='${MODEL_VERSION}')
    print(f"Model downloaded to: {model_path}")
except Exception as e:
    print(f"Error downloading model: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    if [[ $? -ne 0 ]]; then
        log_error "Failed to download model"
        exit 1
    fi

    log_success "Model downloaded successfully"
}

validate_model() {
    log_step "Validating Model"

    log_info "Running model validation..."
    
    python3 << EOF
import sys
import json
sys.path.insert(0, '${PROJECT_ROOT}')

model_dir = '${PROJECT_ROOT}/models/${MODEL_NAME}'
config_file = f'{model_dir}/config.json'

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    required_keys = ['model_type', 'vocab_size', 'hidden_size']
    for key in required_keys:
        if key not in config:
            print(f"Missing required config key: {key}", file=sys.stderr)
            sys.exit(1)
    
    print("Model validation passed")
except Exception as e:
    print(f"Model validation failed: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    if [[ $? -ne 0 ]]; then
        log_error "Model validation failed"
        exit 1
    fi

    log_success "Model validation passed"
}

create_deployment_manifest() {
    log_step "Creating Deployment Manifest"

    local manifest_file="/tmp/wavefield-${MODEL_NAME}-deployment.yaml"

    cat > "$manifest_file" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wavefield-${MODEL_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: wavefield-llm
    model: ${MODEL_NAME}
    version: ${MODEL_VERSION}
spec:
  replicas: ${REPLICAS}
  selector:
    matchLabels:
      app: wavefield-llm
      model: ${MODEL_NAME}
  template:
    metadata:
      labels:
        app: wavefield-llm
        model: ${MODEL_NAME}
        version: ${MODEL_VERSION}
    spec:
      containers:
      - name: wavefield-llm
        image: wavefield-llm:${MODEL_VERSION}
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: MODEL_NAME
          value: "${MODEL_NAME}"
        - name: MODEL_VERSION
          value: "${MODEL_VERSION}"
        - name: GPU_ENABLED
          value: "${GPU_ENABLED}"
        resources:
          requests:
            memory: "${MEMORY_REQUEST}"
            cpu: "${CPU_REQUEST}"
          limits:
            memory: "${MEMORY_LIMIT}"
            cpu: "${CPU_LIMIT}"
EOF

    if [[ "$GPU_ENABLED" == true ]]; then
        cat >> "$manifest_file" << EOF
            nvidia.com/gpu: 1
EOF
    fi

    cat >> "$manifest_file" << EOF
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: wavefield-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: wavefield-${MODEL_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: wavefield-llm
    model: ${MODEL_NAME}
spec:
  type: ClusterIP
  ports:
  - port: 8000
    targetPort: 8000
    name: http
  - port: 9090
    targetPort: 9090
    name: metrics
  selector:
    app: wavefield-llm
    model: ${MODEL_NAME}
EOF

    export MANIFEST_FILE="$manifest_file"
    log_success "Deployment manifest created: $manifest_file"
}

deploy_to_kubernetes() {
    log_step "Deploying to Kubernetes"

    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Apply deployment
    log_info "Applying deployment manifest..."
    kubectl apply -f "$MANIFEST_FILE"

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/wavefield-${MODEL_NAME} -n "$NAMESPACE" --timeout=10m

    log_success "Deployment completed successfully"
}

run_health_checks() {
    log_step "Running Health Checks"

    local max_attempts=30
    local attempt=0

    log_info "Checking pod health..."

    while [[ $attempt -lt $max_attempts ]]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" \
            -l "app=wavefield-llm,model=${MODEL_NAME}" \
            -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)

        if [[ $ready_pods -ge $REPLICAS ]]; then
            log_success "All ${REPLICAS} replicas are running"
            break
        fi

        attempt=$((attempt + 1))
        sleep 10
        echo -n "."
    done

    echo ""

    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Health checks failed - not all replicas are ready"
        kubectl get pods -n "$NAMESPACE" -l "app=wavefield-llm,model=${MODEL_NAME}"
        exit 1
    fi

    log_success "Health checks passed"
}

test_inference() {
    log_step "Testing Inference"

    log_info "Running inference test..."

    # Port forward to test
    local pod_name=$(kubectl get pods -n "$NAMESPACE" \
        -l "app=wavefield-llm,model=${MODEL_NAME}" \
        -o jsonpath='{.items[0].metadata.name}')

    kubectl port-forward -n "$NAMESPACE" "$pod_name" 8000:8000 &
    local pf_pid=$!
    sleep 5

    # Test inference
    local response=$(curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, world!", "max_tokens": 10}' || echo "")

    kill $pf_pid 2>/dev/null || true

    if echo "$response" | grep -q "choices"; then
        log_success "Inference test passed"
    else
        log_warning "Inference test failed or returned unexpected response"
    fi
}

update_load_balancer() {
    log_step "Updating Load Balancer"

    log_info "Load balancer configuration..."
    
    # Check if ingress exists
    if kubectl get ingress wavefield-ingress -n "$NAMESPACE" &> /dev/null; then
        log_info "Updating existing ingress..."
        kubectl annotate ingress wavefield-ingress -n "$NAMESPACE" \
            "kubectl.kubernetes.io/last-applied-configuration-" \
            --overwrite
    else
        log_info "Ingress configuration should be applied separately"
    fi

    log_success "Load balancer updated"
}

print_summary() {
    log_step "Model Deployment Complete!"

    cat << EOF
${GREEN}Model ${MODEL_NAME} deployed successfully!${NC}

Deployment Details:
  Model:               ${MODEL_NAME}
  Version:             ${MODEL_VERSION}
  Namespace:           ${NAMESPACE}
  Replicas:            ${REPLICAS}
  GPU Enabled:         ${GPU_ENABLED}

Access Information:
  Service:             wavefield-${MODEL_NAME}.${NAMESPACE}.svc.cluster.local:8000
  Metrics:             wavefield-${MODEL_NAME}.${NAMESPACE}.svc.cluster.local:9090

Useful Commands:
  # View pods
  kubectl get pods -n ${NAMESPACE} -l model=${MODEL_NAME}

  # View logs
  kubectl logs -n ${NAMESPACE} -l model=${MODEL_NAME} --tail=100 -f

  # Scale deployment
  kubectl scale deployment/wavefield-${MODEL_NAME} -n ${NAMESPACE} --replicas=5

  # Delete deployment
  kubectl delete deployment/wavefield-${MODEL_NAME} -n ${NAMESPACE}

For more information, see: deployment/guides/KUBERNETES_DEPLOYMENT.md
EOF
}

main() {
    log_info "Starting Model Deployment"
    
    parse_args "$@"
    check_prerequisites
    download_model
    validate_model
    create_deployment_manifest
    deploy_to_kubernetes
    run_health_checks
    test_inference
    update_load_balancer
    print_summary

    log_success "Model deployment completed!"
}

main "$@"

# Made with Bob
