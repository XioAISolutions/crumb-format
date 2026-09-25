#!/bin/bash
set -euo pipefail

# Wave Field LLM - Model Update Script
# Zero-downtime model updates with rollback capability

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
MODEL_NAME=""
NEW_VERSION=""
NAMESPACE="default"
STRATEGY="rolling"  # rolling, blue-green, canary
CANARY_PERCENTAGE=10
ROLLBACK_ON_ERROR=true
HEALTH_CHECK_TIMEOUT=300

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}\n"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Update Wave Field LLM model with zero downtime

Required:
    --model MODEL           Model name
    --version VERSION       New model version

Options:
    --namespace NS          Kubernetes namespace (default: default)
    --strategy STRATEGY     Update strategy: rolling, blue-green, canary (default: rolling)
    --canary-percent N      Canary percentage (default: 10)
    --no-rollback           Disable automatic rollback on error
    --timeout SECONDS       Health check timeout (default: 300)
    -h, --help              Show this help

Examples:
    # Rolling update
    $0 --model wavefield-small --version 1.1.0

    # Blue-green deployment
    $0 --model wavefield-medium --version 2.0.0 --strategy blue-green

    # Canary deployment with 20% traffic
    $0 --model wavefield-large --version 1.2.0 --strategy canary --canary-percent 20

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model) MODEL_NAME="$2"; shift 2 ;;
            --version) NEW_VERSION="$2"; shift 2 ;;
            --namespace) NAMESPACE="$2"; shift 2 ;;
            --strategy) STRATEGY="$2"; shift 2 ;;
            --canary-percent) CANARY_PERCENTAGE="$2"; shift 2 ;;
            --no-rollback) ROLLBACK_ON_ERROR=false; shift ;;
            --timeout) HEALTH_CHECK_TIMEOUT="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown option: $1"; usage ;;
        esac
    done

    if [[ -z "$MODEL_NAME" ]] || [[ -z "$NEW_VERSION" ]]; then
        log_error "Model name and version are required"
        usage
    fi
}

get_current_version() {
    log_step "Getting Current Version"

    CURRENT_VERSION=$(kubectl get deployment "wavefield-${MODEL_NAME}" \
        -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.metadata.labels.version}' 2>/dev/null || echo "unknown")

    log_info "Current version: ${CURRENT_VERSION}"
    log_info "New version: ${NEW_VERSION}"

    export CURRENT_VERSION
}

rolling_update() {
    log_step "Performing Rolling Update"

    log_info "Updating deployment image..."
    kubectl set image deployment/wavefield-${MODEL_NAME} \
        -n "$NAMESPACE" \
        wavefield-llm="wavefield-llm:${NEW_VERSION}"

    log_info "Waiting for rollout to complete..."
    if kubectl rollout status deployment/wavefield-${MODEL_NAME} \
        -n "$NAMESPACE" \
        --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        log_success "Rolling update completed successfully"
        return 0
    else
        log_error "Rolling update failed"
        if [[ "$ROLLBACK_ON_ERROR" == true ]]; then
            rollback_deployment
        fi
        return 1
    fi
}

blue_green_update() {
    log_step "Performing Blue-Green Deployment"

    local green_deployment="wavefield-${MODEL_NAME}-green"

    # Create green deployment
    log_info "Creating green deployment..."
    kubectl get deployment "wavefield-${MODEL_NAME}" -n "$NAMESPACE" -o yaml | \
        sed "s/name: wavefield-${MODEL_NAME}/name: ${green_deployment}/" | \
        sed "s/version: ${CURRENT_VERSION}/version: ${NEW_VERSION}/" | \
        sed "s/wavefield-llm:${CURRENT_VERSION}/wavefield-llm:${NEW_VERSION}/" | \
        kubectl apply -f -

    # Wait for green deployment
    log_info "Waiting for green deployment to be ready..."
    if ! kubectl rollout status deployment/${green_deployment} \
        -n "$NAMESPACE" \
        --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        log_error "Green deployment failed"
        kubectl delete deployment ${green_deployment} -n "$NAMESPACE"
        return 1
    fi

    # Run health checks on green
    log_info "Running health checks on green deployment..."
    if ! run_health_checks "${green_deployment}"; then
        log_error "Health checks failed on green deployment"
        kubectl delete deployment ${green_deployment} -n "$NAMESPACE"
        return 1
    fi

    # Switch traffic to green
    log_info "Switching traffic to green deployment..."
    kubectl patch service "wavefield-${MODEL_NAME}" -n "$NAMESPACE" -p \
        "{\"spec\":{\"selector\":{\"app\":\"wavefield-llm\",\"model\":\"${MODEL_NAME}\",\"version\":\"${NEW_VERSION}\"}}}"

    # Wait and verify
    sleep 10

    # Delete blue deployment
    log_info "Deleting blue deployment..."
    kubectl delete deployment "wavefield-${MODEL_NAME}" -n "$NAMESPACE"

    # Rename green to primary
    kubectl get deployment ${green_deployment} -n "$NAMESPACE" -o yaml | \
        sed "s/name: ${green_deployment}/name: wavefield-${MODEL_NAME}/" | \
        kubectl apply -f -
    kubectl delete deployment ${green_deployment} -n "$NAMESPACE"

    log_success "Blue-green deployment completed successfully"
}

canary_update() {
    log_step "Performing Canary Deployment"

    local canary_deployment="wavefield-${MODEL_NAME}-canary"
    local primary_replicas=$(kubectl get deployment "wavefield-${MODEL_NAME}" \
        -n "$NAMESPACE" \
        -o jsonpath='{.spec.replicas}')
    local canary_replicas=$(( primary_replicas * CANARY_PERCENTAGE / 100 ))
    [[ $canary_replicas -lt 1 ]] && canary_replicas=1

    # Create canary deployment
    log_info "Creating canary deployment with ${canary_replicas} replicas..."
    kubectl get deployment "wavefield-${MODEL_NAME}" -n "$NAMESPACE" -o yaml | \
        sed "s/name: wavefield-${MODEL_NAME}/name: ${canary_deployment}/" | \
        sed "s/replicas: ${primary_replicas}/replicas: ${canary_replicas}/" | \
        sed "s/version: ${CURRENT_VERSION}/version: ${NEW_VERSION}/" | \
        sed "s/wavefield-llm:${CURRENT_VERSION}/wavefield-llm:${NEW_VERSION}/" | \
        kubectl apply -f -

    # Wait for canary
    log_info "Waiting for canary deployment..."
    if ! kubectl rollout status deployment/${canary_deployment} \
        -n "$NAMESPACE" \
        --timeout="${HEALTH_CHECK_TIMEOUT}s"; then
        log_error "Canary deployment failed"
        kubectl delete deployment ${canary_deployment} -n "$NAMESPACE"
        return 1
    fi

    # Monitor canary
    log_info "Monitoring canary for 60 seconds..."
    sleep 60

    if ! run_health_checks "${canary_deployment}"; then
        log_error "Canary health checks failed"
        kubectl delete deployment ${canary_deployment} -n "$NAMESPACE"
        return 1
    fi

    # Promote canary
    log_info "Promoting canary to production..."
    kubectl set image deployment/wavefield-${MODEL_NAME} \
        -n "$NAMESPACE" \
        wavefield-llm="wavefield-llm:${NEW_VERSION}"

    kubectl rollout status deployment/wavefield-${MODEL_NAME} \
        -n "$NAMESPACE" \
        --timeout="${HEALTH_CHECK_TIMEOUT}s"

    # Delete canary
    kubectl delete deployment ${canary_deployment} -n "$NAMESPACE"

    log_success "Canary deployment completed successfully"
}

run_health_checks() {
    local deployment_name="$1"
    local max_attempts=10
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" \
            -l "app=wavefield-llm" \
            --field-selector=status.phase=Running \
            -o name | wc -l)

        if [[ $ready_pods -gt 0 ]]; then
            # Test inference
            local pod_name=$(kubectl get pods -n "$NAMESPACE" \
                -l "app=wavefield-llm" \
                --field-selector=status.phase=Running \
                -o jsonpath='{.items[0].metadata.name}')

            kubectl port-forward -n "$NAMESPACE" "$pod_name" 8000:8000 &
            local pf_pid=$!
            sleep 3

            local response=$(curl -s -X POST http://localhost:8000/v1/completions \
                -H "Content-Type: application/json" \
                -d '{"prompt": "test", "max_tokens": 5}' || echo "")

            kill $pf_pid 2>/dev/null || true

            if echo "$response" | grep -q "choices"; then
                return 0
            fi
        fi

        attempt=$((attempt + 1))
        sleep 5
    done

    return 1
}

rollback_deployment() {
    log_step "Rolling Back Deployment"

    log_warning "Initiating automatic rollback..."

    kubectl rollout undo deployment/wavefield-${MODEL_NAME} -n "$NAMESPACE"

    log_info "Waiting for rollback to complete..."
    kubectl rollout status deployment/wavefield-${MODEL_NAME} \
        -n "$NAMESPACE" \
        --timeout="${HEALTH_CHECK_TIMEOUT}s"

    log_success "Rollback completed"
}

verify_update() {
    log_step "Verifying Update"

    local deployed_version=$(kubectl get deployment "wavefield-${MODEL_NAME}" \
        -n "$NAMESPACE" \
        -o jsonpath='{.spec.template.metadata.labels.version}')

    if [[ "$deployed_version" == "$NEW_VERSION" ]]; then
        log_success "Update verified - running version ${NEW_VERSION}"
        return 0
    else
        log_error "Update verification failed - running version ${deployed_version}"
        return 1
    fi
}

print_summary() {
    log_step "Update Complete!"

    cat << EOF
${GREEN}Model update completed successfully!${NC}

Update Details:
  Model:               ${MODEL_NAME}
  Previous Version:    ${CURRENT_VERSION}
  New Version:         ${NEW_VERSION}
  Strategy:            ${STRATEGY}
  Namespace:           ${NAMESPACE}

Deployment Status:
$(kubectl get deployment wavefield-${MODEL_NAME} -n ${NAMESPACE})

Useful Commands:
  # View rollout history
  kubectl rollout history deployment/wavefield-${MODEL_NAME} -n ${NAMESPACE}

  # Rollback to previous version
  kubectl rollout undo deployment/wavefield-${MODEL_NAME} -n ${NAMESPACE}

  # View pods
  kubectl get pods -n ${NAMESPACE} -l model=${MODEL_NAME}

EOF
}

main() {
    log_info "Starting Model Update"
    
    parse_args "$@"
    get_current_version

    case $STRATEGY in
        rolling)
            rolling_update
            ;;
        blue-green)
            blue_green_update
            ;;
        canary)
            canary_update
            ;;
        *)
            log_error "Unknown strategy: $STRATEGY"
            exit 1
            ;;
    esac

    verify_update
    print_summary

    log_success "Model update completed!"
}

main "$@"

# Made with Bob
