#!/bin/bash
# Wave Field LLM Deployment Script
# Deploys the application to specified environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Default values
ENVIRONMENT="${1:-staging}"
VERSION="${2:-latest}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
validate_environment() {
    log_info "Validating environment: ${ENVIRONMENT}"
    
    case "${ENVIRONMENT}" in
        development|dev)
            ENVIRONMENT="development"
            NAMESPACE="wavefield-llm-dev"
            REPLICAS=2
            ;;
        staging|stage)
            ENVIRONMENT="staging"
            NAMESPACE="wavefield-llm-staging"
            REPLICAS=3
            ;;
        production|prod)
            ENVIRONMENT="production"
            NAMESPACE="wavefield-llm-prod"
            REPLICAS=5
            ;;
        *)
            log_error "Invalid environment: ${ENVIRONMENT}"
            log_error "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
    
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Namespace: ${NAMESPACE}"
    log_info "Replicas: ${REPLICAS}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in kubectl helm aws docker; do
        if ! command -v "${tool}" &> /dev/null; then
            missing_tools+=("${tool}")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Helm
    if ! helm version &> /dev/null; then
        log_error "Helm is not properly configured"
        exit 1
    fi
    
    log_info "All prerequisites met"
}

# Configure AWS and Kubernetes
configure_aws() {
    log_info "Configuring AWS credentials..."
    
    local cluster_name="wavefield-llm-${ENVIRONMENT}"
    local region="${AWS_REGION:-us-east-1}"
    
    aws eks update-kubeconfig \
        --name "${cluster_name}" \
        --region "${region}"
    
    log_info "AWS configuration complete"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Ensuring namespace exists: ${NAMESPACE}"
    
    if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        kubectl create namespace "${NAMESPACE}"
        log_info "Created namespace: ${NAMESPACE}"
    else
        log_info "Namespace already exists: ${NAMESPACE}"
    fi
}

# Create backup before deployment
create_backup() {
    if [ "${ENVIRONMENT}" = "production" ]; then
        log_info "Creating backup before deployment..."
        
        local backup_script="${SCRIPT_DIR}/backup-before-deploy.sh"
        if [ -f "${backup_script}" ]; then
            bash "${backup_script}" "${ENVIRONMENT}"
        else
            log_warn "Backup script not found, skipping backup"
        fi
    fi
}

# Deploy with Helm
deploy_helm() {
    log_info "Deploying with Helm..."
    
    local release_name="wavefield-llm"
    local chart_path="${DEPLOYMENT_DIR}/kubernetes/helm/wavefield-llm"
    local values_file="${chart_path}/values-${ENVIRONMENT}.yaml"
    
    # Build Helm command
    local helm_cmd=(
        helm upgrade --install "${release_name}"
        "${chart_path}"
        --namespace "${NAMESPACE}"
        --create-namespace
        --set "image.tag=${VERSION}"
        --set "environment=${ENVIRONMENT}"
        --set "replicaCount=${REPLICAS}"
        --timeout 20m
        --wait
    )
    
    # Add environment-specific values file if it exists
    if [ -f "${values_file}" ]; then
        helm_cmd+=(--values "${values_file}")
    fi
    
    # Add production-specific settings
    if [ "${ENVIRONMENT}" = "production" ]; then
        helm_cmd+=(
            --set "autoscaling.enabled=true"
            --set "autoscaling.minReplicas=5"
            --set "autoscaling.maxReplicas=20"
            --set "resources.limits.memory=16Gi"
            --set "resources.limits.cpu=8"
        )
    fi
    
    # Dry run if requested
    if [ "${DRY_RUN}" = "true" ]; then
        helm_cmd+=(--dry-run --debug)
        log_info "Running in dry-run mode"
    fi
    
    # Execute deployment
    log_info "Executing: ${helm_cmd[*]}"
    "${helm_cmd[@]}"
    
    log_info "Helm deployment complete"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    local max_wait=600  # 10 minutes
    local elapsed=0
    local interval=10
    
    while [ ${elapsed} -lt ${max_wait} ]; do
        local ready_replicas=$(kubectl get deployment wavefield-llm \
            -n "${NAMESPACE}" \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        
        if [ "${ready_replicas}" -ge "${REPLICAS}" ]; then
            log_info "Deployment is ready (${ready_replicas}/${REPLICAS} replicas)"
            return 0
        fi
        
        log_info "Waiting for deployment... (${ready_replicas}/${REPLICAS} replicas ready)"
        sleep ${interval}
        elapsed=$((elapsed + interval))
    done
    
    log_error "Deployment did not become ready within ${max_wait} seconds"
    return 1
}

# Run smoke tests
run_smoke_tests() {
    if [ "${SKIP_TESTS}" = "true" ]; then
        log_warn "Skipping smoke tests"
        return 0
    fi
    
    log_info "Running smoke tests..."
    
    local smoke_test_script="${SCRIPT_DIR}/smoke-test.sh"
    if [ -f "${smoke_test_script}" ]; then
        local service_url
        
        case "${ENVIRONMENT}" in
            development)
                service_url="dev.wavefield-llm.example.com"
                ;;
            staging)
                service_url="staging.wavefield-llm.example.com"
                ;;
            production)
                service_url="api.wavefield-llm.example.com"
                ;;
        esac
        
        bash "${smoke_test_script}" "${service_url}"
    else
        log_warn "Smoke test script not found, skipping tests"
    fi
}

# Rollback on failure
rollback_deployment() {
    if [ "${ROLLBACK_ON_FAILURE}" = "true" ]; then
        log_error "Deployment failed, initiating rollback..."
        
        helm rollback wavefield-llm -n "${NAMESPACE}"
        
        log_info "Rollback complete"
    else
        log_error "Deployment failed, but rollback is disabled"
    fi
}

# Record deployment
record_deployment() {
    log_info "Recording deployment..."
    
    local deployment_record="${PROJECT_ROOT}/deployments.log"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    echo "${timestamp}|${ENVIRONMENT}|${VERSION}|${USER:-unknown}|success" >> "${deployment_record}"
    
    log_info "Deployment recorded"
}

# Send notification
send_notification() {
    local status="$1"
    local message="$2"
    
    log_info "Sending notification: ${message}"
    
    # Slack notification (if webhook is configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST "${SLACK_WEBHOOK_URL}" \
            -H 'Content-Type: application/json' \
            -d "{
                \"text\": \"${message}\",
                \"username\": \"Deployment Bot\",
                \"icon_emoji\": \":rocket:\"
            }" || log_warn "Failed to send Slack notification"
    fi
    
    # Email notification (if configured)
    if [ -n "${NOTIFICATION_EMAIL:-}" ]; then
        echo "${message}" | mail -s "Deployment ${status}: ${ENVIRONMENT}" "${NOTIFICATION_EMAIL}" || \
            log_warn "Failed to send email notification"
    fi
}

# Main deployment flow
main() {
    log_info "Starting deployment to ${ENVIRONMENT}"
    log_info "Version: ${VERSION}"
    
    # Validate and prepare
    validate_environment
    check_prerequisites
    configure_aws
    create_namespace
    
    # Create backup for production
    create_backup
    
    # Deploy
    if deploy_helm && wait_for_deployment; then
        log_info "Deployment successful"
        
        # Run tests
        if run_smoke_tests; then
            log_info "Smoke tests passed"
            record_deployment
            send_notification "SUCCESS" "✅ Deployment to ${ENVIRONMENT} completed successfully (version: ${VERSION})"
            exit 0
        else
            log_error "Smoke tests failed"
            rollback_deployment
            send_notification "FAILED" "❌ Deployment to ${ENVIRONMENT} failed smoke tests (version: ${VERSION})"
            exit 1
        fi
    else
        log_error "Deployment failed"
        rollback_deployment
        send_notification "FAILED" "❌ Deployment to ${ENVIRONMENT} failed (version: ${VERSION})"
        exit 1
    fi
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [ENVIRONMENT] [VERSION]

Deploy Wave Field LLM to specified environment

Arguments:
    ENVIRONMENT    Target environment (development, staging, production)
    VERSION        Version to deploy (default: latest)

Environment Variables:
    DRY_RUN                 Set to 'true' for dry-run mode
    SKIP_TESTS              Set to 'true' to skip smoke tests
    ROLLBACK_ON_FAILURE     Set to 'false' to disable automatic rollback
    AWS_REGION              AWS region (default: us-east-1)
    SLACK_WEBHOOK_URL       Slack webhook for notifications
    NOTIFICATION_EMAIL      Email address for notifications

Examples:
    $0 staging v1.2.3
    DRY_RUN=true $0 production v1.2.3
    SKIP_TESTS=true $0 development latest

EOF
}

# Handle arguments
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

# Run main function
main "$@"

# Made with Bob
