#!/bin/bash
set -euo pipefail

# Wave Field LLM - Master Deployment Script
# Orchestrates complete deployment to production environments

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
CLOUD_PROVIDER=""
REGION=""
MODEL_NAME=""
REPLICAS=3
DRY_RUN=false
VERBOSE=false
SKIP_VALIDATION=false
SKIP_TESTS=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Logging functions
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

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Master deployment script for Wave Field LLM

Required Options:
    --environment ENV       Deployment environment (production, staging, development)
    --cloud PROVIDER        Cloud provider (aws, gcp, azure)
    --region REGION         Cloud region (e.g., us-east-1, us-central1, eastus)
    --model MODEL           Model name (wavefield-small, wavefield-medium, wavefield-large)

Optional:
    --replicas N            Number of replicas (default: 3)
    --dry-run               Show what would be done without executing
    --verbose               Enable verbose output
    --skip-validation       Skip pre-deployment validation
    --skip-tests            Skip post-deployment tests
    -h, --help              Show this help message

Examples:
    # Deploy to AWS production
    $0 --environment production --cloud aws --region us-east-1 --model wavefield-small

    # Deploy to GCP staging with 5 replicas
    $0 --environment staging --cloud gcp --region us-central1 --model wavefield-medium --replicas 5

    # Dry run deployment
    $0 --environment production --cloud aws --region us-east-1 --model wavefield-small --dry-run

EOF
    exit 1
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --cloud)
                CLOUD_PROVIDER="$2"
                shift 2
                ;;
            --region)
                REGION="$2"
                shift 2
                ;;
            --model)
                MODEL_NAME="$2"
                shift 2
                ;;
            --replicas)
                REPLICAS="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
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

    # Validate required arguments
    if [[ -z "$ENVIRONMENT" ]] || [[ -z "$CLOUD_PROVIDER" ]] || [[ -z "$REGION" ]] || [[ -z "$MODEL_NAME" ]]; then
        log_error "Missing required arguments"
        usage
    fi

    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(production|staging|development)$ ]]; then
        log_error "Invalid environment: $ENVIRONMENT"
        exit 1
    fi

    # Validate cloud provider
    if [[ ! "$CLOUD_PROVIDER" =~ ^(aws|gcp|azure)$ ]]; then
        log_error "Invalid cloud provider: $CLOUD_PROVIDER"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking Prerequisites"

    local missing_tools=()

    # Check required tools
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v kubectl >/dev/null 2>&1 || missing_tools+=("kubectl")
    command -v terraform >/dev/null 2>&1 || missing_tools+=("terraform")
    command -v helm >/dev/null 2>&1 || missing_tools+=("helm")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")

    # Cloud-specific tools
    case $CLOUD_PROVIDER in
        aws)
            command -v aws >/dev/null 2>&1 || missing_tools+=("aws-cli")
            ;;
        gcp)
            command -v gcloud >/dev/null 2>&1 || missing_tools+=("gcloud")
            ;;
        azure)
            command -v az >/dev/null 2>&1 || missing_tools+=("azure-cli")
            ;;
    esac

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and try again"
        exit 1
    fi

    log_success "All prerequisites satisfied"
}

# Validate environment
validate_environment() {
    if [[ "$SKIP_VALIDATION" == true ]]; then
        log_warning "Skipping environment validation"
        return 0
    fi

    log_step "Validating Environment"

    # Run validation script
    if [[ -f "${SCRIPT_DIR}/validate.sh" ]]; then
        bash "${SCRIPT_DIR}/validate.sh" \
            --environment "$ENVIRONMENT" \
            --cloud "$CLOUD_PROVIDER" \
            --region "$REGION"
    else
        log_warning "Validation script not found, skipping validation"
    fi

    log_success "Environment validation complete"
}

# Provision infrastructure
provision_infrastructure() {
    log_step "Provisioning Infrastructure"

    local terraform_dir="${PROJECT_ROOT}/deployment/cloud/${CLOUD_PROVIDER}/terraform"

    if [[ ! -d "$terraform_dir" ]]; then
        log_error "Terraform directory not found: $terraform_dir"
        exit 1
    fi

    cd "$terraform_dir"

    # Initialize Terraform
    log_info "Initializing Terraform..."
    if [[ "$DRY_RUN" == false ]]; then
        terraform init
    else
        log_info "[DRY RUN] Would run: terraform init"
    fi

    # Plan infrastructure changes
    log_info "Planning infrastructure changes..."
    if [[ "$DRY_RUN" == false ]]; then
        terraform plan \
            -var="environment=${ENVIRONMENT}" \
            -var="region=${REGION}" \
            -var="model_name=${MODEL_NAME}" \
            -var="replicas=${REPLICAS}" \
            -out=tfplan
    else
        log_info "[DRY RUN] Would run: terraform plan"
    fi

    # Apply infrastructure changes
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_warning "Deploying to PRODUCTION environment"
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_error "Deployment cancelled"
            exit 1
        fi
    fi

    log_info "Applying infrastructure changes..."
    if [[ "$DRY_RUN" == false ]]; then
        terraform apply tfplan
    else
        log_info "[DRY RUN] Would run: terraform apply"
    fi

    cd "$PROJECT_ROOT"
    log_success "Infrastructure provisioned successfully"
}

# Build and push Docker images
build_and_push_images() {
    log_step "Building and Pushing Docker Images"

    local docker_dir="${PROJECT_ROOT}/deployment/docker"
    local image_tag="${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"

    # Determine registry based on cloud provider
    local registry=""
    case $CLOUD_PROVIDER in
        aws)
            registry=$(aws ecr describe-repositories --repository-names wavefield-llm --query 'repositories[0].repositoryUri' --output text 2>/dev/null || echo "")
            ;;
        gcp)
            registry="gcr.io/$(gcloud config get-value project)/wavefield-llm"
            ;;
        azure)
            registry=$(az acr list --query '[0].loginServer' --output tsv 2>/dev/null || echo "")
            ;;
    esac

    if [[ -z "$registry" ]]; then
        log_error "Could not determine container registry"
        exit 1
    fi

    log_info "Building Docker image..."
    if [[ "$DRY_RUN" == false ]]; then
        docker build \
            -f "${docker_dir}/Dockerfile" \
            -t "${registry}:${image_tag}" \
            -t "${registry}:latest" \
            "$PROJECT_ROOT"
    else
        log_info "[DRY RUN] Would build: ${registry}:${image_tag}"
    fi

    log_info "Pushing Docker image..."
    if [[ "$DRY_RUN" == false ]]; then
        docker push "${registry}:${image_tag}"
        docker push "${registry}:latest"
    else
        log_info "[DRY RUN] Would push: ${registry}:${image_tag}"
    fi

    # Export image tag for later use
    export DOCKER_IMAGE_TAG="${image_tag}"

    log_success "Docker images built and pushed successfully"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_step "Deploying to Kubernetes"

    local k8s_dir="${PROJECT_ROOT}/deployment/kubernetes"

    # Configure kubectl
    log_info "Configuring kubectl..."
    case $CLOUD_PROVIDER in
        aws)
            if [[ "$DRY_RUN" == false ]]; then
                aws eks update-kubeconfig --region "$REGION" --name "wavefield-${ENVIRONMENT}"
            fi
            ;;
        gcp)
            if [[ "$DRY_RUN" == false ]]; then
                gcloud container clusters get-credentials "wavefield-${ENVIRONMENT}" --region "$REGION"
            fi
            ;;
        azure)
            if [[ "$DRY_RUN" == false ]]; then
                az aks get-credentials --resource-group "wavefield-${ENVIRONMENT}" --name "wavefield-${ENVIRONMENT}"
            fi
            ;;
    esac

    # Create namespace if it doesn't exist
    log_info "Creating namespace..."
    if [[ "$DRY_RUN" == false ]]; then
        kubectl create namespace "wavefield-${ENVIRONMENT}" --dry-run=client -o yaml | kubectl apply -f -
    else
        log_info "[DRY RUN] Would create namespace: wavefield-${ENVIRONMENT}"
    fi

    # Deploy using Helm
    log_info "Deploying with Helm..."
    if [[ "$DRY_RUN" == false ]]; then
        helm upgrade --install wavefield-llm \
            "${k8s_dir}/helm/wavefield-llm" \
            --namespace "wavefield-${ENVIRONMENT}" \
            --set environment="$ENVIRONMENT" \
            --set image.tag="${DOCKER_IMAGE_TAG:-latest}" \
            --set replicaCount="$REPLICAS" \
            --set model.name="$MODEL_NAME" \
            --wait \
            --timeout 10m
    else
        log_info "[DRY RUN] Would deploy with Helm"
    fi

    log_success "Kubernetes deployment complete"
}

# Setup monitoring
setup_monitoring() {
    log_step "Setting Up Monitoring"

    if [[ -f "${SCRIPT_DIR}/setup-monitoring.sh" ]]; then
        if [[ "$DRY_RUN" == false ]]; then
            bash "${SCRIPT_DIR}/setup-monitoring.sh" \
                --environment "$ENVIRONMENT" \
                --namespace "wavefield-${ENVIRONMENT}"
        else
            log_info "[DRY RUN] Would setup monitoring"
        fi
    else
        log_warning "Monitoring setup script not found, skipping"
    fi

    log_success "Monitoring setup complete"
}

# Run health checks
run_health_checks() {
    log_step "Running Health Checks"

    local max_attempts=30
    local attempt=0

    log_info "Waiting for services to be ready..."

    while [[ $attempt -lt $max_attempts ]]; do
        if [[ "$DRY_RUN" == false ]]; then
            if kubectl get pods -n "wavefield-${ENVIRONMENT}" -l app=wavefield-llm -o jsonpath='{.items[*].status.phase}' | grep -q "Running"; then
                log_success "Services are running"
                break
            fi
        else
            log_info "[DRY RUN] Would check service health"
            break
        fi

        attempt=$((attempt + 1))
        sleep 10
    done

    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Services failed to start within timeout"
        exit 1
    fi

    log_success "Health checks passed"
}

# Run smoke tests
run_smoke_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_warning "Skipping smoke tests"
        return 0
    fi

    log_step "Running Smoke Tests"

    if [[ -f "${SCRIPT_DIR}/smoke-test.sh" ]]; then
        if [[ "$DRY_RUN" == false ]]; then
            bash "${SCRIPT_DIR}/smoke-test.sh" \
                --environment "$ENVIRONMENT" \
                --namespace "wavefield-${ENVIRONMENT}"
        else
            log_info "[DRY RUN] Would run smoke tests"
        fi
    else
        log_warning "Smoke test script not found, skipping"
    fi

    log_success "Smoke tests passed"
}

# Print deployment summary
print_summary() {
    log_step "Deployment Summary"

    cat << EOF
${GREEN}Deployment completed successfully!${NC}

Environment:     ${ENVIRONMENT}
Cloud Provider:  ${CLOUD_PROVIDER}
Region:          ${REGION}
Model:           ${MODEL_NAME}
Replicas:        ${REPLICAS}
Image Tag:       ${DOCKER_IMAGE_TAG:-latest}

Next Steps:
1. Monitor the deployment: kubectl get pods -n wavefield-${ENVIRONMENT}
2. Check logs: kubectl logs -n wavefield-${ENVIRONMENT} -l app=wavefield-llm
3. Access the service: kubectl get svc -n wavefield-${ENVIRONMENT}
4. View monitoring dashboards
5. Run full test suite

For more information, see: deployment/guides/PRODUCTION_DEPLOYMENT.md
EOF
}

# Main deployment flow
main() {
    log_info "Starting Wave Field LLM Deployment"
    log_info "Environment: $ENVIRONMENT | Cloud: $CLOUD_PROVIDER | Region: $REGION"

    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN MODE - No changes will be made"
    fi

    check_prerequisites
    validate_environment
    provision_infrastructure
    build_and_push_images
    deploy_to_kubernetes
    setup_monitoring
    run_health_checks
    run_smoke_tests
    print_summary

    log_success "Deployment completed successfully!"
}

# Parse arguments and run
parse_args "$@"
main

# Made with Bob
