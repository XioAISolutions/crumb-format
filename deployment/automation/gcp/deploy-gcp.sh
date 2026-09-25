#!/bin/bash
set -euo pipefail

# Wave Field LLM - GCP Deployment Script
# Complete GCP infrastructure and application deployment

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT="production"
REGION="us-central1"
ZONE="us-central1-a"
PROJECT_ID=""
CLUSTER_NAME=""
MACHINE_TYPE="n1-standard-4"
NODE_COUNT=3
DB_TIER="db-n1-standard-2"
REDIS_TIER="STANDARD_HA"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}\n"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Wave Field LLM to Google Cloud Platform

Required:
    --project-id ID         GCP project ID

Options:
    --environment ENV       Environment (production, staging, development)
    --region REGION         GCP region (default: us-central1)
    --zone ZONE             GCP zone (default: us-central1-a)
    --cluster-name NAME     GKE cluster name (default: wavefield-ENV)
    --machine-type TYPE     GKE machine type (default: n1-standard-4)
    --node-count N          Number of nodes (default: 3)
    --db-tier TIER          Cloud SQL tier (default: db-n1-standard-2)
    --redis-tier TIER       Memorystore tier (default: STANDARD_HA)
    -h, --help              Show this help

Examples:
    # Production deployment
    $0 --project-id my-project --environment production

    # Staging with smaller instances
    $0 --project-id my-project --environment staging \\
       --machine-type n1-standard-2 --db-tier db-f1-micro

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project-id) PROJECT_ID="$2"; shift 2 ;;
            --environment) ENVIRONMENT="$2"; shift 2 ;;
            --region) REGION="$2"; shift 2 ;;
            --zone) ZONE="$2"; shift 2 ;;
            --cluster-name) CLUSTER_NAME="$2"; shift 2 ;;
            --machine-type) MACHINE_TYPE="$2"; shift 2 ;;
            --node-count) NODE_COUNT="$2"; shift 2 ;;
            --db-tier) DB_TIER="$2"; shift 2 ;;
            --redis-tier) REDIS_TIER="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown option: $1"; usage ;;
        esac
    done

    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required"
        usage
    fi

    [[ -z "$CLUSTER_NAME" ]] && CLUSTER_NAME="wavefield-${ENVIRONMENT}"
}

check_gcloud() {
    log_step "Checking Google Cloud SDK"

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI not installed"
        exit 1
    fi

    gcloud config set project "$PROJECT_ID"
    gcloud config set compute/region "$REGION"
    gcloud config set compute/zone "$ZONE"

    log_success "Google Cloud SDK configured"
}

enable_apis() {
    log_step "Enabling Required APIs"

    local apis=(
        "compute.googleapis.com"
        "container.googleapis.com"
        "sqladmin.googleapis.com"
        "redis.googleapis.com"
        "storage-api.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "servicenetworking.googleapis.com"
        "artifactregistry.googleapis.com"
    )

    for api in "${apis[@]}"; do
        log_info "Enabling ${api}..."
        gcloud services enable "$api" --project="$PROJECT_ID"
    done

    log_success "APIs enabled"
}

create_vpc() {
    log_step "Creating VPC Network"

    local network_name="wavefield-${ENVIRONMENT}"

    if gcloud compute networks describe "$network_name" &> /dev/null; then
        log_info "VPC network already exists"
        return 0
    fi

    log_info "Creating VPC network..."
    gcloud compute networks create "$network_name" \
        --subnet-mode=custom \
        --bgp-routing-mode=regional

    # Create subnet
    log_info "Creating subnet..."
    gcloud compute networks subnets create "${network_name}-subnet" \
        --network="$network_name" \
        --region="$REGION" \
        --range="10.0.0.0/20" \
        --enable-private-ip-google-access

    # Create firewall rules
    log_info "Creating firewall rules..."
    gcloud compute firewall-rules create "${network_name}-allow-internal" \
        --network="$network_name" \
        --allow=tcp,udp,icmp \
        --source-ranges=10.0.0.0/20

    log_success "VPC network created"
}

create_cloud_sql() {
    log_step "Creating Cloud SQL Instance"

    local instance_name="wavefield-${ENVIRONMENT}"

    if gcloud sql instances describe "$instance_name" &> /dev/null; then
        log_info "Cloud SQL instance already exists"
        return 0
    fi

    log_info "Creating Cloud SQL instance (this may take 10-15 minutes)..."
    gcloud sql instances create "$instance_name" \
        --database-version=POSTGRES_15 \
        --tier="$DB_TIER" \
        --region="$REGION" \
        --network="projects/${PROJECT_ID}/global/networks/wavefield-${ENVIRONMENT}" \
        --no-assign-ip \
        --storage-type=SSD \
        --storage-size=100GB \
        --storage-auto-increase \
        --backup-start-time=03:00 \
        --maintenance-window-day=SUN \
        --maintenance-window-hour=4 \
        --enable-bin-log \
        --database-flags=max_connections=200

    # Create database
    log_info "Creating database..."
    gcloud sql databases create wavefield \
        --instance="$instance_name"

    # Create user
    log_info "Creating database user..."
    local db_password=$(openssl rand -base64 32)
    gcloud sql users create wavefield \
        --instance="$instance_name" \
        --password="$db_password"

    # Store password in Secret Manager
    echo -n "$db_password" | gcloud secrets create "wavefield-${ENVIRONMENT}-db-password" \
        --data-file=- \
        --replication-policy=automatic 2>/dev/null || true

    log_success "Cloud SQL instance created"
}

create_memorystore() {
    log_step "Creating Memorystore Redis"

    local instance_name="wavefield-${ENVIRONMENT}"

    if gcloud redis instances describe "$instance_name" --region="$REGION" &> /dev/null; then
        log_info "Memorystore instance already exists"
        return 0
    fi

    log_info "Creating Memorystore Redis instance..."
    gcloud redis instances create "$instance_name" \
        --size=5 \
        --region="$REGION" \
        --tier="$REDIS_TIER" \
        --redis-version=redis_7_0 \
        --network="projects/${PROJECT_ID}/global/networks/wavefield-${ENVIRONMENT}"

    log_success "Memorystore Redis created"
}

create_storage_buckets() {
    log_step "Creating Cloud Storage Buckets"

    local model_bucket="wavefield-${ENVIRONMENT}-models-${PROJECT_ID}"
    local logs_bucket="wavefield-${ENVIRONMENT}-logs-${PROJECT_ID}"

    # Create model bucket
    log_info "Creating model bucket..."
    gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://${model_bucket}/" 2>/dev/null || true
    gsutil versioning set on "gs://${model_bucket}/"
    gsutil encryption set -k "gs://${model_bucket}/"

    # Create logs bucket
    log_info "Creating logs bucket..."
    gsutil mb -p "$PROJECT_ID" -c STANDARD -l "$REGION" "gs://${logs_bucket}/" 2>/dev/null || true
    gsutil lifecycle set - "gs://${logs_bucket}/" << EOF
{
  "lifecycle": {
    "rule": [{
      "action": {"type": "Delete"},
      "condition": {"age": 90}
    }]
  }
}
EOF

    export MODEL_BUCKET="$model_bucket"
    export LOGS_BUCKET="$logs_bucket"

    log_success "Storage buckets created"
}

create_artifact_registry() {
    log_step "Creating Artifact Registry"

    local repo_name="wavefield-llm"

    if gcloud artifacts repositories describe "$repo_name" \
        --location="$REGION" &> /dev/null; then
        log_info "Artifact Registry already exists"
        return 0
    fi

    log_info "Creating Artifact Registry repository..."
    gcloud artifacts repositories create "$repo_name" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Wave Field LLM container images"

    log_success "Artifact Registry created"
}

create_gke_cluster() {
    log_step "Creating GKE Cluster"

    if gcloud container clusters describe "$CLUSTER_NAME" \
        --region="$REGION" &> /dev/null; then
        log_info "GKE cluster already exists"
        return 0
    fi

    log_info "Creating GKE cluster (this may take 10-15 minutes)..."
    gcloud container clusters create "$CLUSTER_NAME" \
        --region="$REGION" \
        --machine-type="$MACHINE_TYPE" \
        --num-nodes="$NODE_COUNT" \
        --disk-size=100GB \
        --disk-type=pd-standard \
        --enable-autoscaling \
        --min-nodes=2 \
        --max-nodes=10 \
        --enable-autorepair \
        --enable-autoupgrade \
        --network="wavefield-${ENVIRONMENT}" \
        --subnetwork="${ENVIRONMENT}-subnet" \
        --enable-ip-alias \
        --enable-stackdriver-kubernetes \
        --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver \
        --workload-pool="${PROJECT_ID}.svc.id.goog" \
        --enable-shielded-nodes \
        --shielded-secure-boot \
        --shielded-integrity-monitoring

    # Get credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" --region="$REGION"

    log_success "GKE cluster created"
}

setup_workload_identity() {
    log_step "Setting Up Workload Identity"

    local ksa_name="wavefield-llm"
    local gsa_name="wavefield-${ENVIRONMENT}@${PROJECT_ID}.iam.gserviceaccount.com"

    # Create Google Service Account
    log_info "Creating Google Service Account..."
    gcloud iam service-accounts create "wavefield-${ENVIRONMENT}" \
        --display-name="Wave Field LLM Service Account" 2>/dev/null || true

    # Grant permissions
    log_info "Granting permissions..."
    for role in roles/storage.objectViewer roles/cloudsql.client roles/secretmanager.secretAccessor; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:${gsa_name}" \
            --role="$role" 2>/dev/null || true
    done

    # Create Kubernetes Service Account
    kubectl create namespace "wavefield-${ENVIRONMENT}" --dry-run=client -o yaml | kubectl apply -f -
    kubectl create serviceaccount "$ksa_name" \
        -n "wavefield-${ENVIRONMENT}" --dry-run=client -o yaml | kubectl apply -f -

    # Bind accounts
    gcloud iam service-accounts add-iam-policy-binding "$gsa_name" \
        --role=roles/iam.workloadIdentityUser \
        --member="serviceAccount:${PROJECT_ID}.svc.id.goog[wavefield-${ENVIRONMENT}/${ksa_name}]"

    kubectl annotate serviceaccount "$ksa_name" \
        -n "wavefield-${ENVIRONMENT}" \
        iam.gke.io/gcp-service-account="$gsa_name" \
        --overwrite

    log_success "Workload Identity configured"
}

setup_cloud_cdn() {
    log_step "Setting Up Cloud CDN"

    log_info "Cloud CDN setup requires manual configuration"
    log_info "See: deployment/guides/CLOUD_DEPLOYMENT.md"

    log_success "Cloud CDN setup instructions provided"
}

print_summary() {
    log_step "GCP Deployment Complete!"

    cat << EOF
${GREEN}Wave Field LLM GCP infrastructure deployed successfully!${NC}

Resources Created:
  Project ID:          ${PROJECT_ID}
  GKE Cluster:         ${CLUSTER_NAME}
  Region:              ${REGION}
  Model Bucket:        gs://${MODEL_BUCKET}
  Logs Bucket:         gs://${LOGS_BUCKET}

Next Steps:
1. Deploy application:
   cd ${PROJECT_ROOT}
   ./deployment/automation/deploy.sh \\
     --environment ${ENVIRONMENT} \\
     --cloud gcp \\
     --region ${REGION} \\
     --model wavefield-small

2. Configure Cloud DNS and SSL certificates
3. Setup monitoring with Cloud Monitoring
4. Run security audit

For more information, see: deployment/guides/CLOUD_DEPLOYMENT.md
EOF
}

main() {
    log_info "Starting GCP Deployment"
    
    parse_args "$@"
    check_gcloud
    enable_apis
    create_vpc
    create_cloud_sql
    create_memorystore
    create_storage_buckets
    create_artifact_registry
    create_gke_cluster
    setup_workload_identity
    setup_cloud_cdn
    print_summary

    log_success "GCP deployment completed!"
}

main "$@"

# Made with Bob
