#!/bin/bash
set -euo pipefail

# Wave Field LLM - Azure Deployment Script
# Complete Azure infrastructure and application deployment

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT="production"
LOCATION="eastus"
RESOURCE_GROUP=""
CLUSTER_NAME=""
VM_SIZE="Standard_D4s_v3"
NODE_COUNT=3
DB_SKU="GP_Gen5_2"
REDIS_SKU="Standard"

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

Deploy Wave Field LLM to Microsoft Azure

Options:
    --environment ENV       Environment (production, staging, development)
    --location LOCATION     Azure location (default: eastus)
    --resource-group RG     Resource group name (default: wavefield-ENV)
    --cluster-name NAME     AKS cluster name (default: wavefield-ENV)
    --vm-size SIZE          AKS VM size (default: Standard_D4s_v3)
    --node-count N          Number of nodes (default: 3)
    --db-sku SKU            Database SKU (default: GP_Gen5_2)
    --redis-sku SKU         Redis SKU (default: Standard)
    -h, --help              Show this help

Examples:
    # Production deployment
    $0 --environment production --location eastus

    # Staging with smaller instances
    $0 --environment staging --vm-size Standard_D2s_v3

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment) ENVIRONMENT="$2"; shift 2 ;;
            --location) LOCATION="$2"; shift 2 ;;
            --resource-group) RESOURCE_GROUP="$2"; shift 2 ;;
            --cluster-name) CLUSTER_NAME="$2"; shift 2 ;;
            --vm-size) VM_SIZE="$2"; shift 2 ;;
            --node-count) NODE_COUNT="$2"; shift 2 ;;
            --db-sku) DB_SKU="$2"; shift 2 ;;
            --redis-sku) REDIS_SKU="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown option: $1"; usage ;;
        esac
    done

    [[ -z "$RESOURCE_GROUP" ]] && RESOURCE_GROUP="wavefield-${ENVIRONMENT}"
    [[ -z "$CLUSTER_NAME" ]] && CLUSTER_NAME="wavefield-${ENVIRONMENT}"
}

check_azure_cli() {
    log_step "Checking Azure CLI"

    if ! command -v az &> /dev/null; then
        log_error "Azure CLI not installed"
        exit 1
    fi

    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Run: az login"
        exit 1
    fi

    log_success "Azure CLI configured"
}

create_resource_group() {
    log_step "Creating Resource Group"

    if az group show --name "$RESOURCE_GROUP" &> /dev/null; then
        log_info "Resource group already exists"
        return 0
    fi

    log_info "Creating resource group..."
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --tags "Environment=${ENVIRONMENT}" "Application=wavefield-llm"

    log_success "Resource group created"
}

create_vnet() {
    log_step "Creating Virtual Network"

    local vnet_name="wavefield-${ENVIRONMENT}-vnet"

    if az network vnet show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$vnet_name" &> /dev/null; then
        log_info "Virtual network already exists"
        return 0
    fi

    log_info "Creating virtual network..."
    az network vnet create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$vnet_name" \
        --address-prefix 10.0.0.0/16 \
        --subnet-name "wavefield-subnet" \
        --subnet-prefix 10.0.0.0/20

    log_success "Virtual network created"
}

create_postgresql() {
    log_step "Creating Azure Database for PostgreSQL"

    local server_name="wavefield-${ENVIRONMENT}-$(date +%s)"

    if az postgres server show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$server_name" &> /dev/null 2>&1; then
        log_info "PostgreSQL server already exists"
        return 0
    fi

    log_info "Creating PostgreSQL server (this may take 10-15 minutes)..."
    local admin_password=$(openssl rand -base64 32)
    
    az postgres server create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$server_name" \
        --location "$LOCATION" \
        --admin-user wavefield \
        --admin-password "$admin_password" \
        --sku-name "$DB_SKU" \
        --storage-size 102400 \
        --backup-retention 7 \
        --geo-redundant-backup Enabled \
        --ssl-enforcement Enabled \
        --version 11

    # Create database
    log_info "Creating database..."
    az postgres db create \
        --resource-group "$RESOURCE_GROUP" \
        --server-name "$server_name" \
        --name wavefield

    # Store password in Key Vault
    local vault_name="wavefield-${ENVIRONMENT}-kv"
    az keyvault secret set \
        --vault-name "$vault_name" \
        --name "db-password" \
        --value "$admin_password" 2>/dev/null || true

    log_success "PostgreSQL server created"
}

create_redis() {
    log_step "Creating Azure Cache for Redis"

    local redis_name="wavefield-${ENVIRONMENT}-redis"

    if az redis show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$redis_name" &> /dev/null; then
        log_info "Redis cache already exists"
        return 0
    fi

    log_info "Creating Redis cache..."
    az redis create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$redis_name" \
        --location "$LOCATION" \
        --sku "$REDIS_SKU" \
        --vm-size C1 \
        --enable-non-ssl-port false

    log_success "Redis cache created"
}

create_storage() {
    log_step "Creating Storage Account"

    local storage_name="wavefield${ENVIRONMENT}$(date +%s | tail -c 6)"

    if az storage account show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$storage_name" &> /dev/null; then
        log_info "Storage account already exists"
        return 0
    fi

    log_info "Creating storage account..."
    az storage account create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$storage_name" \
        --location "$LOCATION" \
        --sku Standard_LRS \
        --kind StorageV2 \
        --https-only true \
        --min-tls-version TLS1_2

    # Create containers
    log_info "Creating blob containers..."
    local account_key=$(az storage account keys list \
        --resource-group "$RESOURCE_GROUP" \
        --account-name "$storage_name" \
        --query '[0].value' -o tsv)

    az storage container create \
        --name models \
        --account-name "$storage_name" \
        --account-key "$account_key"

    az storage container create \
        --name logs \
        --account-name "$storage_name" \
        --account-key "$account_key"

    export STORAGE_ACCOUNT="$storage_name"

    log_success "Storage account created"
}

create_container_registry() {
    log_step "Creating Container Registry"

    local acr_name="wavefield${ENVIRONMENT}acr"

    if az acr show --name "$acr_name" &> /dev/null; then
        log_info "Container registry already exists"
        return 0
    fi

    log_info "Creating container registry..."
    az acr create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$acr_name" \
        --sku Standard \
        --admin-enabled true

    log_success "Container registry created"
}

create_aks_cluster() {
    log_step "Creating AKS Cluster"

    if az aks show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" &> /dev/null; then
        log_info "AKS cluster already exists"
        return 0
    fi

    log_info "Creating AKS cluster (this may take 10-15 minutes)..."
    az aks create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --location "$LOCATION" \
        --node-count "$NODE_COUNT" \
        --node-vm-size "$VM_SIZE" \
        --enable-cluster-autoscaler \
        --min-count 2 \
        --max-count 10 \
        --network-plugin azure \
        --enable-managed-identity \
        --enable-addons monitoring \
        --generate-ssh-keys

    # Get credentials
    az aks get-credentials \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CLUSTER_NAME" \
        --overwrite-existing

    log_success "AKS cluster created"
}

setup_key_vault() {
    log_step "Setting Up Key Vault"

    local vault_name="wavefield-${ENVIRONMENT}-kv"

    if az keyvault show --name "$vault_name" &> /dev/null; then
        log_info "Key Vault already exists"
        return 0
    fi

    log_info "Creating Key Vault..."
    az keyvault create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$vault_name" \
        --location "$LOCATION" \
        --enabled-for-deployment true \
        --enabled-for-template-deployment true

    log_success "Key Vault created"
}

setup_azure_cdn() {
    log_step "Setting Up Azure CDN"

    log_info "Azure CDN setup requires manual configuration"
    log_info "See: deployment/guides/CLOUD_DEPLOYMENT.md"

    log_success "Azure CDN setup instructions provided"
}

print_summary() {
    log_step "Azure Deployment Complete!"

    cat << EOF
${GREEN}Wave Field LLM Azure infrastructure deployed successfully!${NC}

Resources Created:
  Resource Group:      ${RESOURCE_GROUP}
  AKS Cluster:         ${CLUSTER_NAME}
  Location:            ${LOCATION}
  Storage Account:     ${STORAGE_ACCOUNT}

Next Steps:
1. Deploy application:
   cd ${PROJECT_ROOT}
   ./deployment/automation/deploy.sh \\
     --environment ${ENVIRONMENT} \\
     --cloud azure \\
     --region ${LOCATION} \\
     --model wavefield-small

2. Configure Azure DNS and SSL certificates
3. Setup monitoring with Azure Monitor
4. Run security audit

For more information, see: deployment/guides/CLOUD_DEPLOYMENT.md
EOF
}

main() {
    log_info "Starting Azure Deployment"
    
    parse_args "$@"
    check_azure_cli
    create_resource_group
    create_vnet
    create_postgresql
    create_redis
    create_storage
    create_container_registry
    create_aks_cluster
    setup_key_vault
    setup_azure_cdn
    print_summary

    log_success "Azure deployment completed!"
}

main "$@"

# Made with Bob
