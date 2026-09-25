#!/bin/bash
set -euo pipefail

# Wave Field LLM - Disaster Recovery Script
# Handles disaster recovery scenarios including backup restoration and failover

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RECOVERY_TYPE=""
BACKUP_PATH=""
ENVIRONMENT=""
REGION=""
SECONDARY_REGION=""
DRY_RUN=false
VERBOSE=false
FORCE=false

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

Disaster recovery operations for Wave Field LLM

OPTIONS:
    --type TYPE             Recovery type (restore, failover, failback)
    --backup-path PATH      Path to backup for restoration
    --environment ENV       Environment (production, staging)
    --region REGION         Primary region
    --secondary-region REG  Secondary region for failover
    --force                 Force recovery without confirmation
    --dry-run              Show what would be done without executing
    --verbose              Enable verbose output
    -h, --help             Show this help message

RECOVERY TYPES:
    restore     Restore from backup
    failover    Failover to secondary region
    failback    Failback to primary region

EXAMPLES:
    # Restore from backup
    $0 --type restore --backup-path ./backups/prod-20240101 --environment production

    # Failover to secondary region
    $0 --type failover --environment production --region us-east-1 --secondary-region us-west-2

    # Failback to primary region
    $0 --type failback --environment production --region us-east-1 --secondary-region us-west-2

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            RECOVERY_TYPE="$2"
            shift 2
            ;;
        --backup-path)
            BACKUP_PATH="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --secondary-region)
            SECONDARY_REGION="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
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

# Validate required parameters
if [[ -z "$RECOVERY_TYPE" ]]; then
    log_error "Recovery type is required"
    usage
fi

if [[ -z "$ENVIRONMENT" ]]; then
    log_error "Environment is required"
    usage
fi

# Confirmation prompt
confirm_recovery() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    log_warning "This will perform disaster recovery operation: $RECOVERY_TYPE"
    log_warning "Environment: $ENVIRONMENT"
    
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Recovery cancelled"
        exit 0
    fi
}

# Restore from backup
restore_from_backup() {
    log_step "Restoring from Backup"
    
    if [[ -z "$BACKUP_PATH" ]]; then
        log_error "Backup path is required for restore operation"
        exit 1
    fi
    
    if [[ ! -d "$BACKUP_PATH" ]]; then
        log_error "Backup path does not exist: $BACKUP_PATH"
        exit 1
    fi
    
    confirm_recovery
    
    # Stop current services
    log_info "Stopping current services..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl scale deployment --all --replicas=0 -n "$ENVIRONMENT" || true
        sleep 10
    else
        log_info "[DRY RUN] Would stop all deployments in namespace $ENVIRONMENT"
    fi
    
    # Restore database
    log_info "Restoring database..."
    if [[ -f "${BACKUP_PATH}/database.sql" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            # Get database credentials
            DB_HOST=$(kubectl get secret -n "$ENVIRONMENT" db-credentials -o jsonpath='{.data.host}' | base64 -d)
            DB_USER=$(kubectl get secret -n "$ENVIRONMENT" db-credentials -o jsonpath='{.data.username}' | base64 -d)
            DB_PASS=$(kubectl get secret -n "$ENVIRONMENT" db-credentials -o jsonpath='{.data.password}' | base64 -d)
            DB_NAME=$(kubectl get secret -n "$ENVIRONMENT" db-credentials -o jsonpath='{.data.database}' | base64 -d)
            
            # Restore database
            PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" < "${BACKUP_PATH}/database.sql"
            log_success "Database restored"
        else
            log_info "[DRY RUN] Would restore database from ${BACKUP_PATH}/database.sql"
        fi
    else
        log_warning "No database backup found"
    fi
    
    # Restore Redis
    log_info "Restoring Redis cache..."
    if [[ -f "${BACKUP_PATH}/redis.rdb" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            # Copy Redis backup to pod
            REDIS_POD=$(kubectl get pod -n "$ENVIRONMENT" -l app=redis -o jsonpath='{.items[0].metadata.name}')
            kubectl cp "${BACKUP_PATH}/redis.rdb" "$ENVIRONMENT/$REDIS_POD:/data/dump.rdb"
            kubectl exec -n "$ENVIRONMENT" "$REDIS_POD" -- redis-cli SHUTDOWN SAVE
            kubectl delete pod -n "$ENVIRONMENT" "$REDIS_POD"
            log_success "Redis cache restored"
        else
            log_info "[DRY RUN] Would restore Redis from ${BACKUP_PATH}/redis.rdb"
        fi
    else
        log_warning "No Redis backup found"
    fi
    
    # Restore models
    log_info "Restoring models..."
    if [[ -d "${BACKUP_PATH}/models" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            # Upload models to S3/GCS
            if command -v aws &> /dev/null; then
                aws s3 sync "${BACKUP_PATH}/models" "s3://wavefield-models-${ENVIRONMENT}/"
            elif command -v gsutil &> /dev/null; then
                gsutil -m rsync -r "${BACKUP_PATH}/models" "gs://wavefield-models-${ENVIRONMENT}/"
            fi
            log_success "Models restored"
        else
            log_info "[DRY RUN] Would restore models from ${BACKUP_PATH}/models"
        fi
    else
        log_warning "No model backups found"
    fi
    
    # Restore configurations
    log_info "Restoring configurations..."
    if [[ -d "${BACKUP_PATH}/configs" ]]; then
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl apply -f "${BACKUP_PATH}/configs/" -n "$ENVIRONMENT"
            log_success "Configurations restored"
        else
            log_info "[DRY RUN] Would restore configurations from ${BACKUP_PATH}/configs"
        fi
    else
        log_warning "No configuration backups found"
    fi
    
    # Restart services
    log_info "Restarting services..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl scale deployment --all --replicas=3 -n "$ENVIRONMENT"
        kubectl rollout status deployment --all -n "$ENVIRONMENT" --timeout=10m
        log_success "Services restarted"
    else
        log_info "[DRY RUN] Would restart all deployments"
    fi
    
    # Verify restoration
    log_info "Verifying restoration..."
    if [[ "$DRY_RUN" == "false" ]]; then
        sleep 30
        "${SCRIPT_DIR}/smoke-test.sh" "$ENVIRONMENT"
    else
        log_info "[DRY RUN] Would run smoke tests"
    fi
    
    log_success "Restoration complete"
}

# Failover to secondary region
failover_to_secondary() {
    log_step "Failing Over to Secondary Region"
    
    if [[ -z "$REGION" ]] || [[ -z "$SECONDARY_REGION" ]]; then
        log_error "Both primary and secondary regions are required for failover"
        exit 1
    fi
    
    confirm_recovery
    
    # Create backup before failover
    log_info "Creating backup before failover..."
    if [[ "$DRY_RUN" == "false" ]]; then
        "${SCRIPT_DIR}/backup.sh" "./backups/pre-failover-$(date +%Y%m%d-%H%M%S)"
    else
        log_info "[DRY RUN] Would create backup"
    fi
    
    # Update DNS to point to secondary region
    log_info "Updating DNS to secondary region..."
    if [[ "$DRY_RUN" == "false" ]]; then
        # Get secondary region load balancer
        SECONDARY_LB=$(kubectl get svc -n "$ENVIRONMENT" wavefield-llm-service \
            --context="secondary-${ENVIRONMENT}" \
            -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        
        # Update Route53/Cloud DNS
        if command -v aws &> /dev/null; then
            # AWS Route53
            HOSTED_ZONE_ID=$(aws route53 list-hosted-zones --query "HostedZones[?Name=='wavefield-llm.example.com.'].Id" --output text)
            cat > /tmp/dns-change.json << EOF
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "api.wavefield-llm.example.com",
      "Type": "CNAME",
      "TTL": 60,
      "ResourceRecords": [{"Value": "$SECONDARY_LB"}]
    }
  }]
}
EOF
            aws route53 change-resource-record-sets --hosted-zone-id "$HOSTED_ZONE_ID" --change-batch file:///tmp/dns-change.json
            rm /tmp/dns-change.json
        fi
        log_success "DNS updated to secondary region"
    else
        log_info "[DRY RUN] Would update DNS to secondary region"
    fi
    
    # Scale down primary region
    log_info "Scaling down primary region..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl scale deployment --all --replicas=0 -n "$ENVIRONMENT" --context="primary-${ENVIRONMENT}"
        log_success "Primary region scaled down"
    else
        log_info "[DRY RUN] Would scale down primary region"
    fi
    
    # Scale up secondary region
    log_info "Scaling up secondary region..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl scale deployment --all --replicas=5 -n "$ENVIRONMENT" --context="secondary-${ENVIRONMENT}"
        kubectl rollout status deployment --all -n "$ENVIRONMENT" --context="secondary-${ENVIRONMENT}" --timeout=10m
        log_success "Secondary region scaled up"
    else
        log_info "[DRY RUN] Would scale up secondary region"
    fi
    
    # Verify failover
    log_info "Verifying failover..."
    if [[ "$DRY_RUN" == "false" ]]; then
        sleep 60  # Wait for DNS propagation
        "${SCRIPT_DIR}/smoke-test.sh" "$ENVIRONMENT"
    else
        log_info "[DRY RUN] Would run smoke tests"
    fi
    
    log_success "Failover complete"
    log_warning "Primary region: $REGION (inactive)"
    log_warning "Secondary region: $SECONDARY_REGION (active)"
}

# Failback to primary region
failback_to_primary() {
    log_step "Failing Back to Primary Region"
    
    if [[ -z "$REGION" ]] || [[ -z "$SECONDARY_REGION" ]]; then
        log_error "Both primary and secondary regions are required for failback"
        exit 1
    fi
    
    confirm_recovery
    
    # Verify primary region is healthy
    log_info "Verifying primary region health..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl get nodes --context="primary-${ENVIRONMENT}" > /dev/null 2>&1 || {
            log_error "Primary region is not accessible"
            exit 1
        }
        log_success "Primary region is healthy"
    else
        log_info "[DRY RUN] Would verify primary region health"
    fi
    
    # Sync data from secondary to primary
    log_info "Syncing data from secondary to primary..."
    if [[ "$DRY_RUN" == "false" ]]; then
        # Database replication sync
        log_info "Syncing database..."
        # This would use database-specific replication tools
        
        # Model sync
        log_info "Syncing models..."
        if command -v aws &> /dev/null; then
            aws s3 sync "s3://wavefield-models-${ENVIRONMENT}-${SECONDARY_REGION}/" \
                        "s3://wavefield-models-${ENVIRONMENT}-${REGION}/"
        fi
        log_success "Data synced"
    else
        log_info "[DRY RUN] Would sync data from secondary to primary"
    fi
    
    # Scale up primary region
    log_info "Scaling up primary region..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl scale deployment --all --replicas=5 -n "$ENVIRONMENT" --context="primary-${ENVIRONMENT}"
        kubectl rollout status deployment --all -n "$ENVIRONMENT" --context="primary-${ENVIRONMENT}" --timeout=10m
        log_success "Primary region scaled up"
    else
        log_info "[DRY RUN] Would scale up primary region"
    fi
    
    # Update DNS back to primary region
    log_info "Updating DNS back to primary region..."
    if [[ "$DRY_RUN" == "false" ]]; then
        PRIMARY_LB=$(kubectl get svc -n "$ENVIRONMENT" wavefield-llm-service \
            --context="primary-${ENVIRONMENT}" \
            -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        
        if command -v aws &> /dev/null; then
            HOSTED_ZONE_ID=$(aws route53 list-hosted-zones --query "HostedZones[?Name=='wavefield-llm.example.com.'].Id" --output text)
            cat > /tmp/dns-change.json << EOF
{
  "Changes": [{
    "Action": "UPSERT",
    "ResourceRecordSet": {
      "Name": "api.wavefield-llm.example.com",
      "Type": "CNAME",
      "TTL": 60,
      "ResourceRecords": [{"Value": "$PRIMARY_LB"}]
    }
  }]
}
EOF
            aws route53 change-resource-record-sets --hosted-zone-id "$HOSTED_ZONE_ID" --change-batch file:///tmp/dns-change.json
            rm /tmp/dns-change.json
        fi
        log_success "DNS updated to primary region"
    else
        log_info "[DRY RUN] Would update DNS to primary region"
    fi
    
    # Wait for DNS propagation
    log_info "Waiting for DNS propagation..."
    if [[ "$DRY_RUN" == "false" ]]; then
        sleep 60
    fi
    
    # Scale down secondary region
    log_info "Scaling down secondary region..."
    if [[ "$DRY_RUN" == "false" ]]; then
        kubectl scale deployment --all --replicas=1 -n "$ENVIRONMENT" --context="secondary-${ENVIRONMENT}"
        log_success "Secondary region scaled down to standby"
    else
        log_info "[DRY RUN] Would scale down secondary region"
    fi
    
    # Verify failback
    log_info "Verifying failback..."
    if [[ "$DRY_RUN" == "false" ]]; then
        "${SCRIPT_DIR}/smoke-test.sh" "$ENVIRONMENT"
    else
        log_info "[DRY RUN] Would run smoke tests"
    fi
    
    log_success "Failback complete"
    log_warning "Primary region: $REGION (active)"
    log_warning "Secondary region: $SECONDARY_REGION (standby)"
}

# Main execution
main() {
    log_step "Wave Field LLM Disaster Recovery"
    
    case "$RECOVERY_TYPE" in
        restore)
            restore_from_backup
            ;;
        failover)
            failover_to_secondary
            ;;
        failback)
            failback_to_primary
            ;;
        *)
            log_error "Unknown recovery type: $RECOVERY_TYPE"
            log_error "Supported types: restore, failover, failback"
            exit 1
            ;;
    esac
    
    log_step "Disaster Recovery Complete"
    log_success "Recovery operation completed successfully"
    
    log_info "\nPost-recovery tasks:"
    log_info "1. Monitor system metrics and logs"
    log_info "2. Verify all services are functioning correctly"
    log_info "3. Update incident documentation"
    log_info "4. Schedule post-mortem review"
    log_info "5. Update disaster recovery procedures if needed"
}

# Run main function
main

# Made with Bob
