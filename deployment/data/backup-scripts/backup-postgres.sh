#!/bin/bash
# PostgreSQL Backup Script for Wave Field LLM
# Automated backups with compression, encryption, and S3 upload

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups/postgres}"
S3_BUCKET="${S3_BUCKET:-s3://wavefield-llm-backups/postgres}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-wavefield}"
DB_USER="${DB_USER:-postgres}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create backup directory
mkdir -p "${BACKUP_DIR}"

# Generate backup filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/wavefield_${TIMESTAMP}.sql"
COMPRESSED_FILE="${BACKUP_FILE}.gz"
ENCRYPTED_FILE="${COMPRESSED_FILE}.enc"

log_info "Starting PostgreSQL backup..."
log_info "Database: ${DB_NAME}"
log_info "Timestamp: ${TIMESTAMP}"

# Perform backup
log_info "Creating database dump..."
PGPASSWORD="${DB_PASSWORD}" pg_dump \
    -h "${DB_HOST}" \
    -p "${DB_PORT}" \
    -U "${DB_USER}" \
    -d "${DB_NAME}" \
    -F p \
    -f "${BACKUP_FILE}"

# Compress backup
log_info "Compressing backup..."
gzip -9 "${BACKUP_FILE}"

# Encrypt if key provided
if [ -n "${ENCRYPTION_KEY}" ]; then
    log_info "Encrypting backup..."
    openssl enc -aes-256-cbc -salt -pbkdf2 \
        -in "${COMPRESSED_FILE}" \
        -out "${ENCRYPTED_FILE}" \
        -k "${ENCRYPTION_KEY}"
    rm "${COMPRESSED_FILE}"
    FINAL_FILE="${ENCRYPTED_FILE}"
else
    FINAL_FILE="${COMPRESSED_FILE}"
fi

# Upload to S3
log_info "Uploading to S3..."
aws s3 cp "${FINAL_FILE}" "${S3_BUCKET}/" \
    --storage-class STANDARD_IA \
    --metadata "timestamp=${TIMESTAMP},database=${DB_NAME}"

# Verify upload
if aws s3 ls "${S3_BUCKET}/$(basename ${FINAL_FILE})" > /dev/null; then
    log_info "Backup uploaded successfully"
    rm "${FINAL_FILE}"
else
    log_error "Failed to verify S3 upload"
    exit 1
fi

# Clean old backups
log_info "Cleaning old backups (older than ${RETENTION_DAYS} days)..."
find "${BACKUP_DIR}" -name "wavefield_*.sql.gz*" -mtime +${RETENTION_DAYS} -delete

# Clean old S3 backups
aws s3 ls "${S3_BUCKET}/" | while read -r line; do
    FILE_DATE=$(echo "$line" | awk '{print $1}')
    FILE_NAME=$(echo "$line" | awk '{print $4}')
    FILE_AGE=$(( ($(date +%s) - $(date -d "$FILE_DATE" +%s)) / 86400 ))
    
    if [ ${FILE_AGE} -gt ${RETENTION_DAYS} ]; then
        log_info "Deleting old backup: ${FILE_NAME}"
        aws s3 rm "${S3_BUCKET}/${FILE_NAME}"
    fi
done

log_info "Backup completed successfully"

# Made with Bob
