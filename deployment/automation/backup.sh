#!/bin/bash
set -euo pipefail
# Wave Field LLM - Backup Script
BACKUP_DIR="${1:-./backups/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$BACKUP_DIR"
echo "Creating backup in: $BACKUP_DIR"
kubectl get all -A -o yaml > "$BACKUP_DIR/all-resources.yaml"
kubectl get configmaps -A -o yaml > "$BACKUP_DIR/configmaps.yaml"
kubectl get secrets -A -o yaml > "$BACKUP_DIR/secrets.yaml"
echo "Backup completed: $BACKUP_DIR"
