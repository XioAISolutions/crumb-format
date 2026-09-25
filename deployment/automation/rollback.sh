#!/bin/bash
set -euo pipefail
# Wave Field LLM - Rollback Script
MODEL="${1:-wavefield-small}"
NAMESPACE="${2:-default}"
echo "Rolling back deployment: $MODEL in namespace: $NAMESPACE"
kubectl rollout undo deployment/wavefield-${MODEL} -n "$NAMESPACE"
kubectl rollout status deployment/wavefield-${MODEL} -n "$NAMESPACE" --timeout=5m
echo "Rollback completed successfully"
