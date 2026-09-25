#!/bin/bash
set -euo pipefail
# Wave Field LLM - Scale Script
MODEL="${1:-wavefield-small}"
REPLICAS="${2:-3}"
NAMESPACE="${3:-default}"
echo "Scaling $MODEL to $REPLICAS replicas in namespace: $NAMESPACE"
kubectl scale deployment/wavefield-${MODEL} -n "$NAMESPACE" --replicas="$REPLICAS"
kubectl rollout status deployment/wavefield-${MODEL} -n "$NAMESPACE"
echo "Scaling completed"
