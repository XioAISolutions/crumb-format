#!/bin/bash
set -euo pipefail
# Wave Field LLM - Cleanup Script
NAMESPACE="${1:-default}"
echo "Cleaning up old resources in namespace: $NAMESPACE"
kubectl delete pods -n "$NAMESPACE" --field-selector=status.phase=Failed
kubectl delete pods -n "$NAMESPACE" --field-selector=status.phase=Succeeded
echo "Cleanup completed"
