#!/bin/bash
set -euo pipefail
# Wave Field LLM - Smoke Test Script
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
NAMESPACE="${1:-default}"
echo "Running smoke tests on namespace: $NAMESPACE"
# Test 1: Check pods are running
PODS=$(kubectl get pods -n "$NAMESPACE" -l app=wavefield-llm --field-selector=status.phase=Running -o name | wc -l)
[[ $PODS -gt 0 ]] && echo -e "${GREEN}✓${NC} Pods running: $PODS" || { echo -e "${RED}✗${NC} No pods running"; exit 1; }
# Test 2: Check service endpoints
SVC=$(kubectl get svc -n "$NAMESPACE" -l app=wavefield-llm -o name | head -1)
[[ -n "$SVC" ]] && echo -e "${GREEN}✓${NC} Service exists" || { echo -e "${RED}✗${NC} No service found"; exit 1; }
# Test 3: Health check
POD=$(kubectl get pods -n "$NAMESPACE" -l app=wavefield-llm -o name | head -1)
if [[ -n "$POD" ]]; then
  kubectl exec -n "$NAMESPACE" "$POD" -- curl -sf http://localhost:8000/health >/dev/null && \
    echo -e "${GREEN}✓${NC} Health check passed" || echo -e "${RED}✗${NC} Health check failed"
fi
echo -e "${GREEN}Smoke tests completed${NC}"
