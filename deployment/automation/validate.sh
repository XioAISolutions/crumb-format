#!/bin/bash
set -euo pipefail
# Wave Field LLM - Validation Script
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }
ERRORS=0; WARNINGS=0
log_info "Validating deployment..."
command -v kubectl &>/dev/null && log_success "kubectl installed" || { log_error "kubectl missing"; ((ERRORS++)); }
command -v helm &>/dev/null && log_success "helm installed" || { log_error "helm missing"; ((ERRORS++)); }
kubectl cluster-info &>/dev/null && log_success "Kubernetes connected" || { log_error "Kubernetes not connected"; ((ERRORS++)); }
echo "Validation: $ERRORS errors, $WARNINGS warnings"
[[ $ERRORS -eq 0 ]] && exit 0 || exit 1
