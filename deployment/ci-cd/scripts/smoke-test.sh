#!/bin/bash
# Smoke Test Script for Wave Field LLM
# Validates basic functionality after deployment

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SERVICE_URL="${1:-localhost:8000}"
TIMEOUT="${TIMEOUT:-30}"
MAX_RETRIES="${MAX_RETRIES:-5}"

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_test_pass() { echo -e "${GREEN}[PASS]${NC} $1"; TESTS_PASSED=$((TESTS_PASSED + 1)); }
log_test_fail() { echo -e "${RED}[FAIL]${NC} $1"; TESTS_FAILED=$((TESTS_FAILED + 1)); }

# Wait for service to be ready
wait_for_service() {
    log_info "Waiting for service at ${SERVICE_URL}..."
    
    local retries=0
    while [ ${retries} -lt ${MAX_RETRIES} ]; do
        if curl -sf "https://${SERVICE_URL}/health" > /dev/null 2>&1; then
            log_info "Service is ready"
            return 0
        fi
        
        retries=$((retries + 1))
        log_info "Waiting for service... (attempt ${retries}/${MAX_RETRIES})"
        sleep 5
    done
    
    log_error "Service did not become ready"
    return 1
}

# Test health endpoint
test_health() {
    log_info "Testing health endpoint..."
    
    local response=$(curl -sf -w "\n%{http_code}" "https://${SERVICE_URL}/health" 2>/dev/null || echo "000")
    local body=$(echo "${response}" | head -n -1)
    local status=$(echo "${response}" | tail -n 1)
    
    if [ "${status}" = "200" ]; then
        log_test_pass "Health endpoint returned 200"
        
        # Check response body
        if echo "${body}" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
            log_test_pass "Health status is healthy"
        else
            log_test_fail "Health status is not healthy: ${body}"
        fi
    else
        log_test_fail "Health endpoint returned ${status}"
    fi
}

# Test metrics endpoint
test_metrics() {
    log_info "Testing metrics endpoint..."
    
    local status=$(curl -sf -o /dev/null -w "%{http_code}" "https://${SERVICE_URL}/metrics" 2>/dev/null || echo "000")
    
    if [ "${status}" = "200" ]; then
        log_test_pass "Metrics endpoint returned 200"
    else
        log_test_fail "Metrics endpoint returned ${status}"
    fi
}

# Test inference endpoint
test_inference() {
    log_info "Testing inference endpoint..."
    
    local payload='{
        "prompt": "Hello, world!",
        "max_tokens": 10,
        "temperature": 0.7
    }'
    
    local response=$(curl -sf -w "\n%{http_code}" \
        -X POST "https://${SERVICE_URL}/v1/inference" \
        -H "Content-Type: application/json" \
        -d "${payload}" 2>/dev/null || echo -e "\n000")
    
    local body=$(echo "${response}" | head -n -1)
    local status=$(echo "${response}" | tail -n 1)
    
    if [ "${status}" = "200" ]; then
        log_test_pass "Inference endpoint returned 200"
        
        # Check response structure
        if echo "${body}" | jq -e '.text' > /dev/null 2>&1; then
            log_test_pass "Inference response contains text"
        else
            log_test_fail "Inference response missing text field"
        fi
        
        if echo "${body}" | jq -e '.tokens' > /dev/null 2>&1; then
            log_test_pass "Inference response contains token count"
        else
            log_test_fail "Inference response missing tokens field"
        fi
    else
        log_test_fail "Inference endpoint returned ${status}: ${body}"
    fi
}

# Test streaming endpoint
test_streaming() {
    log_info "Testing streaming endpoint..."
    
    local payload='{
        "prompt": "Count to 5:",
        "max_tokens": 20,
        "stream": true
    }'
    
    local response=$(curl -sf -w "\n%{http_code}" \
        -X POST "https://${SERVICE_URL}/v1/stream" \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        --max-time ${TIMEOUT} 2>/dev/null || echo -e "\n000")
    
    local status=$(echo "${response}" | tail -n 1)
    
    if [ "${status}" = "200" ]; then
        log_test_pass "Streaming endpoint returned 200"
    else
        log_test_fail "Streaming endpoint returned ${status}"
    fi
}

# Test model info endpoint
test_model_info() {
    log_info "Testing model info endpoint..."
    
    local response=$(curl -sf -w "\n%{http_code}" "https://${SERVICE_URL}/v1/models" 2>/dev/null || echo -e "\n000")
    local body=$(echo "${response}" | head -n -1)
    local status=$(echo "${response}" | tail -n 1)
    
    if [ "${status}" = "200" ]; then
        log_test_pass "Model info endpoint returned 200"
        
        # Check for model list
        if echo "${body}" | jq -e '.models | length > 0' > /dev/null 2>&1; then
            log_test_pass "Model list is not empty"
        else
            log_test_fail "Model list is empty"
        fi
    else
        log_test_fail "Model info endpoint returned ${status}"
    fi
}

# Test error handling
test_error_handling() {
    log_info "Testing error handling..."
    
    # Test with invalid payload
    local response=$(curl -sf -w "\n%{http_code}" \
        -X POST "https://${SERVICE_URL}/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{"invalid": "payload"}' 2>/dev/null || echo -e "\n000")
    
    local status=$(echo "${response}" | tail -n 1)
    
    if [ "${status}" = "400" ] || [ "${status}" = "422" ]; then
        log_test_pass "Invalid payload returns appropriate error code (${status})"
    else
        log_test_fail "Invalid payload returned unexpected status: ${status}"
    fi
}

# Test rate limiting
test_rate_limiting() {
    log_info "Testing rate limiting..."
    
    local requests=0
    local rate_limited=false
    
    # Send multiple requests quickly
    for i in {1..20}; do
        local status=$(curl -sf -o /dev/null -w "%{http_code}" "https://${SERVICE_URL}/health" 2>/dev/null || echo "000")
        
        if [ "${status}" = "429" ]; then
            rate_limited=true
            break
        fi
        
        requests=$((requests + 1))
    done
    
    if [ "${rate_limited}" = true ]; then
        log_test_pass "Rate limiting is working (triggered after ${requests} requests)"
    else
        log_warn "Rate limiting not triggered (sent ${requests} requests)"
    fi
}

# Test authentication
test_authentication() {
    log_info "Testing authentication..."
    
    # Test without auth token
    local status=$(curl -sf -o /dev/null -w "%{http_code}" \
        -X POST "https://${SERVICE_URL}/v1/inference" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "test"}' 2>/dev/null || echo "000")
    
    if [ "${status}" = "401" ] || [ "${status}" = "403" ]; then
        log_test_pass "Unauthenticated request returns ${status}"
    else
        log_warn "Authentication may not be enforced (status: ${status})"
    fi
}

# Test CORS headers
test_cors() {
    log_info "Testing CORS headers..."
    
    local headers=$(curl -sf -I "https://${SERVICE_URL}/health" 2>/dev/null || echo "")
    
    if echo "${headers}" | grep -qi "access-control-allow-origin"; then
        log_test_pass "CORS headers are present"
    else
        log_warn "CORS headers not found"
    fi
}

# Test response time
test_response_time() {
    log_info "Testing response time..."
    
    local start=$(date +%s%N)
    curl -sf "https://${SERVICE_URL}/health" > /dev/null 2>&1
    local end=$(date +%s%N)
    
    local duration=$(( (end - start) / 1000000 ))  # Convert to milliseconds
    
    if [ ${duration} -lt 1000 ]; then
        log_test_pass "Response time is acceptable (${duration}ms)"
    else
        log_test_fail "Response time is too slow (${duration}ms)"
    fi
}

# Main test execution
main() {
    log_info "Starting smoke tests for ${SERVICE_URL}"
    log_info "Timeout: ${TIMEOUT}s, Max retries: ${MAX_RETRIES}"
    
    # Wait for service
    if ! wait_for_service; then
        log_error "Service is not available, aborting tests"
        exit 1
    fi
    
    # Run all tests
    test_health
    test_metrics
    test_model_info
    test_inference
    test_streaming
    test_error_handling
    test_rate_limiting
    test_authentication
    test_cors
    test_response_time
    
    # Summary
    echo ""
    log_info "========================================="
    log_info "Smoke Test Summary"
    log_info "========================================="
    log_info "Tests Passed: ${TESTS_PASSED}"
    log_info "Tests Failed: ${TESTS_FAILED}"
    log_info "Total Tests:  $((TESTS_PASSED + TESTS_FAILED))"
    log_info "========================================="
    
    if [ ${TESTS_FAILED} -eq 0 ]; then
        log_info "All smoke tests passed! ✅"
        exit 0
    else
        log_error "Some smoke tests failed! ❌"
        exit 1
    fi
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [SERVICE_URL]

Run smoke tests against Wave Field LLM deployment

Arguments:
    SERVICE_URL    Service URL to test (default: localhost:8000)

Environment Variables:
    TIMEOUT        Request timeout in seconds (default: 30)
    MAX_RETRIES    Maximum retries for service readiness (default: 5)

Examples:
    $0 api.wavefield-llm.example.com
    TIMEOUT=60 $0 staging.wavefield-llm.example.com

EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

main "$@"

# Made with Bob
