#!/bin/bash
set -euo pipefail

# Wave Field LLM - Monitoring Setup Script
# Deploy complete monitoring stack (Prometheus, Grafana, Loki, Jaeger)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT="production"
NAMESPACE="monitoring"
GRAFANA_PASSWORD=""
ENABLE_LOKI=true
ENABLE_JAEGER=true
STORAGE_CLASS="standard"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}\n"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup monitoring stack for Wave Field LLM

Options:
    --environment ENV       Environment name (default: production)
    --namespace NS          Kubernetes namespace (default: monitoring)
    --grafana-password PWD  Grafana admin password (generated if not provided)
    --no-loki               Disable Loki log aggregation
    --no-jaeger             Disable Jaeger tracing
    --storage-class CLASS   Storage class for PVCs (default: standard)
    -h, --help              Show this help

Examples:
    # Basic setup
    $0 --environment production

    # Custom namespace with password
    $0 --namespace observability --grafana-password mypassword

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment) ENVIRONMENT="$2"; shift 2 ;;
            --namespace) NAMESPACE="$2"; shift 2 ;;
            --grafana-password) GRAFANA_PASSWORD="$2"; shift 2 ;;
            --no-loki) ENABLE_LOKI=false; shift ;;
            --no-jaeger) ENABLE_JAEGER=false; shift ;;
            --storage-class) STORAGE_CLASS="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown option: $1"; usage ;;
        esac
    done

    [[ -z "$GRAFANA_PASSWORD" ]] && GRAFANA_PASSWORD=$(openssl rand -base64 16)
}

check_prerequisites() {
    log_step "Checking Prerequisites"

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not installed"
        exit 1
    fi

    if ! command -v helm &> /dev/null; then
        log_error "helm not installed"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

create_namespace() {
    log_step "Creating Namespace"

    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    log_success "Namespace created: $NAMESPACE"
}

install_prometheus() {
    log_step "Installing Prometheus"

    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    # Create values file
    cat > /tmp/prometheus-values.yaml << EOF
server:
  persistentVolume:
    enabled: true
    storageClass: ${STORAGE_CLASS}
    size: 50Gi
  retention: 30d
  
  global:
    scrape_interval: 15s
    evaluation_interval: 15s

alertmanager:
  enabled: true
  persistentVolume:
    enabled: true
    storageClass: ${STORAGE_CLASS}
    size: 10Gi

pushgateway:
  enabled: true

nodeExporter:
  enabled: true

kubeStateMetrics:
  enabled: true

serverFiles:
  prometheus.yml:
    scrape_configs:
      - job_name: 'wavefield-llm'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - wavefield-${ENVIRONMENT}
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: wavefield-llm
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          - source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
EOF

    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/prometheus \
        -n "$NAMESPACE" \
        -f /tmp/prometheus-values.yaml \
        --wait

    log_success "Prometheus installed"
}

install_grafana() {
    log_step "Installing Grafana"

    # Add Grafana Helm repo
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update

    # Create values file
    cat > /tmp/grafana-values.yaml << EOF
adminPassword: ${GRAFANA_PASSWORD}

persistence:
  enabled: true
  storageClassName: ${STORAGE_CLASS}
  size: 10Gi

datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        url: http://prometheus-server.${NAMESPACE}.svc.cluster.local
        access: proxy
        isDefault: true
      - name: Loki
        type: loki
        url: http://loki.${NAMESPACE}.svc.cluster.local:3100
        access: proxy

dashboardProviders:
  dashboardproviders.yaml:
    apiVersion: 1
    providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards/default

dashboards:
  default:
    wavefield-overview:
      file: dashboards/system-overview.json
    wavefield-performance:
      file: dashboards/model-performance.json

service:
  type: LoadBalancer
  port: 80
EOF

    # Copy dashboard files
    mkdir -p /tmp/grafana-dashboards
    cp "${PROJECT_ROOT}/deployment/monitoring/grafana-dashboards/"*.json /tmp/grafana-dashboards/ 2>/dev/null || true

    # Install Grafana
    helm upgrade --install grafana grafana/grafana \
        -n "$NAMESPACE" \
        -f /tmp/grafana-values.yaml \
        --set-file dashboards.default.wavefield-overview.json="${PROJECT_ROOT}/deployment/monitoring/grafana-dashboards/system-overview.json" \
        --set-file dashboards.default.wavefield-performance.json="${PROJECT_ROOT}/deployment/monitoring/grafana-dashboards/model-performance.json" \
        --wait

    log_success "Grafana installed"
}

install_loki() {
    if [[ "$ENABLE_LOKI" != true ]]; then
        log_warning "Loki installation skipped"
        return 0
    fi

    log_step "Installing Loki"

    # Create values file
    cat > /tmp/loki-values.yaml << EOF
loki:
  auth_enabled: false
  commonConfig:
    replication_factor: 1
  storage:
    type: filesystem

singleBinary:
  replicas: 1
  persistence:
    enabled: true
    storageClass: ${STORAGE_CLASS}
    size: 30Gi

monitoring:
  selfMonitoring:
    enabled: false
  lokiCanary:
    enabled: false

test:
  enabled: false
EOF

    # Install Loki
    helm upgrade --install loki grafana/loki \
        -n "$NAMESPACE" \
        -f /tmp/loki-values.yaml \
        --wait

    # Install Promtail for log collection
    cat > /tmp/promtail-values.yaml << EOF
config:
  clients:
    - url: http://loki.${NAMESPACE}.svc.cluster.local:3100/loki/api/v1/push
EOF

    helm upgrade --install promtail grafana/promtail \
        -n "$NAMESPACE" \
        -f /tmp/promtail-values.yaml \
        --wait

    log_success "Loki and Promtail installed"
}

install_jaeger() {
    if [[ "$ENABLE_JAEGER" != true ]]; then
        log_warning "Jaeger installation skipped"
        return 0
    fi

    log_step "Installing Jaeger"

    # Install Jaeger operator
    kubectl create namespace observability --dry-run=client -o yaml | kubectl apply -f -
    kubectl apply -f https://github.com/jaegertracing/jaeger-operator/releases/download/v1.49.0/jaeger-operator.yaml -n observability

    # Wait for operator
    sleep 10

    # Create Jaeger instance
    cat << EOF | kubectl apply -f -
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: wavefield-jaeger
  namespace: ${NAMESPACE}
spec:
  strategy: production
  storage:
    type: elasticsearch
    options:
      es:
        server-urls: http://elasticsearch.${NAMESPACE}.svc.cluster.local:9200
  ingress:
    enabled: false
EOF

    log_success "Jaeger installed"
}

deploy_custom_metrics() {
    log_step "Deploying Custom Metrics Exporter"

    # Create ConfigMap for custom metrics script
    kubectl create configmap wavefield-metrics-exporter \
        -n "$NAMESPACE" \
        --from-file="${PROJECT_ROOT}/deployment/monitoring/custom-metrics.py" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Create deployment for metrics exporter
    cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wavefield-metrics-exporter
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: metrics-exporter
  template:
    metadata:
      labels:
        app: metrics-exporter
    spec:
      containers:
      - name: exporter
        image: python:3.11-slim
        command: ["python", "/app/custom-metrics.py"]
        ports:
        - containerPort: 9090
          name: metrics
        volumeMounts:
        - name: metrics-script
          mountPath: /app
      volumes:
      - name: metrics-script
        configMap:
          name: wavefield-metrics-exporter
---
apiVersion: v1
kind: Service
metadata:
  name: wavefield-metrics-exporter
  namespace: ${NAMESPACE}
spec:
  selector:
    app: metrics-exporter
  ports:
  - port: 9090
    targetPort: 9090
    name: metrics
EOF

    log_success "Custom metrics exporter deployed"
}

configure_alerting() {
    log_step "Configuring Alerting Rules"

    # Copy alertmanager config
    kubectl create configmap alertmanager-config \
        -n "$NAMESPACE" \
        --from-file="${PROJECT_ROOT}/deployment/monitoring/alertmanager.yml" \
        --dry-run=client -o yaml | kubectl apply -f -

    log_success "Alerting configured"
}

print_access_info() {
    log_step "Monitoring Stack Deployed!"

    # Get Grafana service info
    local grafana_ip=$(kubectl get svc grafana -n "$NAMESPACE" \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

    cat << EOF
${GREEN}Monitoring stack deployed successfully!${NC}

Access Information:

Grafana:
  URL:      http://${grafana_ip} (or use port-forward)
  Username: admin
  Password: ${GRAFANA_PASSWORD}
  
  Port-forward command:
  kubectl port-forward -n ${NAMESPACE} svc/grafana 3000:80

Prometheus:
  Port-forward command:
  kubectl port-forward -n ${NAMESPACE} svc/prometheus-server 9090:80

EOF

    if [[ "$ENABLE_LOKI" == true ]]; then
        cat << EOF
Loki:
  Port-forward command:
  kubectl port-forward -n ${NAMESPACE} svc/loki 3100:3100

EOF
    fi

    if [[ "$ENABLE_JAEGER" == true ]]; then
        cat << EOF
Jaeger:
  Port-forward command:
  kubectl port-forward -n ${NAMESPACE} svc/wavefield-jaeger-query 16686:16686

EOF
    fi

    cat << EOF
Useful Commands:
  # View all monitoring resources
  kubectl get all -n ${NAMESPACE}

  # Check Prometheus targets
  kubectl port-forward -n ${NAMESPACE} svc/prometheus-server 9090:80
  # Then visit: http://localhost:9090/targets

  # View Grafana dashboards
  kubectl port-forward -n ${NAMESPACE} svc/grafana 3000:80
  # Then visit: http://localhost:3000

Save your Grafana password: ${GRAFANA_PASSWORD}
EOF
}

main() {
    log_info "Starting Monitoring Setup"
    
    parse_args "$@"
    check_prerequisites
    create_namespace
    install_prometheus
    install_grafana
    install_loki
    install_jaeger
    deploy_custom_metrics
    configure_alerting
    print_access_info

    log_success "Monitoring setup completed!"
}

main "$@"

# Made with Bob
