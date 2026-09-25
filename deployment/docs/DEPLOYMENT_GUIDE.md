# Wave Field LLM Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Deployment Options](#deployment-options)
5. [Step-by-Step Deployment](#step-by-step-deployment)
6. [Configuration](#configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

## Overview

This guide provides comprehensive instructions for deploying Wave Field LLM in production environments. The deployment supports multiple platforms including Kubernetes, AWS EKS, Google GKE, and Azure AKS.

**Estimated Deployment Time:** 2-4 hours  
**Skill Level Required:** Intermediate to Advanced  
**Supported Platforms:** Kubernetes 1.24+, Docker 20.10+

## Prerequisites

### Required Tools

```bash
# Verify tool versions
kubectl version --client  # >= 1.24
helm version             # >= 3.10
docker version           # >= 20.10
terraform version        # >= 1.3 (if using IaC)
aws --version           # >= 2.0 (for AWS deployments)
```

### Required Access

- [ ] Kubernetes cluster admin access
- [ ] Container registry credentials
- [ ] Cloud provider credentials (AWS/GCP/Azure)
- [ ] Domain name and DNS management
- [ ] SSL/TLS certificates

### Resource Requirements

#### Minimum (Development)
- **Nodes:** 3 nodes
- **CPU:** 8 cores per node
- **Memory:** 16 GB per node
- **Storage:** 100 GB per node
- **GPU:** Optional

#### Recommended (Production)
- **Nodes:** 5+ nodes
- **CPU:** 16 cores per node
- **Memory:** 64 GB per node
- **Storage:** 500 GB per node
- **GPU:** NVIDIA A100 or equivalent (for inference)

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Internet / Users                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   CloudFlare / CDN                           │
│                   (DDoS Protection)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Load Balancer (ALB/NLB)                     │
│                  (SSL Termination)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   NGINX Ingress Controller                   │
│            (Rate Limiting, Caching, Routing)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  API Pods   │  │ Inference   │  │  Monitoring │
│  (FastAPI)  │  │   Pods      │  │    Stack    │
│             │  │ (PyTorch)   │  │             │
└──────┬──────┘  └──────┬──────┘  └─────────────┘
       │                │
       │                │
       ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │PostgreSQL│  │  Redis   │  │    S3    │                  │
│  │          │  │  Cache   │  │  Models  │                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Purpose | Replicas | Resources |
|-----------|---------|----------|-----------|
| API Service | REST API endpoints | 3-10 | 2 CPU, 4GB RAM |
| Inference Service | Model inference | 2-5 | 8 CPU, 16GB RAM, 1 GPU |
| PostgreSQL | Primary database | 1 primary + 2 replicas | 4 CPU, 16GB RAM |
| Redis | Caching layer | 3 (cluster) | 2 CPU, 8GB RAM |
| NGINX | API Gateway | 2-3 | 1 CPU, 2GB RAM |
| Prometheus | Metrics collection | 1 | 2 CPU, 8GB RAM |
| Grafana | Visualization | 1 | 1 CPU, 2GB RAM |

## Deployment Options

### Option 1: Kubernetes with Helm (Recommended)

**Pros:** Easy management, rollback support, templating  
**Cons:** Requires Helm knowledge  
**Best For:** Production deployments

### Option 2: Kubernetes with kubectl

**Pros:** Direct control, no additional tools  
**Cons:** Manual management, no templating  
**Best For:** Development, testing

### Option 3: Docker Compose

**Pros:** Simple setup, good for development  
**Cons:** Not production-ready, limited scaling  
**Best For:** Local development only

### Option 4: Cloud-Managed Services

**Pros:** Managed infrastructure, auto-scaling  
**Cons:** Vendor lock-in, higher costs  
**Best For:** Enterprise deployments

## Step-by-Step Deployment

### Phase 1: Infrastructure Setup (30-60 minutes)

#### 1.1 Create Kubernetes Cluster

**AWS EKS:**
```bash
# Using eksctl
eksctl create cluster \
  --name wavefield-llm-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type m5.2xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --name wavefield-llm-prod --region us-east-1
```

**GKE:**
```bash
# Create cluster
gcloud container clusters create wavefield-llm-prod \
  --region us-central1 \
  --machine-type n1-standard-8 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials wavefield-llm-prod --region us-central1
```

**Azure AKS:**
```bash
# Create resource group
az group create --name wavefield-llm-rg --location eastus

# Create cluster
az aks create \
  --resource-group wavefield-llm-rg \
  --name wavefield-llm-prod \
  --node-count 3 \
  --node-vm-size Standard_D8s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group wavefield-llm-rg --name wavefield-llm-prod
```

#### 1.2 Install Required Add-ons

```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=2

# Install cert-manager for SSL
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

#### 1.3 Create Namespaces

```bash
# Create application namespace
kubectl create namespace wavefield-llm

# Create monitoring namespace
kubectl create namespace monitoring

# Label namespaces
kubectl label namespace wavefield-llm environment=production
kubectl label namespace monitoring environment=production
```

### Phase 2: Storage and Database Setup (30-45 minutes)

#### 2.1 Deploy PostgreSQL

```bash
# Create storage class (if needed)
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
volumeBindingMode: WaitForFirstConsumer
EOF

# Deploy PostgreSQL using Helm
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgresql bitnami/postgresql \
  --namespace wavefield-llm \
  --set auth.username=wavefield \
  --set auth.password=$(openssl rand -base64 32) \
  --set auth.database=wavefield \
  --set primary.persistence.size=100Gi \
  --set primary.persistence.storageClass=fast-ssd \
  --set readReplicas.replicaCount=2 \
  --set metrics.enabled=true

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql \
  -n wavefield-llm --timeout=300s
```

#### 2.2 Deploy Redis

```bash
# Deploy Redis cluster
helm install redis bitnami/redis \
  --namespace wavefield-llm \
  --set architecture=replication \
  --set auth.password=$(openssl rand -base64 32) \
  --set master.persistence.size=20Gi \
  --set replica.replicaCount=2 \
  --set replica.persistence.size=20Gi \
  --set metrics.enabled=true

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis \
  -n wavefield-llm --timeout=300s
```

#### 2.3 Initialize Database Schema

```bash
# Get PostgreSQL password
export POSTGRES_PASSWORD=$(kubectl get secret --namespace wavefield-llm postgresql \
  -o jsonpath="{.data.password}" | base64 -d)

# Run migrations
kubectl run postgresql-client --rm --tty -i --restart='Never' \
  --namespace wavefield-llm \
  --image docker.io/bitnami/postgresql:15 \
  --env="PGPASSWORD=$POSTGRES_PASSWORD" \
  --command -- psql --host postgresql -U wavefield -d wavefield -f /migrations/schema.sql
```

### Phase 3: Application Deployment (45-60 minutes)

#### 3.1 Create Secrets

```bash
# Create application secrets
kubectl create secret generic wavefield-secrets \
  --namespace wavefield-llm \
  --from-literal=database-url="postgresql://wavefield:${POSTGRES_PASSWORD}@postgresql:5432/wavefield" \
  --from-literal=redis-url="redis://:${REDIS_PASSWORD}@redis-master:6379/0" \
  --from-literal=jwt-secret=$(openssl rand -base64 32) \
  --from-literal=api-key=$(openssl rand -base64 32)

# Create image pull secret (if using private registry)
kubectl create secret docker-registry regcred \
  --namespace wavefield-llm \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN
```

#### 3.2 Deploy Application with Helm

```bash
# Add Helm repository (if published)
helm repo add wavefield https://charts.wavefield-llm.example.com
helm repo update

# Install Wave Field LLM
helm install wavefield-llm wavefield/wavefield-llm \
  --namespace wavefield-llm \
  --values deployment/kubernetes/helm/wavefield-llm/values-production.yaml \
  --set image.tag=v1.0.0 \
  --set replicaCount=3 \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=3 \
  --set autoscaling.maxReplicas=10 \
  --wait \
  --timeout 10m

# Or install from local chart
helm install wavefield-llm ./deployment/kubernetes/helm/wavefield-llm \
  --namespace wavefield-llm \
  --values deployment/kubernetes/helm/wavefield-llm/values-production.yaml \
  --wait
```

#### 3.3 Verify Deployment

```bash
# Check pod status
kubectl get pods -n wavefield-llm

# Check services
kubectl get svc -n wavefield-llm

# Check ingress
kubectl get ingress -n wavefield-llm

# View logs
kubectl logs -n wavefield-llm -l app=wavefield-api --tail=50
```

### Phase 4: Monitoring Setup (30 minutes)

#### 4.1 Deploy Prometheus Stack

```bash
# Install kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
  --set grafana.adminPassword=$(openssl rand -base64 32)
```

#### 4.2 Configure Monitoring

```bash
# Apply custom Prometheus configuration
kubectl apply -f deployment/monitoring/prometheus.yml

# Apply ServiceMonitors
kubectl apply -f deployment/monitoring/servicemonitors/

# Import Grafana dashboards
kubectl create configmap grafana-dashboards \
  --from-file=deployment/monitoring/grafana-dashboards/ \
  -n monitoring
```

### Phase 5: Security Configuration (20 minutes)

#### 5.1 Apply RBAC Policies

```bash
# Apply RBAC configuration
kubectl apply -f deployment/security/rbac.yaml

# Apply network policies
kubectl apply -f deployment/kubernetes/networkpolicy.yaml

# Apply pod security policies
kubectl apply -f deployment/security/pod-security-policy.yaml
```

#### 5.2 Configure SSL/TLS

```bash
# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@wavefield-llm.example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Update ingress to use TLS
kubectl patch ingress wavefield-llm -n wavefield-llm --type=json \
  -p='[{"op": "add", "path": "/spec/tls", "value": [{"hosts": ["api.wavefield-llm.example.com"], "secretName": "wavefield-tls"}]}]'
```

### Phase 6: DNS and Load Balancer (15 minutes)

#### 6.1 Get Load Balancer IP

```bash
# Get external IP
kubectl get svc -n ingress-nginx ingress-nginx-controller

# Wait for external IP
kubectl wait --for=jsonpath='{.status.loadBalancer.ingress[0].ip}' \
  svc/ingress-nginx-controller -n ingress-nginx --timeout=300s
```

#### 6.2 Configure DNS

```bash
# Get the load balancer IP/hostname
export LB_IP=$(kubectl get svc -n ingress-nginx ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "Configure DNS A record:"
echo "api.wavefield-llm.example.com -> $LB_IP"

# Or use AWS Route53
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch file://dns-change.json
```

## Configuration

### Environment Variables

```yaml
# config/production.yaml
environment: production
debug: false

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 60

database:
  host: postgresql
  port: 5432
  name: wavefield
  pool_size: 20
  max_overflow: 10

redis:
  host: redis-master
  port: 6379
  db: 0
  max_connections: 50

model:
  path: /models/wavefield-v1
  device: cuda
  batch_size: 8
  max_length: 2048

logging:
  level: INFO
  format: json
  output: stdout
```

### Resource Limits

```yaml
# Recommended resource limits
resources:
  api:
    requests:
      cpu: 1000m
      memory: 2Gi
    limits:
      cpu: 2000m
      memory: 4Gi
  
  inference:
    requests:
      cpu: 4000m
      memory: 8Gi
      nvidia.com/gpu: 1
    limits:
      cpu: 8000m
      memory: 16Gi
      nvidia.com/gpu: 1
```

## Verification

### Health Checks

```bash
# Check API health
curl https://api.wavefield-llm.example.com/health

# Expected response:
# {"status":"healthy","version":"1.0.0","timestamp":"2024-01-15T10:30:00Z"}

# Check metrics endpoint
curl https://api.wavefield-llm.example.com/metrics

# Run smoke tests
./deployment/ci-cd/scripts/smoke-test.sh api.wavefield-llm.example.com
```

### Performance Testing

```bash
# Load test with Apache Bench
ab -n 1000 -c 10 https://api.wavefield-llm.example.com/v1/inference

# Load test with k6
k6 run deployment/tests/load-test.js
```

## Troubleshooting

### Common Issues

#### Issue 1: Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n wavefield-llm

# Check logs
kubectl logs <pod-name> -n wavefield-llm

# Common causes:
# - Image pull errors
# - Resource constraints
# - Configuration errors
```

#### Issue 2: Database Connection Failures

```bash
# Test database connectivity
kubectl run postgresql-client --rm --tty -i --restart='Never' \
  --namespace wavefield-llm \
  --image docker.io/bitnami/postgresql:15 \
  --command -- psql --host postgresql -U wavefield -d wavefield

# Check database logs
kubectl logs -n wavefield-llm postgresql-0
```

#### Issue 3: High Latency

```bash
# Check resource utilization
kubectl top pods -n wavefield-llm

# Check HPA status
kubectl get hpa -n wavefield-llm

# Scale manually if needed
kubectl scale deployment wavefield-api --replicas=10 -n wavefield-llm
```

## Best Practices

### 1. Security
- Always use secrets for sensitive data
- Enable RBAC and network policies
- Regular security audits
- Keep images updated

### 2. Monitoring
- Set up comprehensive alerting
- Monitor key metrics (latency, errors, saturation)
- Regular log review
- Capacity planning

### 3. Backup
- Automated daily backups
- Test restore procedures monthly
- Multi-region backup storage
- Document recovery procedures

### 4. Scaling
- Use HPA for automatic scaling
- Monitor resource utilization
- Plan for peak loads
- Test scaling procedures

### 5. Updates
- Use rolling updates
- Test in staging first
- Have rollback plan ready
- Monitor during deployment

## Next Steps

1. [Operations Guide](./OPERATIONS.md) - Day-2 operations
2. [Troubleshooting Guide](./TROUBLESHOOTING.md) - Common issues
3. [Scaling Guide](./SCALING.md) - Scaling strategies
4. [Security Guide](./SECURITY.md) - Security best practices

## Support

- **Documentation:** https://docs.wavefield-llm.example.com
- **Issues:** https://github.com/wavefield-llm/issues
- **Email:** support@wavefield-llm.example.com
- **Slack:** #wavefield-support

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0