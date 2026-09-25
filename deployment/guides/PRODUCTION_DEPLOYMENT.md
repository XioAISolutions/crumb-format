# Wave Field LLM Production Deployment Guide

Complete guide for deploying Wave Field LLM to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Infrastructure Setup](#infrastructure-setup)
4. [Application Deployment](#application-deployment)
5. [Post-Deployment Verification](#post-deployment-verification)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

- Docker (20.10+)
- Kubernetes (1.24+)
- kubectl
- Helm (3.0+)
- Terraform (1.0+)
- Cloud CLI (aws-cli, gcloud, or az)

### Required Access

- Cloud provider account with admin access
- Container registry access
- DNS management access
- SSL certificate management

## Pre-Deployment Checklist

### Infrastructure Requirements

- [ ] Cloud account configured
- [ ] VPC/Network configured
- [ ] Kubernetes cluster ready (EKS/GKE/AKS)
- [ ] Database provisioned (PostgreSQL)
- [ ] Cache provisioned (Redis)
- [ ] Object storage configured (S3/GCS/Blob)
- [ ] Container registry setup
- [ ] Load balancer configured
- [ ] DNS records ready
- [ ] SSL certificates obtained

### Security Requirements

- [ ] Secrets management configured
- [ ] RBAC policies defined
- [ ] Network policies configured
- [ ] Security groups/firewall rules set
- [ ] Encryption at rest enabled
- [ ] Encryption in transit enabled
- [ ] Audit logging enabled

### Application Requirements

- [ ] Model files downloaded
- [ ] Configuration files prepared
- [ ] Environment variables set
- [ ] Resource limits defined
- [ ] Scaling policies configured

## Infrastructure Setup

### Option 1: One-Click Cloud Deployment

#### AWS Deployment

```bash
cd deployment/automation/aws
./deploy-aws.sh \
  --environment production \
  --region us-east-1 \
  --eks-node-count 5
```

#### GCP Deployment

```bash
cd deployment/automation/gcp
./deploy-gcp.sh \
  --project-id my-project \
  --environment production \
  --region us-central1
```

#### Azure Deployment

```bash
cd deployment/automation/azure
./deploy-azure.sh \
  --environment production \
  --location eastus
```

### Option 2: Manual Infrastructure Setup

#### 1. Create VPC/Network

```bash
# AWS
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# GCP
gcloud compute networks create wavefield-vpc --subnet-mode=custom

# Azure
az network vnet create --name wavefield-vnet --address-prefix 10.0.0.0/16
```

#### 2. Create Kubernetes Cluster

```bash
# AWS EKS
eksctl create cluster \
  --name wavefield-production \
  --region us-east-1 \
  --nodes 5 \
  --node-type m5.xlarge

# GCP GKE
gcloud container clusters create wavefield-production \
  --region us-central1 \
  --num-nodes 5 \
  --machine-type n1-standard-4

# Azure AKS
az aks create \
  --resource-group wavefield-rg \
  --name wavefield-production \
  --node-count 5 \
  --node-vm-size Standard_D4s_v3
```

#### 3. Configure kubectl

```bash
# AWS
aws eks update-kubeconfig --name wavefield-production --region us-east-1

# GCP
gcloud container clusters get-credentials wavefield-production --region us-central1

# Azure
az aks get-credentials --resource-group wavefield-rg --name wavefield-production
```

## Application Deployment

### Step 1: Prepare Configuration

```bash
# Copy and edit environment template
cp deployment/automation/templates/production.env.template .env.production

# Edit configuration
vim .env.production
```

### Step 2: Deploy Using Master Script

```bash
cd deployment/automation

./deploy.sh \
  --environment production \
  --cloud aws \
  --region us-east-1 \
  --model wavefield-small \
  --replicas 5
```

### Step 3: Deploy Specific Model

```bash
./deploy-model.sh \
  --model wavefield-medium \
  --version 1.0.0 \
  --namespace production \
  --replicas 3 \
  --gpu
```

### Step 4: Setup Monitoring

```bash
./setup-monitoring.sh \
  --environment production \
  --namespace monitoring
```

## Post-Deployment Verification

### 1. Run Validation

```bash
./validate.sh \
  --environment production \
  --cloud aws \
  --region us-east-1
```

### 2. Run Smoke Tests

```bash
./smoke-test.sh production
```

### 3. Verify Deployments

```bash
kubectl get deployments -n production
kubectl get pods -n production
kubectl get services -n production
```

### 4. Check Health Endpoints

```bash
# Port forward to service
kubectl port-forward -n production svc/wavefield-small 8000:8000

# Test health endpoint
curl http://localhost:8000/health

# Test inference
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 50}'
```

### 5. Verify Monitoring

```bash
# Access Grafana
kubectl port-forward -n monitoring svc/grafana 3000:80

# Visit http://localhost:3000
# Login with admin credentials
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Application Metrics**
   - Request rate
   - Response time (p50, p95, p99)
   - Error rate
   - Token throughput
   - Model inference latency

2. **Infrastructure Metrics**
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network traffic
   - Pod restarts

3. **Business Metrics**
   - Active users
   - API calls per minute
   - Cost per request
   - Model accuracy

### Setting Up Alerts

```bash
# Configure Alertmanager
kubectl apply -f deployment/monitoring/alertmanager.yml

# Verify alerts
kubectl port-forward -n monitoring svc/alertmanager 9093:9093
```

### Grafana Dashboards

Pre-configured dashboards available:
- System Overview
- Model Performance
- API Metrics
- Cost Analysis

## Scaling

### Manual Scaling

```bash
# Scale deployment
./scale.sh wavefield-small 10 production

# Or use kubectl
kubectl scale deployment/wavefield-small -n production --replicas=10
```

### Auto-Scaling

```bash
# Apply HPA
kubectl apply -f deployment/kubernetes/hpa.yaml

# Verify HPA
kubectl get hpa -n production
```

## Backup and Recovery

### Create Backup

```bash
./backup.sh ./backups/production-$(date +%Y%m%d)
```

### Restore from Backup

```bash
kubectl apply -f ./backups/production-20260529/
```

## Rollback

### Automatic Rollback

```bash
./rollback.sh wavefield-small production
```

### Manual Rollback

```bash
# View rollout history
kubectl rollout history deployment/wavefield-small -n production

# Rollback to previous version
kubectl rollout undo deployment/wavefield-small -n production

# Rollback to specific revision
kubectl rollout undo deployment/wavefield-small -n production --to-revision=2
```

## Troubleshooting

### Common Issues

#### Pods Not Starting

```bash
# Check pod status
kubectl get pods -n production

# View pod logs
kubectl logs -n production <pod-name>

# Describe pod for events
kubectl describe pod -n production <pod-name>
```

#### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints -n production

# Check ingress
kubectl get ingress -n production

# Test service internally
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://wavefield-small.production.svc.cluster.local:8000/health
```

#### High Memory Usage

```bash
# Check resource usage
kubectl top pods -n production

# Adjust resource limits
kubectl set resources deployment/wavefield-small -n production \
  --limits=memory=16Gi \
  --requests=memory=8Gi
```

#### Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm psql --image=postgres:15 --restart=Never -- \
  psql -h <db-host> -U wavefield -d wavefield

# Check secrets
kubectl get secret -n production
```

## Maintenance

### Regular Tasks

1. **Daily**
   - Monitor dashboards
   - Check error logs
   - Review alerts

2. **Weekly**
   - Review resource usage
   - Check for updates
   - Backup verification

3. **Monthly**
   - Security patches
   - Cost optimization
   - Performance tuning
   - Disaster recovery drill

### Cleanup

```bash
# Remove failed pods
./cleanup.sh production

# Remove old deployments
kubectl delete deployment <old-deployment> -n production
```

## Security Best Practices

1. **Network Security**
   - Use network policies
   - Enable TLS everywhere
   - Restrict ingress/egress

2. **Access Control**
   - Use RBAC
   - Rotate credentials regularly
   - Use service accounts

3. **Data Security**
   - Encrypt at rest
   - Encrypt in transit
   - Secure secrets management

4. **Monitoring**
   - Enable audit logs
   - Monitor for anomalies
   - Set up security alerts

## Performance Optimization

1. **Resource Tuning**
   - Right-size pods
   - Use node affinity
   - Configure resource limits

2. **Caching**
   - Enable Redis caching
   - Use CDN for static assets
   - Implement response caching

3. **Database**
   - Connection pooling
   - Query optimization
   - Read replicas

## Cost Optimization

1. **Right-Sizing**
   - Monitor actual usage
   - Use spot/preemptible instances
   - Scale down during off-hours

2. **Storage**
   - Use appropriate storage classes
   - Implement lifecycle policies
   - Clean up old data

3. **Networking**
   - Use private endpoints
   - Optimize data transfer
   - Enable compression

## Support and Resources

- Documentation: `docs/`
- Issues: GitHub Issues
- Monitoring: Grafana dashboards
- Logs: Loki/CloudWatch/Stackdriver

## Appendix

### Useful Commands

```bash
# View all resources
kubectl get all -n production

# Stream logs
kubectl logs -f -n production -l app=wavefield-llm

# Execute command in pod
kubectl exec -it -n production <pod-name> -- /bin/bash

# Port forward
kubectl port-forward -n production <pod-name> 8000:8000

# Copy files
kubectl cp -n production <pod-name>:/path/to/file ./local-file
```

### Configuration Reference

See [`deployment/automation/templates/`](../automation/templates/) for configuration templates.

### API Reference

See [`api/openapi.yaml`](../../api/openapi.yaml) for API documentation.