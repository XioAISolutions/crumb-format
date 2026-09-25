# Wave Field LLM Kubernetes Deployment Guide

Guide for deploying Wave Field LLM to Kubernetes clusters.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.0+
- Container registry access

## Quick Deployment

```bash
cd deployment/automation

# Deploy model
./deploy-model.sh \
  --model wavefield-small \
  --namespace production \
  --replicas 3
```

## Manual Deployment

### Step 1: Create Namespace

```bash
kubectl create namespace wavefield-production
```

### Step 2: Create Secrets

```bash
kubectl create secret generic wavefield-secrets \
  -n wavefield-production \
  --from-literal=db-password=YOUR_PASSWORD \
  --from-literal=api-key=YOUR_API_KEY
```

### Step 3: Apply ConfigMap

```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
```

### Step 4: Deploy Application

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
```

### Step 5: Configure Ingress

```bash
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### Step 6: Setup Auto-Scaling

```bash
kubectl apply -f deployment/kubernetes/hpa.yaml
```

## Using Helm

```bash
helm install wavefield-llm \
  deployment/kubernetes/helm/wavefield-llm \
  --namespace wavefield-production \
  --set image.tag=latest \
  --set replicaCount=3
```

## Monitoring

```bash
# Deploy monitoring stack
cd deployment/automation
./setup-monitoring.sh --namespace monitoring
```

## Scaling

### Manual Scaling

```bash
kubectl scale deployment/wavefield-small \
  -n wavefield-production \
  --replicas=5
```

### Auto-Scaling

HPA automatically scales based on CPU/memory:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wavefield-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wavefield-small
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Updates

### Rolling Update

```bash
kubectl set image deployment/wavefield-small \
  -n wavefield-production \
  wavefield-llm=wavefield-llm:v1.1.0
```

### Blue-Green Deployment

```bash
./update-model.sh \
  --model wavefield-small \
  --version 1.1.0 \
  --strategy blue-green
```

### Canary Deployment

```bash
./update-model.sh \
  --model wavefield-small \
  --version 1.1.0 \
  --strategy canary \
  --canary-percent 20
```

## Troubleshooting

### View Logs

```bash
kubectl logs -n wavefield-production -l app=wavefield-llm --tail=100 -f
```

### Debug Pod

```bash
kubectl exec -it -n wavefield-production <pod-name> -- /bin/bash
```

### Check Events

```bash
kubectl get events -n wavefield-production --sort-by='.lastTimestamp'
```

## Best Practices

1. **Resource Limits**: Always set resource requests and limits
2. **Health Checks**: Configure liveness and readiness probes
3. **Security**: Use network policies and RBAC
4. **Monitoring**: Enable metrics and logging
5. **Backups**: Regular backup of configurations

## See Also

- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - Complete production guide
- [LOCAL_DEPLOYMENT.md](LOCAL_DEPLOYMENT.md) - Local development setup