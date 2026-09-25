# Wave Field LLM Deployment Automation

Complete automation scripts for deploying Wave Field LLM to production environments.

## Quick Start

### Local Development

```bash
./quick-start.sh
```

### Production Deployment

```bash
./deploy.sh \
  --environment production \
  --cloud aws \
  --region us-east-1 \
  --model wavefield-small \
  --replicas 3
```

## Available Scripts

### Core Deployment

| Script | Description | Usage |
|--------|-------------|-------|
| [`deploy.sh`](deploy.sh) | Master deployment orchestrator | `./deploy.sh --environment prod --cloud aws` |
| [`quick-start.sh`](quick-start.sh) | Local development setup | `./quick-start.sh` |
| [`deploy-model.sh`](deploy-model.sh) | Deploy specific model | `./deploy-model.sh --model wavefield-small` |
| [`update-model.sh`](update-model.sh) | Zero-downtime model updates | `./update-model.sh --model wavefield-small --version 1.1.0` |

### Cloud-Specific

| Script | Description | Usage |
|--------|-------------|-------|
| [`aws/deploy-aws.sh`](aws/deploy-aws.sh) | AWS infrastructure setup | `./aws/deploy-aws.sh --environment production` |
| [`gcp/deploy-gcp.sh`](gcp/deploy-gcp.sh) | GCP infrastructure setup | `./gcp/deploy-gcp.sh --project-id my-project` |
| [`azure/deploy-azure.sh`](azure/deploy-azure.sh) | Azure infrastructure setup | `./azure/deploy-azure.sh --environment production` |

### Operations

| Script | Description | Usage |
|--------|-------------|-------|
| [`setup-monitoring.sh`](setup-monitoring.sh) | Deploy monitoring stack | `./setup-monitoring.sh --namespace monitoring` |
| [`validate.sh`](validate.sh) | Validate deployment | `./validate.sh --environment production` |
| [`smoke-test.sh`](smoke-test.sh) | Run smoke tests | `./smoke-test.sh production` |
| [`rollback.sh`](rollback.sh) | Rollback deployment | `./rollback.sh wavefield-small production` |
| [`scale.sh`](scale.sh) | Scale deployment | `./scale.sh wavefield-small 5 production` |
| [`backup.sh`](backup.sh) | Create backup | `./backup.sh ./backups/prod` |
| [`cleanup.sh`](cleanup.sh) | Clean up resources | `./cleanup.sh production` |

### CI/CD

| Script | Description | Usage |
|--------|-------------|-------|
| [`ci-cd-setup.sh`](ci-cd-setup.sh) | Setup CI/CD pipelines | `./ci-cd-setup.sh` |

## Configuration Templates

Located in [`templates/`](templates/):

- [`production.env.template`](templates/production.env.template) - Production environment variables
- [`staging.env.template`](templates/staging.env.template) - Staging environment variables

## Deployment Workflows

### 1. Initial Production Deployment

```bash
# Step 1: Setup cloud infrastructure
./aws/deploy-aws.sh --environment production --region us-east-1

# Step 2: Deploy application
./deploy.sh \
  --environment production \
  --cloud aws \
  --region us-east-1 \
  --model wavefield-small \
  --replicas 5

# Step 3: Setup monitoring
./setup-monitoring.sh --environment production

# Step 4: Validate deployment
./validate.sh --environment production --cloud aws

# Step 5: Run smoke tests
./smoke-test.sh production
```

### 2. Model Update (Zero Downtime)

```bash
# Rolling update
./update-model.sh \
  --model wavefield-small \
  --version 1.1.0 \
  --namespace production

# Blue-green deployment
./update-model.sh \
  --model wavefield-small \
  --version 1.1.0 \
  --strategy blue-green

# Canary deployment
./update-model.sh \
  --model wavefield-small \
  --version 1.1.0 \
  --strategy canary \
  --canary-percent 20
```

### 3. Scaling Operations

```bash
# Manual scaling
./scale.sh wavefield-small 10 production

# Auto-scaling is configured via HPA
kubectl get hpa -n production
```

### 4. Disaster Recovery

```bash
# Create backup
./backup.sh ./backups/production-$(date +%Y%m%d)

# Rollback if needed
./rollback.sh wavefield-small production
```

## Script Options

### Common Options

All scripts support these common options:

- `--environment ENV` - Environment (production, staging, development)
- `--namespace NS` - Kubernetes namespace
- `--dry-run` - Show what would be done without executing
- `--verbose` - Enable verbose output
- `-h, --help` - Show help message

### Environment-Specific Options

#### AWS
- `--region REGION` - AWS region (e.g., us-east-1)
- `--eks-node-type TYPE` - EKS node instance type
- `--db-instance CLASS` - RDS instance class

#### GCP
- `--project-id ID` - GCP project ID
- `--region REGION` - GCP region (e.g., us-central1)
- `--machine-type TYPE` - GKE machine type

#### Azure
- `--location LOCATION` - Azure location (e.g., eastus)
- `--resource-group RG` - Resource group name
- `--vm-size SIZE` - AKS VM size

## Best Practices

### 1. Always Validate Before Deployment

```bash
./validate.sh --environment production --cloud aws
```

### 2. Use Dry Run for Testing

```bash
./deploy.sh --environment production --dry-run
```

### 3. Monitor Deployments

```bash
# Watch deployment progress
kubectl rollout status deployment/wavefield-small -n production

# View logs
kubectl logs -f -n production -l app=wavefield-llm
```

### 4. Regular Backups

```bash
# Daily backup
./backup.sh ./backups/daily-$(date +%Y%m%d)

# Upload to cloud storage
aws s3 sync ./backups/ s3://wavefield-backups/
```

### 5. Test Rollback Procedures

```bash
# Test rollback in staging first
./rollback.sh wavefield-small staging
```

## Troubleshooting

### Script Fails with Permission Error

```bash
chmod +x deployment/automation/*.sh
chmod +x deployment/automation/*/*.sh
```

### Kubernetes Connection Issues

```bash
# Verify kubectl configuration
kubectl cluster-info

# Update kubeconfig
aws eks update-kubeconfig --name wavefield-production --region us-east-1
```

### Docker Build Fails

```bash
# Check Docker daemon
docker info

# Clean up Docker
docker system prune -a
```

### Deployment Timeout

```bash
# Increase timeout
./deploy.sh --environment production --timeout 20m

# Check pod status
kubectl get pods -n production
kubectl describe pod <pod-name> -n production
```

## Monitoring and Logging

### Access Grafana

```bash
kubectl port-forward -n monitoring svc/grafana 3000:80
# Visit http://localhost:3000
```

### View Logs

```bash
# Application logs
kubectl logs -n production -l app=wavefield-llm --tail=100 -f

# All logs
kubectl logs -n production --all-containers=true --tail=100 -f
```

### Check Metrics

```bash
# Prometheus
kubectl port-forward -n monitoring svc/prometheus-server 9090:80

# Pod metrics
kubectl top pods -n production
```

## Security Considerations

1. **Secrets Management**
   - Never commit secrets to git
   - Use Kubernetes secrets or cloud secret managers
   - Rotate credentials regularly

2. **Network Security**
   - Use network policies
   - Enable TLS everywhere
   - Restrict ingress/egress

3. **Access Control**
   - Use RBAC
   - Principle of least privilege
   - Audit access logs

4. **Image Security**
   - Scan images for vulnerabilities
   - Use trusted base images
   - Keep images updated

## Support

- **Documentation**: See [`../guides/`](../guides/)
- **Issues**: Report issues on GitHub
- **Monitoring**: Check Grafana dashboards
- **Logs**: View in Loki or cloud logging service

## Contributing

When adding new automation scripts:

1. Follow existing script structure
2. Add comprehensive help text
3. Include error handling
4. Add to this README
5. Test in staging first

## License

See project LICENSE file.