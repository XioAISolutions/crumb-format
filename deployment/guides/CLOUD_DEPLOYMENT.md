# Wave Field LLM Cloud Deployment Guide

Platform-specific deployment guides for AWS, GCP, and Azure.

## AWS Deployment

### Quick Start

```bash
cd deployment/automation/aws
./deploy-aws.sh --environment production --region us-east-1
```

### Manual Setup

#### 1. Create EKS Cluster

```bash
eksctl create cluster \
  --name wavefield-production \
  --region us-east-1 \
  --nodes 5 \
  --node-type m5.xlarge \
  --managed
```

#### 2. Create RDS Database

```bash
aws rds create-db-instance \
  --db-instance-identifier wavefield-prod \
  --db-instance-class db.r5.large \
  --engine postgres \
  --master-username wavefield \
  --master-user-password YOUR_PASSWORD \
  --allocated-storage 100
```

#### 3. Create ElastiCache Redis

```bash
aws elasticache create-cache-cluster \
  --cache-cluster-id wavefield-prod \
  --cache-node-type cache.r5.large \
  --engine redis \
  --num-cache-nodes 1
```

#### 4. Create S3 Buckets

```bash
aws s3 mb s3://wavefield-prod-models
aws s3 mb s3://wavefield-prod-logs
```

#### 5. Create ECR Repository

```bash
aws ecr create-repository --repository-name wavefield-llm
```

### AWS-Specific Features

- **CloudFront CDN**: For global content delivery
- **Route53**: For DNS management
- **CloudWatch**: For monitoring and logging
- **Secrets Manager**: For secure credential storage
- **WAF**: For web application firewall

## GCP Deployment

### Quick Start

```bash
cd deployment/automation/gcp
./deploy-gcp.sh \
  --project-id my-project \
  --environment production \
  --region us-central1
```

### Manual Setup

#### 1. Create GKE Cluster

```bash
gcloud container clusters create wavefield-production \
  --region us-central1 \
  --num-nodes 5 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10
```

#### 2. Create Cloud SQL

```bash
gcloud sql instances create wavefield-prod \
  --database-version=POSTGRES_15 \
  --tier=db-n1-standard-2 \
  --region=us-central1
```

#### 3. Create Memorystore Redis

```bash
gcloud redis instances create wavefield-prod \
  --size=5 \
  --region=us-central1 \
  --tier=standard
```

#### 4. Create Cloud Storage Buckets

```bash
gsutil mb -p my-project -c STANDARD -l us-central1 gs://wavefield-prod-models
gsutil mb -p my-project -c STANDARD -l us-central1 gs://wavefield-prod-logs
```

#### 5. Create Artifact Registry

```bash
gcloud artifacts repositories create wavefield-llm \
  --repository-format=docker \
  --location=us-central1
```

### GCP-Specific Features

- **Cloud CDN**: For content delivery
- **Cloud DNS**: For DNS management
- **Cloud Monitoring**: For observability
- **Secret Manager**: For secrets
- **Cloud Armor**: For DDoS protection

## Azure Deployment

### Quick Start

```bash
cd deployment/automation/azure
./deploy-azure.sh --environment production --location eastus
```

### Manual Setup

#### 1. Create AKS Cluster

```bash
az aks create \
  --resource-group wavefield-rg \
  --name wavefield-production \
  --node-count 5 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 10
```

#### 2. Create Azure Database for PostgreSQL

```bash
az postgres server create \
  --resource-group wavefield-rg \
  --name wavefield-prod \
  --location eastus \
  --admin-user wavefield \
  --admin-password YOUR_PASSWORD \
  --sku-name GP_Gen5_2
```

#### 3. Create Azure Cache for Redis

```bash
az redis create \
  --resource-group wavefield-rg \
  --name wavefield-prod \
  --location eastus \
  --sku Standard \
  --vm-size c1
```

#### 4. Create Storage Account

```bash
az storage account create \
  --resource-group wavefield-rg \
  --name wavefieldprodstorage \
  --location eastus \
  --sku Standard_LRS
```

#### 5. Create Container Registry

```bash
az acr create \
  --resource-group wavefield-rg \
  --name wavefieldprodacr \
  --sku Standard
```

### Azure-Specific Features

- **Azure CDN**: For content delivery
- **Azure DNS**: For DNS management
- **Azure Monitor**: For monitoring
- **Key Vault**: For secrets management
- **Azure Firewall**: For network security

## Multi-Cloud Considerations

### Load Balancing

- Use cloud-native load balancers
- Configure health checks
- Enable SSL/TLS termination

### DNS and SSL

```bash
# Example: Let's Encrypt with cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

### Monitoring

All clouds support:
- Prometheus for metrics
- Grafana for visualization
- Loki for logs
- Jaeger for tracing

### Cost Optimization

1. **Use Reserved/Committed Instances**: 30-70% savings
2. **Auto-scaling**: Scale down during off-hours
3. **Spot/Preemptible Instances**: For non-critical workloads
4. **Storage Lifecycle**: Archive old data
5. **Right-sizing**: Monitor and adjust resources

## Comparison Matrix

| Feature | AWS | GCP | Azure |
|---------|-----|-----|-------|
| Kubernetes | EKS | GKE | AKS |
| Database | RDS | Cloud SQL | Azure Database |
| Cache | ElastiCache | Memorystore | Azure Cache |
| Storage | S3 | Cloud Storage | Blob Storage |
| Registry | ECR | Artifact Registry | ACR |
| CDN | CloudFront | Cloud CDN | Azure CDN |
| DNS | Route53 | Cloud DNS | Azure DNS |

## Best Practices

1. **Infrastructure as Code**: Use Terraform for all clouds
2. **Multi-Region**: Deploy to multiple regions for HA
3. **Backup Strategy**: Regular backups across regions
4. **Security**: Enable encryption, use private networks
5. **Monitoring**: Centralized monitoring across clouds
6. **Cost Management**: Set budgets and alerts

## See Also

- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - General production guide
- [KUBERNETES_DEPLOYMENT.md](KUBERNETES_DEPLOYMENT.md) - Kubernetes specifics