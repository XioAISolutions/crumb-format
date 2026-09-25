# Wave Field LLM - AWS Deployment Guide

Complete production-grade deployment guide for Wave Field LLM on Amazon Web Services.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Infrastructure Components](#infrastructure-components)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Cost Optimization](#cost-optimization)

## Overview

This deployment provides enterprise-grade infrastructure for Wave Field LLM on AWS with:

- **High Availability**: Multi-AZ deployment across 3 availability zones
- **Scalability**: Auto-scaling for compute, storage, and database
- **Security**: Encryption at rest and in transit, VPC isolation, IAM roles
- **Monitoring**: CloudWatch dashboards, alarms, and log aggregation
- **Disaster Recovery**: Automated backups, cross-region replication
- **Cost Optimization**: Spot instances, auto-scaling, resource tagging

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CloudFront CDN                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                    Application Load Balancer                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│   EKS Cluster  │  │   SageMaker    │  │   Lambda       │
│   (Primary)    │  │   Endpoints    │  │   Functions    │
└───────┬────────┘  └────────────────┘  └────────────────┘
        │
        ├─────────────┬─────────────┬─────────────┐
        │             │             │             │
┌───────▼────┐ ┌─────▼─────┐ ┌─────▼─────┐ ┌────▼─────┐
│  General   │ │    GPU    │ │   Spot    │ │   ECS    │
│   Nodes    │ │   Nodes   │ │   Nodes   │ │ Fargate  │
└────────────┘ └───────────┘ └───────────┘ └──────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼────────┐ ┌─▼──────────┐ ┌▼────────────┐
│  RDS Postgres  │ │ ElastiCache│ │  S3 Buckets │
│   Multi-AZ     │ │   Redis    │ │   (Models,  │
│  + Replica     │ │  Cluster   │ │  Data, Logs)│
└────────────────┘ └────────────┘ └─────────────┘
```

## Prerequisites

### Required Tools

```bash
# AWS CLI
aws --version  # >= 2.13.0

# Terraform
terraform --version  # >= 1.5.0

# kubectl
kubectl version --client  # >= 1.28.0

# eksctl
eksctl version  # >= 0.160.0

# Helm
helm version  # >= 3.12.0
```

### AWS Account Setup

1. **IAM Permissions**: Administrator access or equivalent
2. **Service Quotas**: Verify limits for EC2, EKS, RDS
3. **Cost Alerts**: Set up billing alerts
4. **Support Plan**: Business or Enterprise recommended

### Domain & SSL

- Registered domain name
- Route53 hosted zone (optional, can be created by Terraform)
- ACM certificate (will be created automatically)

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/wavefield-llm.git
cd wavefield-llm/deployment/cloud/aws
```

### 2. Configure Backend

```bash
# Create S3 bucket for Terraform state
aws s3api create-bucket \
  --bucket wavefield-llm-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket wavefield-llm-terraform-state \
  --versioning-configuration Status=Enabled

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name wavefield-llm-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### 3. Configure Variables

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

**Critical Variables to Set**:
```hcl
project_name = "wavefield-llm"
environment  = "prod"
aws_region   = "us-east-1"

# Security - CHANGE THESE!
db_password  = "STRONG_PASSWORD_HERE"
redis_auth_token = "STRONG_TOKEN_HERE"

# Domain
domain_name = "your-domain.com"

# Alerting
alert_emails = ["ops@your-company.com"]
```

### 4. Deploy Infrastructure

```bash
# Initialize Terraform
terraform init -backend-config=backend.hcl

# Review plan
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan
```

**Deployment Time**: ~30-45 minutes

### 5. Configure kubectl

```bash
# Update kubeconfig
aws eks update-kubeconfig \
  --region us-east-1 \
  --name wavefield-llm-cluster

# Verify access
kubectl get nodes
```

### 6. Install EKS Addons

```bash
cd ../
chmod +x eks-addons.sh
./eks-addons.sh
```

### 7. Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -f ../kubernetes/

# Verify deployment
kubectl get pods -n wavefield-llm
```

## Infrastructure Components

### Networking

- **VPC**: 10.0.0.0/16 CIDR
- **Subnets**: 
  - 3 Public subnets (for ALB, NAT)
  - 3 Private subnets (for EKS, ECS)
  - 3 Database subnets (for RDS, ElastiCache)
- **NAT Gateways**: 3 (one per AZ for HA)
- **VPC Endpoints**: S3, ECR, CloudWatch Logs

### Compute

#### EKS Cluster
- **Version**: 1.28
- **Node Groups**:
  - General: 2-10 t3.xlarge instances
  - GPU: 0-5 g5.xlarge instances
  - Spot: 0-20 mixed instances
- **Addons**: VPC CNI, CoreDNS, kube-proxy, EBS CSI

#### SageMaker
- **Endpoints**: Real-time, batch, serverless
- **Instance Types**: ml.g5.xlarge, ml.g5.2xlarge
- **Auto-scaling**: 1-5 instances

#### Lambda
- **Memory**: 10GB
- **Timeout**: 15 minutes
- **Concurrency**: 100
- **Storage**: EFS for model files

#### ECS Fargate
- **CPU**: 4 vCPU
- **Memory**: 16GB
- **Tasks**: 3-10 auto-scaled

### Database

#### RDS PostgreSQL
- **Version**: 15.4
- **Instance**: db.r6g.xlarge
- **Storage**: 100GB-1TB (auto-scaling)
- **Multi-AZ**: Enabled
- **Read Replica**: Optional
- **Backups**: 30 days retention

#### ElastiCache Redis
- **Version**: 7.0
- **Node Type**: cache.r6g.large
- **Nodes**: 3 (cluster mode)
- **Multi-AZ**: Enabled
- **Encryption**: At rest and in transit

### Storage

#### S3 Buckets
- **Models**: Versioned, lifecycle policies
- **Data**: Training and inference data
- **Logs**: Application and access logs

### Load Balancing

- **ALB**: Application Load Balancer
- **CloudFront**: CDN with custom domain
- **Route53**: DNS management

### Security

- **KMS Keys**: Separate keys for EKS, RDS, S3, ElastiCache
- **Security Groups**: Least privilege rules
- **IAM Roles**: IRSA for Kubernetes service accounts
- **VPC Flow Logs**: Network traffic monitoring

## Deployment Options

### Option 1: EKS (Recommended)

**Best for**: Production workloads, high availability, auto-scaling

```bash
# Already deployed with Quick Start
kubectl get deployments -n wavefield-llm
```

### Option 2: SageMaker

**Best for**: Managed ML inference, A/B testing, model monitoring

```bash
cd sagemaker
pip install -r requirements.txt

python deploy.py \
  --model-path /path/to/model \
  --role-arn arn:aws:iam::ACCOUNT:role/SageMakerRole \
  --instance-type ml.g5.xlarge \
  --enable-autoscaling
```

### Option 3: Lambda

**Best for**: Serverless, event-driven, cost-effective for low traffic

```bash
cd lambda
sam build
sam deploy --guided
```

### Option 4: ECS Fargate

**Best for**: Containerized apps without Kubernetes overhead

```bash
cd ecs
aws ecs create-cluster --cluster-name wavefield-llm

aws ecs register-task-definition \
  --cli-input-json file://task-definition.json

aws ecs create-service \
  --cli-input-json file://service.json
```

## Configuration

### Environment Variables

```bash
# Application
MODEL_PATH=/models/wavefield-llm
MAX_WORKERS=4
LOG_LEVEL=INFO

# Database
DB_HOST=<rds-endpoint>
DB_PORT=5432
DB_NAME=wavefieldllm
DB_USER=dbadmin
DB_PASSWORD=<from-secrets-manager>

# Redis
REDIS_HOST=<elasticache-endpoint>
REDIS_PORT=6379
REDIS_AUTH_TOKEN=<from-secrets-manager>

# AWS
AWS_REGION=us-east-1
S3_BUCKET=wavefield-llm-models-ACCOUNT_ID
```

### Secrets Management

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
  --name wavefield-llm/db-password \
  --secret-string "YOUR_DB_PASSWORD"

aws secretsmanager create-secret \
  --name wavefield-llm/redis-token \
  --secret-string "YOUR_REDIS_TOKEN"
```

### Auto-Scaling Configuration

```hcl
# EKS Node Groups
general_min_size = 2
general_max_size = 10
general_desired_size = 3

# SageMaker Endpoints
min_capacity = 1
max_capacity = 5
target_invocations_per_instance = 1000
```

## Monitoring & Maintenance

### CloudWatch Dashboards

Access dashboards:
```bash
# Get dashboard URL
terraform output cloudwatch_dashboard_url
```

**Key Metrics**:
- EKS: Node count, pod count, CPU/memory
- RDS: Connections, CPU, storage, latency
- Redis: CPU, memory, evictions
- ALB: Request count, latency, errors

### Alarms

**Critical Alarms** (immediate notification):
- EKS cluster failed nodes
- RDS CPU > 80%
- Redis memory > 85%
- ALB 5xx errors > 10/5min

**Warning Alarms** (1-hour response):
- RDS connections > 1000
- Redis evictions detected
- ALB response time > 1s

### Logs

```bash
# EKS cluster logs
aws logs tail /aws/eks/wavefield-llm-cluster/cluster --follow

# Application logs
kubectl logs -f deployment/wavefield-llm -n wavefield-llm

# RDS slow queries
aws logs tail /aws/rds/instance/wavefield-llm-postgres/postgresql --follow
```

### Backups

**Automated**:
- RDS: Daily snapshots, 30-day retention
- Redis: Daily snapshots, 7-day retention
- S3: Versioning enabled

**Manual Backup**:
```bash
# RDS snapshot
aws rds create-db-snapshot \
  --db-instance-identifier wavefield-llm-postgres \
  --db-snapshot-identifier manual-backup-$(date +%Y%m%d)

# Redis snapshot
aws elasticache create-snapshot \
  --replication-group-id wavefield-llm-redis \
  --snapshot-name manual-backup-$(date +%Y%m%d)
```

## Troubleshooting

### Common Issues

#### 1. EKS Nodes Not Joining Cluster

```bash
# Check node status
kubectl get nodes

# Check node logs
aws ec2 describe-instances --filters "Name=tag:eks:cluster-name,Values=wavefield-llm-cluster"

# Verify IAM role
aws iam get-role --role-name wavefield-llm-eks-node-group-role
```

#### 2. RDS Connection Issues

```bash
# Test connectivity
nc -zv <rds-endpoint> 5432

# Check security group
aws ec2 describe-security-groups --group-ids <sg-id>

# Verify credentials
aws secretsmanager get-secret-value --secret-id wavefield-llm/db-password
```

#### 3. High Costs

```bash
# Check resource usage
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Identify unused resources
aws ec2 describe-volumes --filters "Name=status,Values=available"
aws ec2 describe-addresses --filters "Name=domain,Values=vpc"
```

### Debug Commands

```bash
# EKS cluster info
kubectl cluster-info
kubectl get events --all-namespaces

# Pod debugging
kubectl describe pod <pod-name> -n wavefield-llm
kubectl logs <pod-name> -n wavefield-llm --previous

# Network debugging
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- /bin/bash
```

## Cost Optimization

### Recommendations

1. **Use Spot Instances**: 70% savings for non-critical workloads
2. **Right-size Resources**: Monitor and adjust instance types
3. **Enable Auto-scaling**: Scale down during off-hours
4. **Use Savings Plans**: 1-year commitment for 30-40% savings
5. **Optimize Storage**: Use lifecycle policies, compression
6. **Review Unused Resources**: Delete unused EBS volumes, EIPs

### Cost Monitoring

```bash
# Set up budget alerts
aws budgets create-budget \
  --account-id ACCOUNT_ID \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json

# Enable Cost Explorer
aws ce get-cost-and-usage --help
```

### Estimated Monthly Costs

| Component | Cost |
|-----------|------|
| EKS Cluster | $73 |
| EC2 Nodes (General) | $300 |
| EC2 Nodes (GPU) | $500 |
| RDS PostgreSQL | $400 |
| ElastiCache Redis | $300 |
| S3 Storage | $25 |
| Data Transfer | $200 |
| ALB | $25 |
| NAT Gateways | $100 |
| CloudFront | $100 |
| **Total** | **$3,000-$8,000** |

## Support

### Documentation
- [AWS EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Getting Help
- GitHub Issues: [Create an issue](https://github.com/your-org/wavefield-llm/issues)
- Slack: #wavefield-llm-support
- Email: ml-platform@your-company.com

### Emergency Contacts
- On-call Engineer: +1-XXX-XXX-XXXX
- Team Lead: +1-XXX-XXX-XXXX
- AWS Support: [AWS Console](https://console.aws.amazon.com/support/)

---

**Last Updated**: 2026-05-28  
**Version**: 1.0  
**Maintained By**: ML Infrastructure Team