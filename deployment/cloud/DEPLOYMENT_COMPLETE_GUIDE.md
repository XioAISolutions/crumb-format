# Wave Field LLM - Complete Cloud Deployment Guide

This document provides a comprehensive overview of the production-grade cloud deployment infrastructure for Wave Field LLM across AWS, GCP, and Azure.

## 📋 Table of Contents

1. [AWS Deployment](#aws-deployment)
2. [GCP Deployment](#gcp-deployment)
3. [Azure Deployment](#azure-deployment)
4. [Cost Estimates](#cost-estimates)
5. [Security & Compliance](#security--compliance)
6. [Disaster Recovery](#disaster-recovery)
7. [Monitoring & Alerting](#monitoring--alerting)

---

## AWS Deployment

### ✅ Completed Components

#### Terraform Infrastructure (10 files)
- **main.tf**: Complete multi-AZ infrastructure with VPC, EKS, RDS, ElastiCache, S3, ALB, CloudFront, Route53
- **variables.tf**: 50+ configurable parameters with validation
- **outputs.tf**: Comprehensive outputs for all resources
- **backend.tf**: S3 backend with DynamoDB locking
- **versions.tf**: Provider version pinning
- **data.tf**: Data sources and IAM policies
- **locals.tf**: Computed values and tagging strategy
- **security.tf**: KMS keys, security groups, IAM roles, VPC flow logs
- **monitoring.tf**: CloudWatch dashboards, alarms, log metric filters
- **autoscaling.tf**: Auto-scaling policies and configurations

#### EKS Configuration
- **eks-config.yaml**: Complete eksctl configuration (already exists)
- **eks-addons.sh**: Installation script for:
  - AWS Load Balancer Controller
  - External DNS
  - Cluster Autoscaler
  - Metrics Server
  - Cert Manager
  - External Secrets Operator
  - NVIDIA Device Plugin
  - Optional: Kubernetes Dashboard, Prometheus & Grafana

#### SageMaker Deployment
- **deploy.py**: Complete deployment script with:
  - Standard endpoints
  - Multi-model endpoints
  - Async inference endpoints
  - Serverless inference endpoints
  - Auto-scaling configuration
  - A/B testing support
- **inference.py**: Custom inference handler
- **requirements.txt**: Dependencies

### 🔄 Remaining AWS Components

#### Lambda Serverless (`lambda/`)

**handler.py**:
```python
import json
import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variables for model reuse
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        # Load from S3 or EFS
        model = AutoModelForCausalLM.from_pretrained("/mnt/efs/models/wavefield-llm")
        tokenizer = AutoTokenizer.from_pretrained("/mnt/efs/models/wavefield-llm")
    return model, tokenizer

def lambda_handler(event, context):
    model, tokenizer = load_model()
    
    # Parse input
    body = json.loads(event['body'])
    prompt = body.get('prompt', '')
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.decode(outputs[0])
    
    return {
        'statusCode': 200,
        'body': json.dumps({'generated_text': text})
    }
```

**template.yaml** (SAM):
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  WaveFieldLLMFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: wavefield-llm-inference
      Runtime: python3.11
      Handler: handler.lambda_handler
      MemorySize: 10240
      Timeout: 900
      EphemeralStorage:
        Size: 10240
      FileSystemConfigs:
        - Arn: !GetAtt EFSAccessPoint.Arn
          LocalMountPath: /mnt/efs
      VpcConfig:
        SecurityGroupIds:
          - !Ref LambdaSecurityGroup
        SubnetIds:
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2
      Environment:
        Variables:
          MODEL_PATH: /mnt/efs/models/wavefield-llm
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /generate
            Method: post
```

#### ECS Fargate (`ecs/`)

**task-definition.json**:
```json
{
  "family": "wavefield-llm",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "wavefield-llm",
      "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/wavefield-llm:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "MODEL_PATH", "value": "/models"},
        {"name": "MAX_WORKERS", "value": "4"}
      ],
      "secrets": [
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:db-password"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "efs-models",
          "containerPath": "/models"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/wavefield-llm",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ],
  "volumes": [
    {
      "name": "efs-models",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-XXXXXXXX",
        "transitEncryption": "ENABLED"
      }
    }
  ]
}
```

**service.json**:
```json
{
  "serviceName": "wavefield-llm-service",
  "taskDefinition": "wavefield-llm",
  "desiredCount": 3,
  "launchType": "FARGATE",
  "networkConfiguration": {
    "awsvpcConfiguration": {
      "subnets": ["subnet-xxx", "subnet-yyy"],
      "securityGroups": ["sg-xxx"],
      "assignPublicIp": "DISABLED"
    }
  },
  "loadBalancers": [
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:...",
      "containerName": "wavefield-llm",
      "containerPort": 8000
    }
  ],
  "deploymentConfiguration": {
    "maximumPercent": 200,
    "minimumHealthyPercent": 100,
    "deploymentCircuitBreaker": {
      "enable": true,
      "rollback": true
    }
  }
}
```

---

## GCP Deployment

### Infrastructure Components

#### Terraform (`gcp/terraform/`)

**main.tf** (Key Resources):
```hcl
# VPC Network
resource "google_compute_network" "main" {
  name                    = "${var.project_name}-vpc"
  auto_create_subnetworks = false
}

# GKE Cluster
resource "google_container_cluster" "main" {
  name     = "${var.project_name}-gke"
  location = var.region
  
  node_config {
    machine_type = "n1-standard-4"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }
}

# Cloud SQL
resource "google_sql_database_instance" "main" {
  name             = "${var.project_name}-postgres"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier              = "db-custom-4-16384"
    availability_type = "REGIONAL"
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.id
    }
  }
}

# Memorystore Redis
resource "google_redis_instance" "main" {
  name           = "${var.project_name}-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 5
  region         = var.region
  
  redis_version     = "REDIS_7_0"
  auth_enabled      = true
  transit_encryption_mode = "SERVER_AUTHENTICATION"
}

# Cloud Storage
resource "google_storage_bucket" "models" {
  name          = "${var.project_name}-models"
  location      = var.region
  force_destroy = false
  
  versioning {
    enabled = true
  }
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.storage.id
  }
}
```

#### Vertex AI (`gcp/vertex-ai/`)

**deploy.py**:
```python
from google.cloud import aiplatform

def deploy_model(
    project_id: str,
    location: str,
    model_path: str,
    endpoint_name: str
):
    aiplatform.init(project=project_id, location=location)
    
    # Upload model
    model = aiplatform.Model.upload(
        display_name="wavefield-llm",
        artifact_uri=model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest"
    )
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
    
    # Deploy model
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=5
    )
    
    return endpoint
```

#### Cloud Run (`gcp/cloud-run/`)

**service.yaml**:
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: wavefield-llm
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/PROJECT_ID/wavefield-llm:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            cpu: "4"
            memory: "16Gi"
        env:
        - name: MODEL_PATH
          value: "/models"
```

---

## Azure Deployment

### Infrastructure Components

#### Terraform (`azure/terraform/`)

**main.tf** (Key Resources):
```hcl
# Resource Group
resource "azurerm_resource_group" "main" {
  name     = "${var.project_name}-rg"
  location = var.location
}

# Virtual Network
resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-vnet"
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.project_name}-aks"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.project_name
  
  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D4s_v3"
  }
  
  identity {
    type = "SystemAssigned"
  }
  
  network_profile {
    network_plugin = "azure"
    network_policy = "calico"
  }
}

# PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${var.project_name}-postgres"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "15"
  administrator_login    = "dbadmin"
  administrator_password = var.db_password
  
  storage_mb = 131072
  sku_name   = "GP_Standard_D4s_v3"
  
  high_availability {
    mode = "ZoneRedundant"
  }
}

# Azure Cache for Redis
resource "azurerm_redis_cache" "main" {
  name                = "${var.project_name}-redis"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  capacity            = 2
  family              = "P"
  sku_name            = "Premium"
  
  redis_configuration {
    maxmemory_policy = "allkeys-lru"
  }
}
```

#### Azure ML (`azure/ml-workspace/`)

**deploy.py**:
```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment

def deploy_model(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    model_path: str
):
    ml_client = MLClient(
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    # Register model
    model = Model(
        path=model_path,
        name="wavefield-llm",
        description="Wave Field LLM"
    )
    registered_model = ml_client.models.create_or_update(model)
    
    # Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name="wavefield-llm-endpoint",
        description="Wave Field LLM Endpoint"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint.name,
        model=registered_model.id,
        instance_type="Standard_NC6s_v3",
        instance_count=1
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    return endpoint
```

---

## Cost Estimates

### AWS (Monthly)
- **EKS Cluster**: $73
- **EC2 Nodes (3x t3.xlarge)**: ~$300
- **GPU Nodes (1x g5.xlarge)**: ~$500
- **RDS PostgreSQL (db.r6g.xlarge)**: ~$400
- **ElastiCache Redis**: ~$300
- **S3 Storage (1TB)**: ~$25
- **Data Transfer**: ~$200
- **ALB**: ~$25
- **NAT Gateways (3)**: ~$100
- **CloudFront**: ~$100
- **Total**: **$3,000-$8,000/month**

### GCP (Monthly)
- **GKE Cluster**: $73
- **Compute Nodes**: ~$300
- **Cloud SQL**: ~$400
- **Memorystore Redis**: ~$250
- **Cloud Storage**: ~$25
- **Load Balancing**: ~$50
- **Cloud CDN**: ~$100
- **Total**: **$2,500-$6,000/month**

### Azure (Monthly)
- **AKS Cluster**: $73
- **VM Nodes**: ~$350
- **PostgreSQL Flexible Server**: ~$450
- **Azure Cache for Redis**: ~$300
- **Storage Accounts**: ~$30
- **Application Gateway**: ~$150
- **Azure CDN**: ~$100
- **Total**: **$2,800-$7,000/month**

---

## Security & Compliance

### Encryption
- **At Rest**: KMS/CMK encryption for all storage
- **In Transit**: TLS 1.2+ for all communications
- **Key Rotation**: Automatic annual rotation

### Access Control
- **IAM**: Least privilege principle
- **RBAC**: Kubernetes role-based access
- **MFA**: Required for admin access
- **Audit Logs**: All API calls logged

### Compliance Frameworks
- **HIPAA**: Healthcare data protection
- **SOC 2**: Security controls
- **GDPR**: Data privacy and residency
- **PCI DSS**: Payment card data (if applicable)

---

## Disaster Recovery

### Backup Strategy
- **RDS**: Automated daily backups, 30-day retention
- **Redis**: Daily snapshots
- **S3/Storage**: Versioning enabled
- **Cross-Region**: Optional replication

### RTO/RPO Targets
- **RTO**: < 4 hours
- **RPO**: < 1 hour
- **Multi-Region**: Active-passive setup

### Failover Procedures
1. DNS failover to secondary region
2. Restore database from latest backup
3. Scale up secondary region resources
4. Validate application functionality
5. Monitor and adjust

---

## Monitoring & Alerting

### Metrics Tracked
- **Infrastructure**: CPU, memory, disk, network
- **Application**: Request rate, latency, errors
- **Model**: Inference time, throughput, accuracy
- **Cost**: Resource utilization, spending trends

### Alert Thresholds
- **Critical**: P1 - Immediate response
- **High**: P2 - 1 hour response
- **Medium**: P3 - 4 hour response
- **Low**: P4 - Next business day

### Dashboards
- **Executive**: High-level KPIs
- **Operations**: Infrastructure health
- **Development**: Application metrics
- **ML**: Model performance

---

## Deployment Checklist

### Pre-Deployment
- [ ] Review and customize terraform.tfvars
- [ ] Set up backend state storage
- [ ] Configure DNS and certificates
- [ ] Create service accounts and IAM roles
- [ ] Set up monitoring and alerting
- [ ] Configure backup policies

### Deployment
- [ ] Run terraform plan and review
- [ ] Apply infrastructure changes
- [ ] Deploy EKS/GKE/AKS addons
- [ ] Deploy application workloads
- [ ] Configure ingress and load balancing
- [ ] Test endpoints and connectivity

### Post-Deployment
- [ ] Verify monitoring and alerts
- [ ] Test disaster recovery procedures
- [ ] Document runbooks
- [ ] Train operations team
- [ ] Schedule regular reviews
- [ ] Optimize costs

---

## Support & Maintenance

### Regular Tasks
- **Daily**: Monitor dashboards, review alerts
- **Weekly**: Review logs, check backups
- **Monthly**: Cost optimization, security patches
- **Quarterly**: DR testing, capacity planning
- **Annually**: Architecture review, compliance audit

### Escalation Path
1. On-call engineer
2. Team lead
3. Engineering manager
4. CTO/VP Engineering

---

## Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [GCP Best Practices](https://cloud.google.com/architecture/framework)
- [Azure Architecture Center](https://docs.microsoft.com/en-us/azure/architecture/)
- [Kubernetes Best Practices](https://kubernetes.io/docs/concepts/configuration/overview/)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-28  
**Maintained By**: ML Infrastructure Team