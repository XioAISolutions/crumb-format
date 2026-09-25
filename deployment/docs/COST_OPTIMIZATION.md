# Wave Field LLM Cost Optimization Guide

## Executive Summary

This guide provides comprehensive strategies for optimizing infrastructure costs while maintaining performance and reliability for Wave Field LLM deployments.

**Potential Savings:** 30-60% of infrastructure costs  
**Implementation Time:** 2-4 weeks  
**ROI:** Typically achieved within 3 months

## Current Cost Breakdown

### Typical Monthly Costs (Production)

| Category | Service | Monthly Cost | % of Total |
|----------|---------|--------------|------------|
| **Compute** | EKS/GKE Nodes (5x m5.2xlarge) | $1,460 | 35% |
| **GPU** | 3x NVIDIA A100 instances | $2,190 | 52% |
| **Database** | RDS PostgreSQL (db.r5.xlarge) | $292 | 7% |
| **Storage** | EBS volumes (500GB) | $50 | 1% |
| **Networking** | Data transfer & Load Balancer | $100 | 2% |
| **Monitoring** | CloudWatch, Prometheus storage | $50 | 1% |
| **Backup** | S3 storage & snapshots | $45 | 1% |
| **Other** | DNS, certificates, misc | $13 | <1% |
| **Total** | | **$4,200** | **100%** |

## Optimization Strategies

### 1. Compute Optimization (Save 20-40%)

#### 1.1 Right-Sizing Instances

**Current State:**
```yaml
# Over-provisioned
resources:
  requests:
    cpu: 4000m
    memory: 8Gi
  limits:
    cpu: 8000m
    memory: 16Gi
```

**Optimized:**
```yaml
# Right-sized based on actual usage
resources:
  requests:
    cpu: 2000m
    memory: 4Gi
  limits:
    cpu: 4000m
    memory: 8Gi
```

**Analysis Script:**
```bash
#!/bin/bash
# Analyze actual resource usage
kubectl top pods -n wavefield-llm --containers | \
  awk '{print $1, $2, $3}' | \
  sort -k3 -h | \
  tail -20

# Recommendation: Set requests to 80% of average usage
# Set limits to 150% of peak usage
```

**Estimated Savings:** $300-500/month

#### 1.2 Use Spot/Preemptible Instances

**AWS Spot Instances:**
```yaml
# EKS Node Group with Spot
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: wavefield-llm-prod
nodeGroups:
  - name: spot-workers
    instancesDistribution:
      instanceTypes:
        - m5.2xlarge
        - m5a.2xlarge
        - m5n.2xlarge
      onDemandBaseCapacity: 2
      onDemandPercentageAboveBaseCapacity: 20
      spotInstancePools: 3
    minSize: 3
    maxSize: 10
    labels:
      workload: stateless
    taints:
      - key: spot
        value: "true"
        effect: NoSchedule
```

**Pod Configuration:**
```yaml
# Tolerate spot instances
tolerations:
  - key: spot
    operator: Equal
    value: "true"
    effect: NoSchedule

# Use pod disruption budget
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: wavefield-api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: wavefield-api
```

**Estimated Savings:** $500-700/month (60-70% discount)

#### 1.3 Implement Cluster Autoscaling

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: wavefield-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: wavefield-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 30
```

**Estimated Savings:** $200-400/month

### 2. GPU Optimization (Save 30-50%)

#### 2.1 GPU Sharing and Time-Slicing

**NVIDIA MIG (Multi-Instance GPU):**
```yaml
# Enable MIG on A100
apiVersion: v1
kind: ConfigMap
metadata:
  name: nvidia-device-plugin-config
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        resources:
          - name: nvidia.com/gpu
            replicas: 4  # Share GPU among 4 pods
```

**Estimated Savings:** $800-1,200/month

#### 2.2 Use Smaller GPU Instances

| GPU Type | Cost/hour | Performance | Use Case |
|----------|-----------|-------------|----------|
| A100 (80GB) | $3.06 | 100% | Large models |
| A100 (40GB) | $2.45 | 95% | Most workloads |
| A10G | $1.01 | 60% | Inference only |
| T4 | $0.526 | 40% | Light inference |

**Recommendation:** Use A10G for inference, reserve A100 for training

**Estimated Savings:** $600-900/month

#### 2.3 Implement GPU Scheduling

```python
# Smart GPU scheduling
class GPUScheduler:
    def schedule_inference(self, request):
        # Use cheaper GPUs during off-peak hours
        current_hour = datetime.now().hour
        
        if 2 <= current_hour <= 6:  # Off-peak
            return self.schedule_on_spot_gpu(request)
        else:
            return self.schedule_on_demand_gpu(request)
    
    def batch_requests(self, requests, max_wait_ms=100):
        # Batch multiple requests to maximize GPU utilization
        batch = []
        start_time = time.time()
        
        while len(batch) < 32 and (time.time() - start_time) < max_wait_ms/1000:
            if requests:
                batch.append(requests.pop(0))
        
        return self.process_batch(batch)
```

**Estimated Savings:** $300-500/month

### 3. Storage Optimization (Save 40-60%)

#### 3.1 Use Appropriate Storage Classes

```yaml
# Storage tiering strategy
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: standard-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "1000"
  throughput: "125"
---
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: cold-storage
provisioner: kubernetes.io/aws-ebs
parameters:
  type: sc1  # Cold HDD
```

**Usage Guidelines:**
- **fast-ssd:** Database, hot cache
- **standard-ssd:** Application data, logs
- **cold-storage:** Backups, archives

**Estimated Savings:** $20-30/month

#### 3.2 Implement Data Lifecycle Policies

```yaml
# S3 Lifecycle Policy
{
  "Rules": [
    {
      "Id": "MoveToIA",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ],
      "Expiration": {
        "Days": 2555  # 7 years
      }
    }
  ]
}
```

**Estimated Savings:** $15-25/month

### 4. Database Optimization (Save 20-30%)

#### 4.1 Right-Size Database Instance

**Analysis Query:**
```sql
-- Check actual database usage
SELECT 
    pg_size_pretty(pg_database_size('wavefield')) as db_size,
    (SELECT count(*) FROM pg_stat_activity) as connections,
    (SELECT setting FROM pg_settings WHERE name = 'max_connections') as max_conn;

-- Check query performance
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 20;
```

**Recommendation:**
- Current: db.r5.xlarge (4 vCPU, 32GB) - $292/month
- Optimized: db.r5.large (2 vCPU, 16GB) - $146/month

**Estimated Savings:** $146/month

#### 4.2 Use Read Replicas Efficiently

```python
# Smart read/write splitting
class DatabaseRouter:
    def route_query(self, query_type, load):
        if query_type == 'write':
            return self.primary_db
        
        # Route reads to least loaded replica
        if load < 0.5:
            return self.replica_1
        elif load < 0.8:
            return self.replica_2
        else:
            return self.primary_db  # Fallback
```

**Estimated Savings:** $50-100/month

### 5. Network Optimization (Save 30-50%)

#### 5.1 Reduce Data Transfer Costs

**Strategies:**
1. **Use CloudFront/CDN:** Cache static content
2. **Compress Responses:** Enable gzip/brotli
3. **Regional Deployment:** Deploy closer to users
4. **VPC Endpoints:** Avoid NAT gateway costs

```nginx
# NGINX compression
gzip on;
gzip_vary on;
gzip_comp_level 6;
gzip_types text/plain text/css application/json application/javascript;

# Response size reduction
location /api/ {
    # Enable compression
    gzip on;
    
    # Add caching headers
    add_header Cache-Control "public, max-age=300";
}
```

**Estimated Savings:** $30-50/month

#### 5.2 Optimize Load Balancer Usage

```yaml
# Use NLB instead of ALB for simple routing
apiVersion: v1
kind: Service
metadata:
  name: wavefield-api
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  ports:
    - port: 443
      targetPort: 8000
```

**Cost Comparison:**
- ALB: $22.50/month + $0.008/LCU-hour
- NLB: $16.20/month + $0.006/NLCU-hour

**Estimated Savings:** $10-20/month

### 6. Monitoring and Observability (Save 40-60%)

#### 6.1 Optimize Log Retention

```yaml
# Loki retention configuration
limits_config:
  retention_period: 168h  # 7 days instead of 30

# CloudWatch log retention
aws logs put-retention-policy \
  --log-group-name /aws/eks/wavefield-llm \
  --retention-in-days 7
```

**Estimated Savings:** $20-30/month

#### 6.2 Sample Metrics Strategically

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'high-priority'
    scrape_interval: 15s
    static_configs:
      - targets: ['api:8000']
  
  - job_name: 'low-priority'
    scrape_interval: 60s  # Less frequent
    static_configs:
      - targets: ['background-jobs:8001']
```

**Estimated Savings:** $10-15/month

### 7. Reserved Capacity (Save 30-70%)

#### 7.1 Purchase Reserved Instances

**1-Year Reserved Instances:**
- Standard: 40% discount
- Convertible: 31% discount

**3-Year Reserved Instances:**
- Standard: 60% discount
- Convertible: 54% discount

**Recommendation:**
```bash
# Reserve baseline capacity (60% of average usage)
# Use on-demand/spot for burst capacity

# Example: 3 m5.2xlarge reserved (1-year)
# Cost: $1,460/month on-demand
# Reserved: $876/month (40% savings)
```

**Estimated Savings:** $584/month

#### 7.2 Savings Plans

**Compute Savings Plans:**
- 1-year: 42% discount
- 3-year: 66% discount

**Estimated Savings:** $600-900/month

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
- [ ] Enable autoscaling
- [ ] Implement log retention policies
- [ ] Enable compression
- [ ] Right-size over-provisioned resources

**Expected Savings:** $300-500/month

### Phase 2: Infrastructure Changes (Weeks 2-3)
- [ ] Migrate to spot instances
- [ ] Implement GPU sharing
- [ ] Optimize storage classes
- [ ] Set up read replicas

**Expected Savings:** $800-1,200/month

### Phase 3: Long-term Commitments (Week 4)
- [ ] Purchase reserved instances
- [ ] Implement savings plans
- [ ] Negotiate enterprise discounts

**Expected Savings:** $600-900/month

## Monitoring Cost Optimization

### Cost Tracking Dashboard

```python
# Cost tracking script
import boto3
from datetime import datetime, timedelta

def get_daily_costs():
    ce = boto3.client('ce')
    
    end = datetime.now().date()
    start = end - timedelta(days=30)
    
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start.strftime('%Y-%m-%d'),
            'End': end.strftime('%Y-%m-%d')
        },
        Granularity='DAILY',
        Metrics=['UnblendedCost'],
        GroupBy=[
            {'Type': 'SERVICE', 'Key': 'SERVICE'},
            {'Type': 'TAG', 'Key': 'Environment'}
        ]
    )
    
    return response['ResultsByTime']

# Set up cost alerts
def create_budget_alert():
    budgets = boto3.client('budgets')
    
    budgets.create_budget(
        AccountId='123456789012',
        Budget={
            'BudgetName': 'wavefield-llm-monthly',
            'BudgetLimit': {
                'Amount': '4000',
                'Unit': 'USD'
            },
            'TimeUnit': 'MONTHLY',
            'BudgetType': 'COST'
        },
        NotificationsWithSubscribers=[
            {
                'Notification': {
                    'NotificationType': 'ACTUAL',
                    'ComparisonOperator': 'GREATER_THAN',
                    'Threshold': 80,
                    'ThresholdType': 'PERCENTAGE'
                },
                'Subscribers': [
                    {
                        'SubscriptionType': 'EMAIL',
                        'Address': 'finance@wavefield-llm.example.com'
                    }
                ]
            }
        ]
    )
```

### Key Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Cost per 1K requests | $0.05 | $0.08 |
| Cost per 1K tokens | $0.002 | $0.003 |
| GPU utilization | >70% | <50% |
| CPU utilization | 60-80% | <40% or >90% |
| Storage growth | <10% monthly | >15% monthly |

## Cost Optimization Checklist

### Monthly Review
- [ ] Review resource utilization
- [ ] Identify unused resources
- [ ] Check for cost anomalies
- [ ] Update reserved capacity
- [ ] Review and optimize queries

### Quarterly Review
- [ ] Evaluate instance types
- [ ] Review storage classes
- [ ] Assess network patterns
- [ ] Update cost forecasts
- [ ] Benchmark against industry

### Annual Review
- [ ] Comprehensive cost audit
- [ ] Renegotiate contracts
- [ ] Evaluate new services
- [ ] Update optimization strategy
- [ ] Set new cost targets

## Expected Results

### Before Optimization
- **Monthly Cost:** $4,200
- **Cost per 1K requests:** $0.084
- **GPU utilization:** 45%
- **Wasted resources:** ~35%

### After Optimization
- **Monthly Cost:** $2,100-2,800
- **Cost per 1K requests:** $0.042-0.056
- **GPU utilization:** 70%+
- **Wasted resources:** <10%

### Total Savings
- **Monthly:** $1,400-2,100 (33-50%)
- **Annual:** $16,800-25,200

## Conclusion

Implementing these cost optimization strategies can reduce infrastructure costs by 30-60% while maintaining or improving performance. The key is continuous monitoring and iterative optimization.

## Resources

- [AWS Cost Optimization](https://aws.amazon.com/pricing/cost-optimization/)
- [GCP Cost Management](https://cloud.google.com/cost-management)
- [Azure Cost Management](https://azure.microsoft.com/en-us/services/cost-management/)
- [FinOps Foundation](https://www.finops.org/)

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0