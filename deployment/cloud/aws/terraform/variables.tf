# Wave Field LLM - AWS Terraform Variables
# All configurable parameters with descriptions and validation

# Project Configuration
variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "wavefield-llm"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

# EKS Configuration
variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "wavefield-llm-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "eks_public_access" {
  description = "Enable public access to EKS API endpoint"
  type        = bool
  default     = true
}

variable "eks_public_access_cidrs" {
  description = "CIDR blocks allowed to access EKS API endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# EKS Addon Versions
variable "vpc_cni_version" {
  description = "VPC CNI addon version"
  type        = string
  default     = "v1.15.1-eksbuild.1"
}

variable "coredns_version" {
  description = "CoreDNS addon version"
  type        = string
  default     = "v1.10.1-eksbuild.6"
}

variable "kube_proxy_version" {
  description = "Kube-proxy addon version"
  type        = string
  default     = "v1.28.2-eksbuild.2"
}

variable "ebs_csi_version" {
  description = "EBS CSI driver addon version"
  type        = string
  default     = "v1.25.0-eksbuild.1"
}

# General Node Group Configuration
variable "general_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["t3.xlarge", "t3a.xlarge"]
}

variable "general_desired_size" {
  description = "Desired number of nodes in general node group"
  type        = number
  default     = 3

  validation {
    condition     = var.general_desired_size >= 1
    error_message = "Desired size must be at least 1."
  }
}

variable "general_min_size" {
  description = "Minimum number of nodes in general node group"
  type        = number
  default     = 2

  validation {
    condition     = var.general_min_size >= 1
    error_message = "Minimum size must be at least 1."
  }
}

variable "general_max_size" {
  description = "Maximum number of nodes in general node group"
  type        = number
  default     = 10

  validation {
    condition     = var.general_max_size >= var.general_min_size
    error_message = "Maximum size must be greater than or equal to minimum size."
  }
}

# GPU Node Group Configuration
variable "gpu_instance_types" {
  description = "Instance types for GPU node group"
  type        = list(string)
  default     = ["g5.xlarge", "g5.2xlarge"]
}

variable "gpu_desired_size" {
  description = "Desired number of nodes in GPU node group"
  type        = number
  default     = 1
}

variable "gpu_min_size" {
  description = "Minimum number of nodes in GPU node group"
  type        = number
  default     = 0
}

variable "gpu_max_size" {
  description = "Maximum number of nodes in GPU node group"
  type        = number
  default     = 5
}

# Spot Node Group Configuration
variable "spot_instance_types" {
  description = "Instance types for spot node group"
  type        = list(string)
  default     = ["t3.large", "t3a.large", "t3.xlarge", "t3a.xlarge"]
}

variable "spot_desired_size" {
  description = "Desired number of nodes in spot node group"
  type        = number
  default     = 2
}

variable "spot_min_size" {
  description = "Minimum number of nodes in spot node group"
  type        = number
  default     = 0
}

variable "spot_max_size" {
  description = "Maximum number of nodes in spot node group"
  type        = number
  default     = 20
}

# RDS Configuration
variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15.4"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "db_allocated_storage" {
  description = "Initial allocated storage in GB"
  type        = number
  default     = 100

  validation {
    condition     = var.db_allocated_storage >= 20
    error_message = "Allocated storage must be at least 20 GB."
  }
}

variable "db_max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling in GB"
  type        = number
  default     = 1000

  validation {
    condition     = var.db_max_allocated_storage >= var.db_allocated_storage
    error_message = "Max allocated storage must be greater than or equal to allocated storage."
  }
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "wavefieldllm"

  validation {
    condition     = can(regex("^[a-zA-Z][a-zA-Z0-9_]*$", var.db_name))
    error_message = "Database name must start with a letter and contain only alphanumeric characters and underscores."
  }
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "dbadmin"
  sensitive   = true
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.db_password) >= 16
    error_message = "Database password must be at least 16 characters long."
  }
}

variable "db_backup_retention_days" {
  description = "Number of days to retain automated backups"
  type        = number
  default     = 30

  validation {
    condition     = var.db_backup_retention_days >= 7 && var.db_backup_retention_days <= 35
    error_message = "Backup retention must be between 7 and 35 days."
  }
}

variable "create_read_replica" {
  description = "Create RDS read replica"
  type        = bool
  default     = true
}

variable "db_replica_instance_class" {
  description = "RDS read replica instance class"
  type        = string
  default     = "db.r6g.large"
}

# ElastiCache Redis Configuration
variable "redis_version" {
  description = "Redis version"
  type        = string
  default     = "7.0"
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in the cluster"
  type        = number
  default     = 3

  validation {
    condition     = var.redis_num_cache_nodes >= 2
    error_message = "Must have at least 2 cache nodes for high availability."
  }
}

variable "redis_auth_token" {
  description = "Redis authentication token"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.redis_auth_token) >= 16
    error_message = "Redis auth token must be at least 16 characters long."
  }
}

variable "redis_snapshot_retention_limit" {
  description = "Number of days to retain Redis snapshots"
  type        = number
  default     = 7

  validation {
    condition     = var.redis_snapshot_retention_limit >= 0 && var.redis_snapshot_retention_limit <= 35
    error_message = "Snapshot retention must be between 0 and 35 days."
  }
}

# CloudFront Configuration
variable "cloudfront_price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_100"

  validation {
    condition     = contains(["PriceClass_100", "PriceClass_200", "PriceClass_All"], var.cloudfront_price_class)
    error_message = "Price class must be PriceClass_100, PriceClass_200, or PriceClass_All."
  }
}

# DNS Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "wavefield-llm.example.com"
}

variable "create_route53_zone" {
  description = "Create Route53 hosted zone"
  type        = bool
  default     = false
}

# Logging Configuration
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 90

  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

# Alerting Configuration
variable "alert_emails" {
  description = "Email addresses for CloudWatch alerts"
  type        = list(string)
  default     = []
}

# Security Configuration
variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access resources"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Compliance Configuration
variable "enable_hipaa_compliance" {
  description = "Enable HIPAA compliance configurations"
  type        = bool
  default     = false
}

variable "enable_soc2_compliance" {
  description = "Enable SOC2 compliance configurations"
  type        = bool
  default     = false
}

variable "enable_gdpr_compliance" {
  description = "Enable GDPR compliance configurations"
  type        = bool
  default     = false
}

# Disaster Recovery Configuration
variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = false
}

variable "backup_region" {
  description = "AWS region for backup replication"
  type        = string
  default     = "us-west-2"
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "enable_savings_plans" {
  description = "Enable Savings Plans recommendations"
  type        = bool
  default     = true
}

# Tagging Configuration
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "ml-infrastructure"
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "ml-team"
}

# Monitoring Configuration
variable "enable_enhanced_monitoring" {
  description = "Enable enhanced monitoring for RDS and ElastiCache"
  type        = bool
  default     = true
}

variable "enable_performance_insights" {
  description = "Enable Performance Insights for RDS"
  type        = bool
  default     = true
}

# Auto Scaling Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable Kubernetes Cluster Autoscaler"
  type        = bool
  default     = true
}

variable "enable_horizontal_pod_autoscaler" {
  description = "Enable Horizontal Pod Autoscaler"
  type        = bool
  default     = true
}

# Network Security
variable "enable_vpc_flow_logs" {
  description = "Enable VPC Flow Logs"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable AWS WAF for ALB"
  type        = bool
  default     = true
}

variable "enable_shield" {
  description = "Enable AWS Shield Advanced"
  type        = bool
  default     = false
}

# Encryption Configuration
variable "enable_encryption_at_rest" {
  description = "Enable encryption at rest for all resources"
  type        = bool
  default     = true
}

variable "enable_encryption_in_transit" {
  description = "Enable encryption in transit for all resources"
  type        = bool
  default     = true
}

# Blue-Green Deployment
variable "enable_blue_green_deployment" {
  description = "Enable blue-green deployment strategy"
  type        = bool
  default     = false
}

variable "blue_green_traffic_shift_percentage" {
  description = "Percentage of traffic to shift during blue-green deployment"
  type        = number
  default     = 10

  validation {
    condition     = var.blue_green_traffic_shift_percentage >= 0 && var.blue_green_traffic_shift_percentage <= 100
    error_message = "Traffic shift percentage must be between 0 and 100."
  }
}

# Multi-Region Configuration
variable "enable_multi_region" {
  description = "Enable multi-region deployment"
  type        = bool
  default     = false
}

variable "secondary_regions" {
  description = "List of secondary regions for multi-region deployment"
  type        = list(string)
  default     = []
}