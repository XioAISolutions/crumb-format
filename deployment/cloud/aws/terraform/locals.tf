# Wave Field LLM - Terraform Local Values
# Computed values and common tags

locals {
  # Common tags applied to all resources
  common_tags = merge(
    {
      Project             = var.project_name
      Environment         = var.environment
      ManagedBy           = "Terraform"
      Owner               = var.owner
      CostCenter          = var.cost_center
      Application         = "Wave Field LLM"
      Repository          = "wavefield-llm"
      Compliance          = join(",", compact([
        var.enable_hipaa_compliance ? "HIPAA" : "",
        var.enable_soc2_compliance ? "SOC2" : "",
        var.enable_gdpr_compliance ? "GDPR" : ""
      ]))
      BackupEnabled       = var.enable_cross_region_backup ? "true" : "false"
      DisasterRecovery    = var.enable_multi_region ? "true" : "false"
      CreatedDate         = formatdate("YYYY-MM-DD", timestamp())
    },
    var.additional_tags
  )

  # EKS cluster name with environment suffix
  cluster_full_name = "${var.cluster_name}-${var.environment}"

  # OIDC provider URL without https://
  oidc_provider_url = replace(aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")

  # Account and region information
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.name
  partition  = data.aws_partition.current.partition

  # Availability zones (limit to 3 for cost optimization)
  azs = slice(data.aws_availability_zones.available.names, 0, 3)

  # VPC CIDR calculations
  vpc_cidr_prefix = split("/", var.vpc_cidr)[1]
  
  # Subnet CIDR blocks
  public_subnet_cidrs   = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i)]
  private_subnet_cidrs  = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i + 3)]
  database_subnet_cidrs = [for i in range(3) : cidrsubnet(var.vpc_cidr, 4, i + 6)]

  # S3 bucket names (must be globally unique)
  s3_models_bucket = "${var.project_name}-models-${local.account_id}-${local.region}"
  s3_data_bucket   = "${var.project_name}-data-${local.account_id}-${local.region}"
  s3_logs_bucket   = "${var.project_name}-logs-${local.account_id}-${local.region}"

  # KMS key aliases
  kms_eks_alias         = "alias/${var.project_name}-eks-${var.environment}"
  kms_rds_alias         = "alias/${var.project_name}-rds-${var.environment}"
  kms_s3_alias          = "alias/${var.project_name}-s3-${var.environment}"
  kms_elasticache_alias = "alias/${var.project_name}-elasticache-${var.environment}"
  kms_cloudwatch_alias  = "alias/${var.project_name}-cloudwatch-${var.environment}"
  kms_sns_alias         = "alias/${var.project_name}-sns-${var.environment}"

  # CloudWatch log group names
  log_group_eks          = "/aws/eks/${var.cluster_name}/cluster"
  log_group_redis_slow   = "/aws/elasticache/${var.project_name}-redis/slow-log"
  log_group_redis_engine = "/aws/elasticache/${var.project_name}-redis/engine-log"
  log_group_lambda       = "/aws/lambda/${var.project_name}"
  log_group_ecs          = "/aws/ecs/${var.project_name}"

  # Security group names
  sg_eks_cluster_name   = "${var.project_name}-eks-cluster-sg"
  sg_eks_nodes_name     = "${var.project_name}-eks-nodes-sg"
  sg_alb_name           = "${var.project_name}-alb-sg"
  sg_rds_name           = "${var.project_name}-rds-sg"
  sg_redis_name         = "${var.project_name}-redis-sg"
  sg_vpc_endpoints_name = "${var.project_name}-vpc-endpoints-sg"

  # IAM role names
  iam_eks_cluster_role_name              = "${var.project_name}-eks-cluster-role"
  iam_eks_node_group_role_name           = "${var.project_name}-eks-node-group-role"
  iam_vpc_cni_role_name                  = "${var.project_name}-vpc-cni-role"
  iam_ebs_csi_role_name                  = "${var.project_name}-ebs-csi-role"
  iam_aws_lb_controller_role_name        = "${var.project_name}-aws-lb-controller-role"
  iam_external_dns_role_name             = "${var.project_name}-external-dns-role"
  iam_cluster_autoscaler_role_name       = "${var.project_name}-cluster-autoscaler-role"

  # Database configuration
  db_identifier        = "${var.project_name}-postgres-${var.environment}"
  db_replica_identifier = "${var.project_name}-postgres-replica-${var.environment}"
  db_subnet_group_name = "${var.project_name}-db-subnet-group"
  db_parameter_group_name = "${var.project_name}-postgres-params"

  # ElastiCache configuration
  redis_replication_group_id = "${var.project_name}-redis-${var.environment}"
  redis_subnet_group_name    = "${var.project_name}-redis-subnet-group"
  redis_parameter_group_name = "${var.project_name}-redis-params"

  # Load balancer configuration
  alb_name            = "${var.project_name}-alb"
  alb_target_group_name = "${var.project_name}-tg"

  # CloudFront configuration
  cloudfront_comment = "CDN for ${var.project_name} ${var.environment}"

  # Route53 configuration
  domain_parts = split(".", var.domain_name)
  root_domain  = length(local.domain_parts) > 2 ? join(".", slice(local.domain_parts, length(local.domain_parts) - 2, length(local.domain_parts))) : var.domain_name

  # SNS topic names
  sns_alerts_topic_name = "${var.project_name}-alerts-${var.environment}"

  # Kubernetes service account names
  k8s_sa_vpc_cni                = "aws-node"
  k8s_sa_ebs_csi                = "ebs-csi-controller-sa"
  k8s_sa_aws_lb_controller      = "aws-load-balancer-controller"
  k8s_sa_external_dns           = "external-dns"
  k8s_sa_cluster_autoscaler     = "cluster-autoscaler"

  # Kubernetes namespaces
  k8s_namespace_system = "kube-system"
  k8s_namespace_app    = var.project_name

  # Node group configurations
  node_groups = {
    general = {
      name           = "${var.cluster_name}-general"
      instance_types = var.general_instance_types
      capacity_type  = "ON_DEMAND"
      desired_size   = var.general_desired_size
      min_size       = var.general_min_size
      max_size       = var.general_max_size
      labels = {
        role = "general"
      }
      taints = []
    }
    gpu = {
      name           = "${var.cluster_name}-gpu"
      instance_types = var.gpu_instance_types
      capacity_type  = "ON_DEMAND"
      desired_size   = var.gpu_desired_size
      min_size       = var.gpu_min_size
      max_size       = var.gpu_max_size
      labels = {
        role = "gpu"
        "nvidia.com/gpu" = "true"
      }
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    spot = {
      name           = "${var.cluster_name}-spot"
      instance_types = var.spot_instance_types
      capacity_type  = "SPOT"
      desired_size   = var.spot_desired_size
      min_size       = var.spot_min_size
      max_size       = var.spot_max_size
      labels = {
        role = "spot"
        "node.kubernetes.io/lifecycle" = "spot"
      }
      taints = [
        {
          key    = "spot"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  # Compliance configurations
  compliance_tags = {
    HIPAA = var.enable_hipaa_compliance ? {
      "hipaa:encryption-at-rest"    = "true"
      "hipaa:encryption-in-transit" = "true"
      "hipaa:audit-logging"         = "true"
      "hipaa:backup-enabled"        = "true"
    } : {}
    
    SOC2 = var.enable_soc2_compliance ? {
      "soc2:data-classification"    = "confidential"
      "soc2:access-control"         = "enabled"
      "soc2:monitoring"             = "enabled"
      "soc2:incident-response"      = "enabled"
    } : {}
    
    GDPR = var.enable_gdpr_compliance ? {
      "gdpr:data-residency"         = local.region
      "gdpr:data-retention"         = "${var.log_retention_days}-days"
      "gdpr:right-to-erasure"       = "enabled"
      "gdpr:data-portability"       = "enabled"
    } : {}
  }

  # Merge all compliance tags
  all_compliance_tags = merge(
    local.compliance_tags.HIPAA,
    local.compliance_tags.SOC2,
    local.compliance_tags.GDPR
  )

  # Final tags with compliance
  final_tags = merge(
    local.common_tags,
    local.all_compliance_tags
  )

  # Cost allocation tags
  cost_tags = {
    "cost:project"     = var.project_name
    "cost:environment" = var.environment
    "cost:center"      = var.cost_center
    "cost:owner"       = var.owner
  }

  # Monitoring and alerting configuration
  alarm_actions = var.enable_enhanced_monitoring ? [aws_sns_topic.alerts.arn] : []

  # Blue-green deployment configuration
  blue_green_config = var.enable_blue_green_deployment ? {
    enabled                = true
    traffic_shift_percent  = var.blue_green_traffic_shift_percentage
    deployment_type        = "BLUE_GREEN"
    termination_wait_time  = 300
  } : null

  # Multi-region configuration
  multi_region_config = var.enable_multi_region ? {
    enabled           = true
    primary_region    = local.region
    secondary_regions = var.secondary_regions
    replication_enabled = true
  } : null

  # Backup configuration
  backup_config = {
    enabled               = true
    retention_days        = var.db_backup_retention_days
    cross_region_enabled  = var.enable_cross_region_backup
    backup_region         = var.backup_region
    backup_window         = "03:00-04:00"
    maintenance_window    = "mon:04:00-mon:05:00"
  }

  # Auto-scaling configuration
  autoscaling_config = {
    cluster_autoscaler_enabled = var.enable_cluster_autoscaler
    hpa_enabled                = var.enable_horizontal_pod_autoscaler
    target_cpu_utilization     = 70
    target_memory_utilization  = 80
    scale_down_delay           = 300
    scale_up_delay             = 60
  }

  # Network security configuration
  network_security_config = {
    vpc_flow_logs_enabled = var.enable_vpc_flow_logs
    waf_enabled           = var.enable_waf
    shield_enabled        = var.enable_shield
    nacl_enabled          = true
    security_groups_strict = true
  }

  # Encryption configuration
  encryption_config = {
    at_rest_enabled     = var.enable_encryption_at_rest
    in_transit_enabled  = var.enable_encryption_in_transit
    kms_key_rotation    = true
    tls_version_minimum = "1.2"
  }

  # Resource naming convention
  resource_prefix = "${var.project_name}-${var.environment}"
  
  # DNS configuration
  dns_config = {
    domain_name         = var.domain_name
    create_zone         = var.create_route53_zone
    ttl_default         = 300
    health_check_enabled = true
  }

  # Monitoring thresholds
  monitoring_thresholds = {
    cpu_high              = 80
    memory_high           = 85
    disk_high             = 90
    connection_count_high = 1000
    error_rate_high       = 5
    latency_high          = 1000
  }
}