# Wave Field LLM - AWS Terraform Outputs
# Export all important values for use by other systems

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

output "database_subnet_ids" {
  description = "IDs of database subnets"
  value       = aws_subnet.database[*].id
}

output "nat_gateway_ips" {
  description = "Elastic IPs of NAT gateways"
  value       = aws_eip.nat[*].public_ip
}

# EKS Outputs
output "eks_cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "eks_cluster_version" {
  description = "EKS cluster Kubernetes version"
  value       = aws_eks_cluster.main.version
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_security_group.eks_cluster.id
}

output "eks_cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data for cluster authentication"
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "eks_cluster_oidc_issuer_url" {
  description = "OIDC issuer URL for the EKS cluster"
  value       = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

output "eks_node_group_general_id" {
  description = "ID of the general EKS node group"
  value       = aws_eks_node_group.general.id
}

output "eks_node_group_gpu_id" {
  description = "ID of the GPU EKS node group"
  value       = aws_eks_node_group.gpu.id
}

output "eks_node_group_spot_id" {
  description = "ID of the spot EKS node group"
  value       = aws_eks_node_group.spot.id
}

# IAM Outputs
output "eks_cluster_role_arn" {
  description = "ARN of the EKS cluster IAM role"
  value       = aws_iam_role.eks_cluster.arn
}

output "eks_node_group_role_arn" {
  description = "ARN of the EKS node group IAM role"
  value       = aws_iam_role.eks_node_group.arn
}

output "vpc_cni_role_arn" {
  description = "ARN of the VPC CNI IAM role"
  value       = aws_iam_role.vpc_cni.arn
}

output "ebs_csi_role_arn" {
  description = "ARN of the EBS CSI driver IAM role"
  value       = aws_iam_role.ebs_csi.arn
}

# RDS Outputs
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
}

output "rds_address" {
  description = "RDS instance address"
  value       = aws_db_instance.main.address
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "rds_database_name" {
  description = "RDS database name"
  value       = aws_db_instance.main.db_name
}

output "rds_username" {
  description = "RDS master username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

output "rds_arn" {
  description = "ARN of the RDS instance"
  value       = aws_db_instance.main.arn
}

output "rds_replica_endpoint" {
  description = "RDS read replica endpoint"
  value       = var.create_read_replica ? aws_db_instance.read_replica[0].endpoint : null
}

output "rds_security_group_id" {
  description = "Security group ID for RDS"
  value       = aws_security_group.rds.id
}

# ElastiCache Outputs
output "redis_endpoint" {
  description = "ElastiCache Redis primary endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}

output "redis_reader_endpoint" {
  description = "ElastiCache Redis reader endpoint"
  value       = aws_elasticache_replication_group.main.reader_endpoint_address
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.main.port
}

output "redis_arn" {
  description = "ARN of the ElastiCache replication group"
  value       = aws_elasticache_replication_group.main.arn
}

output "redis_security_group_id" {
  description = "Security group ID for Redis"
  value       = aws_security_group.redis.id
}

# S3 Outputs
output "s3_models_bucket_name" {
  description = "Name of the S3 bucket for models"
  value       = aws_s3_bucket.models.id
}

output "s3_models_bucket_arn" {
  description = "ARN of the S3 bucket for models"
  value       = aws_s3_bucket.models.arn
}

output "s3_data_bucket_name" {
  description = "Name of the S3 bucket for data"
  value       = aws_s3_bucket.data.id
}

output "s3_data_bucket_arn" {
  description = "ARN of the S3 bucket for data"
  value       = aws_s3_bucket.data.arn
}

output "s3_logs_bucket_name" {
  description = "Name of the S3 bucket for logs"
  value       = aws_s3_bucket.logs.id
}

output "s3_logs_bucket_arn" {
  description = "ARN of the S3 bucket for logs"
  value       = aws_s3_bucket.logs.arn
}

# Load Balancer Outputs
output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer"
  value       = aws_lb.main.arn
}

output "alb_zone_id" {
  description = "Zone ID of the Application Load Balancer"
  value       = aws_lb.main.zone_id
}

output "alb_target_group_arn" {
  description = "ARN of the ALB target group"
  value       = aws_lb_target_group.main.arn
}

output "alb_security_group_id" {
  description = "Security group ID for ALB"
  value       = aws_security_group.alb.id
}

# CloudFront Outputs
output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.id
}

output "cloudfront_distribution_arn" {
  description = "ARN of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.arn
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.domain_name
}

output "cloudfront_hosted_zone_id" {
  description = "Hosted zone ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.main.hosted_zone_id
}

# Route53 Outputs
output "route53_zone_id" {
  description = "ID of the Route53 hosted zone"
  value       = var.create_route53_zone ? aws_route53_zone.main[0].zone_id : null
}

output "route53_name_servers" {
  description = "Name servers for the Route53 hosted zone"
  value       = var.create_route53_zone ? aws_route53_zone.main[0].name_servers : null
}

# ACM Outputs
output "acm_certificate_arn" {
  description = "ARN of the ACM certificate"
  value       = aws_acm_certificate.main.arn
}

output "acm_certificate_status" {
  description = "Status of the ACM certificate"
  value       = aws_acm_certificate.main.status
}

# KMS Outputs
output "kms_eks_key_id" {
  description = "ID of the KMS key for EKS"
  value       = aws_kms_key.eks.id
}

output "kms_eks_key_arn" {
  description = "ARN of the KMS key for EKS"
  value       = aws_kms_key.eks.arn
}

output "kms_rds_key_id" {
  description = "ID of the KMS key for RDS"
  value       = aws_kms_key.rds.id
}

output "kms_rds_key_arn" {
  description = "ARN of the KMS key for RDS"
  value       = aws_kms_key.rds.arn
}

output "kms_s3_key_id" {
  description = "ID of the KMS key for S3"
  value       = aws_kms_key.s3.id
}

output "kms_s3_key_arn" {
  description = "ARN of the KMS key for S3"
  value       = aws_kms_key.s3.arn
}

output "kms_elasticache_key_id" {
  description = "ID of the KMS key for ElastiCache"
  value       = aws_kms_key.elasticache.id
}

output "kms_elasticache_key_arn" {
  description = "ARN of the KMS key for ElastiCache"
  value       = aws_kms_key.elasticache.arn
}

# CloudWatch Outputs
output "cloudwatch_log_group_eks" {
  description = "Name of the CloudWatch log group for EKS"
  value       = aws_cloudwatch_log_group.eks_cluster.name
}

output "cloudwatch_log_group_redis_slow" {
  description = "Name of the CloudWatch log group for Redis slow logs"
  value       = aws_cloudwatch_log_group.redis_slow_log.name
}

output "cloudwatch_log_group_redis_engine" {
  description = "Name of the CloudWatch log group for Redis engine logs"
  value       = aws_cloudwatch_log_group.redis_engine_log.name
}

# SNS Outputs
output "sns_alerts_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}

# Connection Strings (for application configuration)
output "database_connection_string" {
  description = "PostgreSQL connection string (without password)"
  value       = "postgresql://${aws_db_instance.main.username}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
  sensitive   = true
}

output "redis_connection_string" {
  description = "Redis connection string (without auth token)"
  value       = "rediss://${aws_elasticache_replication_group.main.primary_endpoint_address}:${aws_elasticache_replication_group.main.port}"
  sensitive   = true
}

# Kubectl Configuration Command
output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${data.aws_region.current.name} --name ${aws_eks_cluster.main.name}"
}

# Application URL
output "application_url" {
  description = "Application URL"
  value       = "https://${var.domain_name}"
}

# Cost Estimation
output "estimated_monthly_cost" {
  description = "Estimated monthly cost in USD (approximate)"
  value = {
    eks_cluster       = "73.00"
    eks_nodes_general = "${var.general_desired_size * 100}"
    eks_nodes_gpu     = "${var.gpu_desired_size * 500}"
    eks_nodes_spot    = "${var.spot_desired_size * 40}"
    rds_primary       = "300-500"
    rds_replica       = var.create_read_replica ? "150-250" : "0"
    elasticache       = "200-400"
    s3_storage        = "50-200"
    data_transfer     = "100-500"
    cloudfront        = "50-200"
    alb               = "25"
    nat_gateways      = "100"
    total_estimate    = "3000-8000"
  }
}

# Security Group IDs (for reference)
output "security_groups" {
  description = "Map of security group IDs"
  value = {
    eks_cluster   = aws_security_group.eks_cluster.id
    eks_nodes     = aws_security_group.eks_nodes.id
    alb           = aws_security_group.alb.id
    rds           = aws_security_group.rds.id
    redis         = aws_security_group.redis.id
    vpc_endpoints = aws_security_group.vpc_endpoints.id
  }
}

# Deployment Information
output "deployment_info" {
  description = "Deployment information"
  value = {
    project_name  = var.project_name
    environment   = var.environment
    region        = data.aws_region.current.name
    account_id    = data.aws_caller_identity.current.account_id
    deployed_at   = timestamp()
  }
}