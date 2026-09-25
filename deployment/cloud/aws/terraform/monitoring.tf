# Wave Field LLM - Monitoring and Alerting Configuration
# CloudWatch dashboards, alarms, and metrics

# ============================================================================
# CloudWatch Dashboard
# ============================================================================

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-${var.environment}"

  dashboard_body = jsonencode({
    widgets = [
      # EKS Cluster Metrics
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/EKS", "cluster_failed_node_count", { stat = "Average" }],
            [".", "cluster_node_count", { stat = "Average" }]
          ]
          period = 300
          stat   = "Average"
          region = data.aws_region.current.name
          title  = "EKS Cluster Nodes"
          yAxis = {
            left = {
              min = 0
            }
          }
        }
      },
      # RDS Metrics
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", { stat = "Average", dimensions = { DBInstanceIdentifier = aws_db_instance.main.identifier } }],
            [".", "DatabaseConnections", { stat = "Average", dimensions = { DBInstanceIdentifier = aws_db_instance.main.identifier } }],
            [".", "FreeableMemory", { stat = "Average", dimensions = { DBInstanceIdentifier = aws_db_instance.main.identifier } }],
            [".", "ReadLatency", { stat = "Average", dimensions = { DBInstanceIdentifier = aws_db_instance.main.identifier } }],
            [".", "WriteLatency", { stat = "Average", dimensions = { DBInstanceIdentifier = aws_db_instance.main.identifier } }]
          ]
          period = 300
          stat   = "Average"
          region = data.aws_region.current.name
          title  = "RDS Performance"
        }
      },
      # ElastiCache Metrics
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ElastiCache", "CPUUtilization", { stat = "Average", dimensions = { ReplicationGroupId = aws_elasticache_replication_group.main.id } }],
            [".", "DatabaseMemoryUsagePercentage", { stat = "Average", dimensions = { ReplicationGroupId = aws_elasticache_replication_group.main.id } }],
            [".", "NetworkBytesIn", { stat = "Sum", dimensions = { ReplicationGroupId = aws_elasticache_replication_group.main.id } }],
            [".", "NetworkBytesOut", { stat = "Sum", dimensions = { ReplicationGroupId = aws_elasticache_replication_group.main.id } }]
          ]
          period = 300
          stat   = "Average"
          region = data.aws_region.current.name
          title  = "ElastiCache Performance"
        }
      },
      # ALB Metrics
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", { stat = "Sum", dimensions = { LoadBalancer = aws_lb.main.arn_suffix } }],
            [".", "TargetResponseTime", { stat = "Average", dimensions = { LoadBalancer = aws_lb.main.arn_suffix } }],
            [".", "HTTPCode_Target_2XX_Count", { stat = "Sum", dimensions = { LoadBalancer = aws_lb.main.arn_suffix } }],
            [".", "HTTPCode_Target_4XX_Count", { stat = "Sum", dimensions = { LoadBalancer = aws_lb.main.arn_suffix } }],
            [".", "HTTPCode_Target_5XX_Count", { stat = "Sum", dimensions = { LoadBalancer = aws_lb.main.arn_suffix } }]
          ]
          period = 300
          stat   = "Average"
          region = data.aws_region.current.name
          title  = "Application Load Balancer"
        }
      },
      # CloudFront Metrics
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/CloudFront", "Requests", { stat = "Sum", dimensions = { DistributionId = aws_cloudfront_distribution.main.id } }],
            [".", "BytesDownloaded", { stat = "Sum", dimensions = { DistributionId = aws_cloudfront_distribution.main.id } }],
            [".", "4xxErrorRate", { stat = "Average", dimensions = { DistributionId = aws_cloudfront_distribution.main.id } }],
            [".", "5xxErrorRate", { stat = "Average", dimensions = { DistributionId = aws_cloudfront_distribution.main.id } }]
          ]
          period = 300
          stat   = "Average"
          region = "us-east-1"
          title  = "CloudFront Distribution"
        }
      }
    ]
  })
}

# ============================================================================
# CloudWatch Alarms - EKS
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "eks_cluster_failed_nodes" {
  alarm_name          = "${var.project_name}-eks-failed-nodes"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "cluster_failed_node_count"
  namespace           = "AWS/EKS"
  period              = 300
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "EKS cluster has failed nodes"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    ClusterName = aws_eks_cluster.main.name
  }

  tags = local.common_tags
}

# ============================================================================
# CloudWatch Alarms - RDS
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  alarm_name          = "${var.project_name}-rds-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = local.monitoring_thresholds.cpu_high
  alarm_description   = "RDS CPU utilization is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "rds_memory_low" {
  alarm_name          = "${var.project_name}-rds-memory-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "FreeableMemory"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 1000000000 # 1GB in bytes
  alarm_description   = "RDS freeable memory is too low"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "rds_storage_low" {
  alarm_name          = "${var.project_name}-rds-storage-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 10000000000 # 10GB in bytes
  alarm_description   = "RDS free storage space is too low"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "rds_connections_high" {
  alarm_name          = "${var.project_name}-rds-connections-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = local.monitoring_thresholds.connection_count_high
  alarm_description   = "RDS database connections are too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "rds_read_latency_high" {
  alarm_name          = "${var.project_name}-rds-read-latency-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ReadLatency"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 0.1 # 100ms
  alarm_description   = "RDS read latency is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "rds_write_latency_high" {
  alarm_name          = "${var.project_name}-rds-write-latency-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "WriteLatency"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 0.1 # 100ms
  alarm_description   = "RDS write latency is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.identifier
  }

  tags = local.common_tags
}

# ============================================================================
# CloudWatch Alarms - ElastiCache
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "redis_cpu_high" {
  alarm_name          = "${var.project_name}-redis-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = local.monitoring_thresholds.cpu_high
  alarm_description   = "Redis CPU utilization is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "redis_memory_high" {
  alarm_name          = "${var.project_name}-redis-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Average"
  threshold           = local.monitoring_thresholds.memory_high
  alarm_description   = "Redis memory usage is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "redis_evictions" {
  alarm_name          = "${var.project_name}-redis-evictions"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Evictions"
  namespace           = "AWS/ElastiCache"
  period              = 300
  statistic           = "Sum"
  threshold           = 1000
  alarm_description   = "Redis is evicting keys"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.main.id
  }

  tags = local.common_tags
}

# ============================================================================
# CloudWatch Alarms - ALB
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "alb_target_response_time_high" {
  alarm_name          = "${var.project_name}-alb-response-time-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Average"
  threshold           = local.monitoring_thresholds.latency_high / 1000 # Convert to seconds
  alarm_description   = "ALB target response time is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "alb_5xx_errors_high" {
  alarm_name          = "${var.project_name}-alb-5xx-errors-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "ALB 5xx errors are too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "alb_unhealthy_targets" {
  alarm_name          = "${var.project_name}-alb-unhealthy-targets"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "UnHealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "ALB has unhealthy targets"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    TargetGroup  = aws_lb_target_group.main.arn_suffix
    LoadBalancer = aws_lb.main.arn_suffix
  }

  tags = local.common_tags
}

# ============================================================================
# CloudWatch Alarms - CloudFront
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "cloudfront_5xx_error_rate_high" {
  alarm_name          = "${var.project_name}-cloudfront-5xx-error-rate-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "5xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = 300
  statistic           = "Average"
  threshold           = local.monitoring_thresholds.error_rate_high
  alarm_description   = "CloudFront 5xx error rate is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    DistributionId = aws_cloudfront_distribution.main.id
  }

  tags = local.common_tags
}

# ============================================================================
# CloudWatch Log Metric Filters
# ============================================================================

resource "aws_cloudwatch_log_metric_filter" "eks_api_errors" {
  name           = "${var.project_name}-eks-api-errors"
  log_group_name = aws_cloudwatch_log_group.eks_cluster.name
  pattern        = "[time, request_id, ...] \"error\""

  metric_transformation {
    name      = "EKSAPIErrors"
    namespace = "${var.project_name}/EKS"
    value     = "1"
    default_value = 0
  }
}

resource "aws_cloudwatch_metric_alarm" "eks_api_errors_high" {
  alarm_name          = "${var.project_name}-eks-api-errors-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "EKSAPIErrors"
  namespace           = "${var.project_name}/EKS"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "EKS API errors are too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  tags = local.common_tags
}

# ============================================================================
# CloudWatch Composite Alarms
# ============================================================================

resource "aws_cloudwatch_composite_alarm" "system_health" {
  alarm_name          = "${var.project_name}-system-health"
  alarm_description   = "Composite alarm for overall system health"
  actions_enabled     = true
  alarm_actions       = [aws_sns_topic.alerts.arn]

  alarm_rule = join(" OR ", [
    "ALARM(${aws_cloudwatch_metric_alarm.rds_cpu_high.alarm_name})",
    "ALARM(${aws_cloudwatch_metric_alarm.redis_cpu_high.alarm_name})",
    "ALARM(${aws_cloudwatch_metric_alarm.alb_5xx_errors_high.alarm_name})",
    "ALARM(${aws_cloudwatch_metric_alarm.alb_unhealthy_targets.alarm_name})"
  ])

  tags = local.common_tags
}

# ============================================================================
# CloudWatch Insights Queries
# ============================================================================

resource "aws_cloudwatch_query_definition" "eks_pod_errors" {
  name = "${var.project_name}-eks-pod-errors"

  log_group_names = [
    aws_cloudwatch_log_group.eks_cluster.name
  ]

  query_string = <<-QUERY
    fields @timestamp, @message
    | filter @message like /error/
    | sort @timestamp desc
    | limit 100
  QUERY
}

resource "aws_cloudwatch_query_definition" "rds_slow_queries" {
  name = "${var.project_name}-rds-slow-queries"

  log_group_names = [
    "/aws/rds/instance/${aws_db_instance.main.identifier}/postgresql"
  ]

  query_string = <<-QUERY
    fields @timestamp, @message
    | filter @message like /duration:/
    | parse @message /duration: (?<duration>\d+\.\d+) ms/
    | filter duration > 1000
    | sort duration desc
    | limit 50
  QUERY
}

resource "aws_cloudwatch_query_definition" "redis_slow_commands" {
  name = "${var.project_name}-redis-slow-commands"

  log_group_names = [
    aws_cloudwatch_log_group.redis_slow_log.name
  ]

  query_string = <<-QUERY
    fields @timestamp, @message
    | sort @timestamp desc
    | limit 100
  QUERY
}

# ============================================================================
# CloudWatch Anomaly Detectors
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "rds_cpu_anomaly" {
  count               = var.enable_enhanced_monitoring ? 1 : 0
  alarm_name          = "${var.project_name}-rds-cpu-anomaly"
  comparison_operator = "LessThanLowerOrGreaterThanUpperThreshold"
  evaluation_periods  = 2
  threshold_metric_id = "e1"
  alarm_description   = "RDS CPU utilization anomaly detected"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"

  metric_query {
    id          = "e1"
    expression  = "ANOMALY_DETECTION_BAND(m1)"
    label       = "CPUUtilization (Expected)"
    return_data = true
  }

  metric_query {
    id          = "m1"
    return_data = true
    metric {
      metric_name = "CPUUtilization"
      namespace   = "AWS/RDS"
      period      = 300
      stat        = "Average"
      dimensions = {
        DBInstanceIdentifier = aws_db_instance.main.identifier
      }
    }
  }

  tags = local.common_tags
}

# ============================================================================
# EventBridge Rules for Automated Responses
# ============================================================================

resource "aws_cloudwatch_event_rule" "rds_backup_failed" {
  name        = "${var.project_name}-rds-backup-failed"
  description = "Trigger when RDS backup fails"

  event_pattern = jsonencode({
    source      = ["aws.rds"]
    detail-type = ["RDS DB Instance Event"]
    detail = {
      EventCategories = ["backup"]
      Message = [{
        prefix = "Backup failed"
      }]
    }
  })

  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "rds_backup_failed_sns" {
  rule      = aws_cloudwatch_event_rule.rds_backup_failed.name
  target_id = "SendToSNS"
  arn       = aws_sns_topic.alerts.arn
}

resource "aws_cloudwatch_event_rule" "eks_cluster_error" {
  name        = "${var.project_name}-eks-cluster-error"
  description = "Trigger when EKS cluster has errors"

  event_pattern = jsonencode({
    source      = ["aws.eks"]
    detail-type = ["EKS Cluster State Change"]
    detail = {
      status = ["FAILED"]
    }
  })

  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "eks_cluster_error_sns" {
  rule      = aws_cloudwatch_event_rule.eks_cluster_error.name
  target_id = "SendToSNS"
  arn       = aws_sns_topic.alerts.arn
}