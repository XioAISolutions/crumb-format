# Wave Field LLM - Auto Scaling Configuration
# ASG policies, target tracking, and scaling strategies

# ============================================================================
# RDS Auto Scaling (Storage)
# ============================================================================

# RDS automatically scales storage based on max_allocated_storage in main.tf

# ============================================================================
# ElastiCache Auto Scaling (Not directly supported, handled by cluster size)
# ============================================================================

# ElastiCache Redis cluster size is configured in main.tf
# For scaling, modify redis_num_cache_nodes variable

# ============================================================================
# Application Auto Scaling for ECS (if using ECS)
# ============================================================================

resource "aws_appautoscaling_target" "ecs_target" {
  count              = 0 # Enable when using ECS
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/cluster-name/service-name"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_cpu" {
  count              = 0 # Enable when using ECS
  name               = "${var.project_name}-ecs-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = local.autoscaling_config.target_cpu_utilization
    scale_in_cooldown  = local.autoscaling_config.scale_down_delay
    scale_out_cooldown = local.autoscaling_config.scale_up_delay
  }
}

resource "aws_appautoscaling_policy" "ecs_memory" {
  count              = 0 # Enable when using ECS
  name               = "${var.project_name}-ecs-memory-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }
    target_value       = local.autoscaling_config.target_memory_utilization
    scale_in_cooldown  = local.autoscaling_config.scale_down_delay
    scale_out_cooldown = local.autoscaling_config.scale_up_delay
  }
}

# ============================================================================
# ALB Target Group Auto Scaling
# ============================================================================

resource "aws_appautoscaling_policy" "alb_request_count" {
  count              = 0 # Enable for request-based scaling
  name               = "${var.project_name}-alb-request-count-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.main.arn_suffix}/${aws_lb_target_group.main.arn_suffix}"
    }
    target_value = 1000
  }
}

# ============================================================================
# Scheduled Scaling Actions
# ============================================================================

# Scale up during business hours
resource "aws_appautoscaling_scheduled_action" "scale_up_business_hours" {
  count              = 0 # Enable for scheduled scaling
  name               = "${var.project_name}-scale-up-business-hours"
  service_namespace  = aws_appautoscaling_target.ecs_target[0].service_namespace
  resource_id        = aws_appautoscaling_target.ecs_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[0].scalable_dimension
  schedule           = "cron(0 8 ? * MON-FRI *)" # 8 AM UTC Monday-Friday

  scalable_target_action {
    min_capacity = 5
    max_capacity = 20
  }
}

# Scale down during off-hours
resource "aws_appautoscaling_scheduled_action" "scale_down_off_hours" {
  count              = 0 # Enable for scheduled scaling
  name               = "${var.project_name}-scale-down-off-hours"
  service_namespace  = aws_appautoscaling_target.ecs_target[0].service_namespace
  resource_id        = aws_appautoscaling_target.ecs_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target[0].scalable_dimension
  schedule           = "cron(0 20 ? * MON-FRI *)" # 8 PM UTC Monday-Friday

  scalable_target_action {
    min_capacity = 2
    max_capacity = 10
  }
}

# ============================================================================
# Lambda Auto Scaling (Provisioned Concurrency)
# ============================================================================

resource "aws_appautoscaling_target" "lambda_target" {
  count              = 0 # Enable when using Lambda with provisioned concurrency
  max_capacity       = 100
  min_capacity       = 5
  resource_id        = "function:lambda-function-name:provisioned-concurrency:alias-name"
  scalable_dimension = "lambda:function:ProvisionedConcurrentExecutions"
  service_namespace  = "lambda"
}

resource "aws_appautoscaling_policy" "lambda_scaling" {
  count              = 0 # Enable when using Lambda with provisioned concurrency
  name               = "${var.project_name}-lambda-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.lambda_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.lambda_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.lambda_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "LambdaProvisionedConcurrencyUtilization"
    }
    target_value = 0.7
  }
}

# ============================================================================
# DynamoDB Auto Scaling (if using DynamoDB)
# ============================================================================

resource "aws_appautoscaling_target" "dynamodb_table_read" {
  count              = 0 # Enable when using DynamoDB
  max_capacity       = 100
  min_capacity       = 5
  resource_id        = "table/table-name"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_policy" "dynamodb_table_read_policy" {
  count              = 0 # Enable when using DynamoDB
  name               = "${var.project_name}-dynamodb-read-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_read[0].resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_read[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_read[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }
    target_value = 70.0
  }
}

resource "aws_appautoscaling_target" "dynamodb_table_write" {
  count              = 0 # Enable when using DynamoDB
  max_capacity       = 100
  min_capacity       = 5
  resource_id        = "table/table-name"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}

resource "aws_appautoscaling_policy" "dynamodb_table_write_policy" {
  count              = 0 # Enable when using DynamoDB
  name               = "${var.project_name}-dynamodb-write-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_write[0].resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_write[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_write[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBWriteCapacityUtilization"
    }
    target_value = 70.0
  }
}

# ============================================================================
# EKS Node Group Auto Scaling (Managed by Cluster Autoscaler)
# ============================================================================

# EKS node groups are configured with min/max/desired sizes in main.tf
# Cluster Autoscaler will be installed via Helm in eks-addons.sh
# Node groups will scale based on pod resource requests

# ============================================================================
# Custom Metrics for Auto Scaling
# ============================================================================

# Custom CloudWatch metric for application-specific scaling
resource "aws_cloudwatch_metric_alarm" "custom_metric_scale_up" {
  count               = 0 # Enable for custom metric scaling
  alarm_name          = "${var.project_name}-custom-metric-scale-up"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CustomMetric"
  namespace           = "${var.project_name}/Application"
  period              = 60
  statistic           = "Average"
  threshold           = 100
  alarm_description   = "Scale up based on custom metric"
  alarm_actions       = [] # Add scaling policy ARN

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "custom_metric_scale_down" {
  count               = 0 # Enable for custom metric scaling
  alarm_name          = "${var.project_name}-custom-metric-scale-down"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 5
  metric_name         = "CustomMetric"
  namespace           = "${var.project_name}/Application"
  period              = 60
  statistic           = "Average"
  threshold           = 20
  alarm_description   = "Scale down based on custom metric"
  alarm_actions       = [] # Add scaling policy ARN

  tags = local.common_tags
}

# ============================================================================
# Predictive Scaling (AWS Auto Scaling Plans)
# ============================================================================

# Note: Predictive scaling requires historical data and is configured separately
# through AWS Auto Scaling Plans console or API

# ============================================================================
# Scaling Notifications
# ============================================================================

resource "aws_autoscaling_notification" "scaling_notifications" {
  count = 0 # Enable when using EC2 Auto Scaling Groups

  group_names = [
    # Add ASG names here
  ]

  notifications = [
    "autoscaling:EC2_INSTANCE_LAUNCH",
    "autoscaling:EC2_INSTANCE_TERMINATE",
    "autoscaling:EC2_INSTANCE_LAUNCH_ERROR",
    "autoscaling:EC2_INSTANCE_TERMINATE_ERROR",
  ]

  topic_arn = aws_sns_topic.alerts.arn
}

# ============================================================================
# Scaling Policies Documentation
# ============================================================================

# Scaling Strategy:
# 1. EKS Node Groups: Managed by Cluster Autoscaler based on pod resource requests
# 2. ECS Services: Target tracking based on CPU/Memory utilization
# 3. Lambda: Provisioned concurrency with target tracking
# 4. RDS: Storage auto-scaling enabled, read replicas for read scaling
# 5. ElastiCache: Manual scaling by adjusting cluster size
# 6. ALB: Automatically scales based on traffic
# 7. CloudFront: Automatically scales globally

# Cost Optimization:
# - Use spot instances for non-critical workloads
# - Schedule scaling for predictable traffic patterns
# - Set appropriate cooldown periods to prevent flapping
# - Monitor scaling metrics and adjust thresholds
# - Use reserved instances for baseline capacity

# Best Practices:
# - Test scaling policies in non-production environments
# - Set up proper monitoring and alerting
# - Document scaling thresholds and rationale
# - Review and adjust policies based on actual usage
# - Consider using AWS Auto Scaling Plans for coordinated scaling