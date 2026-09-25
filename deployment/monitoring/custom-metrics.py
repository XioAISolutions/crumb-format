#!/usr/bin/env python3
"""
Custom Metrics Exporter for Wave Field LLM
Exports application-specific business and operational metrics to Prometheus
"""

import os
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

import psycopg2
import redis
from prometheus_client import (
    start_http_server,
    Gauge,
    Counter,
    Histogram,
    Summary,
    Info,
    CollectorRegistry,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics exporter"""
    port: int = int(os.getenv('METRICS_PORT', '8000'))
    scrape_interval: int = int(os.getenv('SCRAPE_INTERVAL', '15'))
    
    # Database configuration
    db_host: str = os.getenv('DB_HOST', 'localhost')
    db_port: int = int(os.getenv('DB_PORT', '5432'))
    db_name: str = os.getenv('DB_NAME', 'wavefield')
    db_user: str = os.getenv('DB_USER', 'wavefield')
    db_password: str = os.getenv('DB_PASSWORD', '')
    
    # Redis configuration
    redis_host: str = os.getenv('REDIS_HOST', 'localhost')
    redis_port: int = int(os.getenv('REDIS_PORT', '6379'))
    redis_db: int = int(os.getenv('REDIS_DB', '0'))
    redis_password: Optional[str] = os.getenv('REDIS_PASSWORD')


class WaveFieldMetricsCollector:
    """Collects custom metrics for Wave Field LLM"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = CollectorRegistry()
        
        # Initialize database connection
        self.db_conn = None
        self.redis_client = None
        
        # Define metrics
        self._define_metrics()
        
    def _define_metrics(self):
        """Define all custom metrics"""
        
        # Business metrics
        self.total_requests = Counter(
            'wavefield_business_total_requests',
            'Total number of inference requests',
            ['model_version', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.total_tokens = Counter(
            'wavefield_business_total_tokens',
            'Total tokens processed',
            ['model_version', 'type'],  # type: input/output
            registry=self.registry
        )
        
        self.revenue_per_request = Histogram(
            'wavefield_business_revenue_per_request_dollars',
            'Revenue per request in dollars',
            ['model_version', 'tier'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.cost_per_request = Histogram(
            'wavefield_business_cost_per_request_dollars',
            'Cost per request in dollars',
            ['model_version', 'resource_type'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'wavefield_business_active_users',
            'Number of active users',
            ['time_window'],  # 1h, 24h, 7d
            registry=self.registry
        )
        
        self.user_tier_distribution = Gauge(
            'wavefield_business_user_tier_distribution',
            'Distribution of users by tier',
            ['tier'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'wavefield_model_accuracy_score',
            'Model accuracy score',
            ['model_version', 'metric_type'],
            registry=self.registry
        )
        
        self.model_latency_by_length = Histogram(
            'wavefield_model_latency_by_input_length_seconds',
            'Inference latency by input length',
            ['model_version', 'length_bucket'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.tokens_per_second = Gauge(
            'wavefield_model_tokens_per_second',
            'Token generation rate',
            ['model_version', 'gpu_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_size_bytes = Gauge(
            'wavefield_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],  # prompt, kv, result
            registry=self.registry
        )
        
        self.cache_evictions = Counter(
            'wavefield_cache_evictions_total',
            'Total cache evictions',
            ['cache_type', 'reason'],
            registry=self.registry
        )
        
        self.cache_hit_latency = Histogram(
            'wavefield_cache_hit_latency_seconds',
            'Cache hit latency',
            ['cache_type'],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            registry=self.registry
        )
        
        # Queue metrics
        self.queue_depth = Gauge(
            'wavefield_queue_depth',
            'Current queue depth',
            ['queue_name', 'priority'],
            registry=self.registry
        )
        
        self.queue_wait_time = Histogram(
            'wavefield_queue_wait_time_seconds',
            'Time requests spend in queue',
            ['queue_name', 'priority'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.queue_processing_time = Histogram(
            'wavefield_queue_processing_time_seconds',
            'Time to process queued requests',
            ['queue_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        # Resource utilization
        self.gpu_memory_allocated = Gauge(
            'wavefield_gpu_memory_allocated_bytes',
            'GPU memory allocated',
            ['gpu_id', 'model_version'],
            registry=self.registry
        )
        
        self.batch_size_actual = Histogram(
            'wavefield_batch_size_actual',
            'Actual batch sizes used',
            ['model_version'],
            buckets=[1, 2, 4, 8, 16, 32, 64, 128],
            registry=self.registry
        )
        
        # SLA metrics
        self.sla_compliance = Gauge(
            'wavefield_sla_compliance_percent',
            'SLA compliance percentage',
            ['sla_type', 'tier'],
            registry=self.registry
        )
        
        self.sla_violations = Counter(
            'wavefield_sla_violations_total',
            'Total SLA violations',
            ['sla_type', 'tier', 'severity'],
            registry=self.registry
        )
        
        # Error tracking
        self.error_rate_by_type = Gauge(
            'wavefield_error_rate_by_type',
            'Error rate by error type',
            ['error_type', 'model_version'],
            registry=self.registry
        )
        
        self.retry_attempts = Histogram(
            'wavefield_retry_attempts',
            'Number of retry attempts',
            ['endpoint', 'error_type'],
            buckets=[0, 1, 2, 3, 5, 10],
            registry=self.registry
        )
        
        # Cost tracking
        self.hourly_cost = Gauge(
            'wavefield_hourly_cost_dollars',
            'Estimated hourly cost',
            ['resource_type', 'region'],
            registry=self.registry
        )
        
        self.cost_per_1k_tokens = Gauge(
            'wavefield_cost_per_1k_tokens_dollars',
            'Cost per 1000 tokens',
            ['model_version', 'token_type'],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'wavefield_system',
            'System information',
            registry=self.registry
        )
        
    def connect_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                database=self.config.db_name,
                user=self.config.db_user,
                password=self.config.db_password
            )
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            
    def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            
    def collect_business_metrics(self):
        """Collect business metrics from database"""
        if not self.db_conn:
            return
            
        try:
            cursor = self.db_conn.cursor()
            
            # Active users in different time windows
            for window, hours in [('1h', 1), ('24h', 24), ('7d', 168)]:
                cursor.execute("""
                    SELECT COUNT(DISTINCT user_id)
                    FROM requests
                    WHERE timestamp > NOW() - INTERVAL '%s hours'
                """, (hours,))
                count = cursor.fetchone()[0]
                self.active_users.labels(time_window=window).set(count)
            
            # User tier distribution
            cursor.execute("""
                SELECT tier, COUNT(*)
                FROM users
                WHERE active = true
                GROUP BY tier
            """)
            for tier, count in cursor.fetchall():
                self.user_tier_distribution.labels(tier=tier).set(count)
            
            # Cost and revenue metrics (last hour)
            cursor.execute("""
                SELECT 
                    model_version,
                    tier,
                    AVG(revenue) as avg_revenue,
                    AVG(cost) as avg_cost
                FROM requests
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY model_version, tier
            """)
            for model_version, tier, revenue, cost in cursor.fetchall():
                if revenue:
                    self.revenue_per_request.labels(
                        model_version=model_version,
                        tier=tier
                    ).observe(float(revenue))
                if cost:
                    self.cost_per_request.labels(
                        model_version=model_version,
                        resource_type='compute'
                    ).observe(float(cost))
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")
            self.db_conn.rollback()
            
    def collect_cache_metrics(self):
        """Collect cache metrics from Redis"""
        if not self.redis_client:
            return
            
        try:
            # Cache sizes
            for cache_type in ['prompt', 'kv', 'result']:
                key_pattern = f"{cache_type}:*"
                keys = self.redis_client.keys(key_pattern)
                total_size = sum(
                    self.redis_client.memory_usage(key) or 0
                    for key in keys[:1000]  # Sample first 1000 keys
                )
                self.cache_size_bytes.labels(cache_type=cache_type).set(total_size)
            
            # Queue depths
            for queue_name in ['high', 'normal', 'low']:
                for priority in ['urgent', 'normal', 'batch']:
                    key = f"queue:{queue_name}:{priority}"
                    depth = self.redis_client.llen(key)
                    self.queue_depth.labels(
                        queue_name=queue_name,
                        priority=priority
                    ).set(depth)
                    
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
            
    def collect_sla_metrics(self):
        """Collect SLA compliance metrics"""
        if not self.db_conn:
            return
            
        try:
            cursor = self.db_conn.cursor()
            
            # SLA compliance by tier (last hour)
            cursor.execute("""
                SELECT 
                    tier,
                    sla_type,
                    (COUNT(*) FILTER (WHERE met_sla = true)::float / COUNT(*)) * 100 as compliance
                FROM requests
                WHERE timestamp > NOW() - INTERVAL '1 hour'
                GROUP BY tier, sla_type
            """)
            for tier, sla_type, compliance in cursor.fetchall():
                self.sla_compliance.labels(
                    sla_type=sla_type,
                    tier=tier
                ).set(float(compliance))
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error collecting SLA metrics: {e}")
            
    def collect_cost_metrics(self):
        """Collect cost metrics"""
        # This would integrate with cloud provider APIs
        # Placeholder implementation
        try:
            # Example: GPU costs
            self.hourly_cost.labels(
                resource_type='gpu_a100',
                region='us-east-1'
            ).set(3.06)  # Example rate
            
            # Cost per 1k tokens
            self.cost_per_1k_tokens.labels(
                model_version='v1.0',
                token_type='input'
            ).set(0.0015)
            
            self.cost_per_1k_tokens.labels(
                model_version='v1.0',
                token_type='output'
            ).set(0.002)
            
        except Exception as e:
            logger.error(f"Error collecting cost metrics: {e}")
            
    def collect_all_metrics(self):
        """Collect all custom metrics"""
        logger.info("Collecting metrics...")
        
        self.collect_business_metrics()
        self.collect_cache_metrics()
        self.collect_sla_metrics()
        self.collect_cost_metrics()
        
        # Update system info
        self.system_info.info({
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            'cluster': os.getenv('CLUSTER_NAME', 'wavefield-prod')
        })
        
        logger.info("Metrics collection complete")
        
    def run(self):
        """Run the metrics collector"""
        logger.info(f"Starting metrics server on port {self.config.port}")
        
        # Connect to data sources
        self.connect_database()
        self.connect_redis()
        
        # Start Prometheus HTTP server
        start_http_server(self.config.port, registry=self.registry)
        
        # Collect metrics periodically
        while True:
            try:
                self.collect_all_metrics()
                time.sleep(self.config.scrape_interval)
            except KeyboardInterrupt:
                logger.info("Shutting down metrics collector")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(self.config.scrape_interval)


def main():
    """Main entry point"""
    config = MetricsConfig()
    collector = WaveFieldMetricsCollector(config)
    collector.run()


if __name__ == '__main__':
    main()

# Made with Bob
