#!/usr/bin/env python3
"""
Wave Field LLM - SageMaker Deployment Script
Complete deployment script for deploying models to Amazon SageMaker
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """Handles deployment of Wave Field LLM to SageMaker"""
    
    def __init__(
        self,
        model_path: str,
        role_arn: str,
        region: str = "us-east-1",
        project_name: str = "wavefield-llm"
    ):
        self.model_path = model_path
        self.role_arn = role_arn
        self.region = region
        self.project_name = project_name
        
        # Initialize AWS clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.session = sagemaker.Session(boto_session=boto3.Session(region_name=region))
        
        # Get account ID
        sts_client = boto3.client('sts', region_name=region)
        self.account_id = sts_client.get_caller_identity()['Account']
        
        # S3 bucket for models
        self.bucket = f"{project_name}-models-{self.account_id}"
        
        logger.info(f"Initialized SageMaker deployer for region: {region}")
        logger.info(f"Using S3 bucket: {self.bucket}")
    
    def upload_model_to_s3(self) -> str:
        """Upload model artifacts to S3"""
        logger.info(f"Uploading model from {self.model_path} to S3...")
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_key = f"models/wavefield-llm/{timestamp}/model.tar.gz"
        
        # Create tarball if directory
        if os.path.isdir(self.model_path):
            import tarfile
            tarball_path = f"/tmp/model-{timestamp}.tar.gz"
            
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(self.model_path, arcname=".")
            
            upload_path = tarball_path
        else:
            upload_path = self.model_path
        
        # Upload to S3
        self.s3_client.upload_file(upload_path, self.bucket, s3_key)
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        
        logger.info(f"Model uploaded to: {s3_uri}")
        return s3_uri
    
    def create_model(
        self,
        model_data: str,
        instance_type: str = "ml.g5.xlarge",
        framework_version: str = "2.0.0",
        py_version: str = "py310"
    ) -> HuggingFaceModel:
        """Create SageMaker model"""
        logger.info("Creating SageMaker model...")
        
        model_name = f"{self.project_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create HuggingFace model
        huggingface_model = HuggingFaceModel(
            model_data=model_data,
            role=self.role_arn,
            transformers_version=framework_version,
            pytorch_version="2.0.0",
            py_version=py_version,
            name=model_name,
            sagemaker_session=self.session,
            env={
                'HF_TASK': 'text-generation',
                'MAX_LENGTH': '2048',
                'NUM_RETURN_SEQUENCES': '1',
                'TOP_K': '50',
                'TOP_P': '0.95',
                'DO_SAMPLE': 'true',
                'TEMPERATURE': '0.7',
            }
        )
        
        logger.info(f"Model created: {model_name}")
        return huggingface_model
    
    def deploy_endpoint(
        self,
        model: HuggingFaceModel,
        endpoint_name: Optional[str] = None,
        instance_type: str = "ml.g5.xlarge",
        instance_count: int = 1,
        enable_data_capture: bool = True,
        enable_autoscaling: bool = True,
        min_capacity: int = 1,
        max_capacity: int = 5
    ) -> Predictor:
        """Deploy model to SageMaker endpoint"""
        if endpoint_name is None:
            endpoint_name = f"{self.project_name}-endpoint"
        
        logger.info(f"Deploying endpoint: {endpoint_name}")
        logger.info(f"Instance type: {instance_type}, count: {instance_count}")
        
        # Configure data capture
        data_capture_config = None
        if enable_data_capture:
            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=100,
                destination_s3_uri=f"s3://{self.bucket}/data-capture/{endpoint_name}"
            )
        
        # Deploy endpoint
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            data_capture_config=data_capture_config,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
            wait=True
        )
        
        logger.info(f"Endpoint deployed successfully: {endpoint_name}")
        
        # Configure autoscaling
        if enable_autoscaling:
            self._configure_autoscaling(
                endpoint_name,
                min_capacity,
                max_capacity
            )
        
        return predictor
    
    def _configure_autoscaling(
        self,
        endpoint_name: str,
        min_capacity: int,
        max_capacity: int
    ):
        """Configure autoscaling for endpoint"""
        logger.info("Configuring autoscaling...")
        
        autoscaling_client = boto3.client('application-autoscaling', region_name=self.region)
        
        # Register scalable target
        resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
        
        try:
            autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity
            )
            
            # Configure target tracking scaling policy
            autoscaling_client.put_scaling_policy(
                PolicyName=f"{endpoint_name}-scaling-policy",
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': 70.0,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleInCooldown': 300,
                    'ScaleOutCooldown': 60
                }
            )
            
            logger.info("Autoscaling configured successfully")
        except Exception as e:
            logger.warning(f"Failed to configure autoscaling: {e}")
    
    def create_multi_model_endpoint(
        self,
        model_data_prefix: str,
        endpoint_name: Optional[str] = None,
        instance_type: str = "ml.g5.2xlarge",
        instance_count: int = 2
    ):
        """Create multi-model endpoint for serving multiple models"""
        if endpoint_name is None:
            endpoint_name = f"{self.project_name}-multi-model-endpoint"
        
        logger.info(f"Creating multi-model endpoint: {endpoint_name}")
        
        from sagemaker.multidatamodel import MultiDataModel
        
        model_name = f"{self.project_name}-multi-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create multi-model
        multi_model = MultiDataModel(
            name=model_name,
            model_data_prefix=model_data_prefix,
            role=self.role_arn,
            sagemaker_session=self.session
        )
        
        # Deploy
        predictor = multi_model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            wait=True
        )
        
        logger.info(f"Multi-model endpoint deployed: {endpoint_name}")
        return predictor
    
    def create_async_endpoint(
        self,
        model: HuggingFaceModel,
        endpoint_name: Optional[str] = None,
        instance_type: str = "ml.g5.xlarge",
        max_concurrent_invocations: int = 10
    ):
        """Create async inference endpoint for batch processing"""
        if endpoint_name is None:
            endpoint_name = f"{self.project_name}-async-endpoint"
        
        logger.info(f"Creating async endpoint: {endpoint_name}")
        
        from sagemaker.async_inference import AsyncInferenceConfig
        
        async_config = AsyncInferenceConfig(
            output_path=f"s3://{self.bucket}/async-inference/output",
            max_concurrent_invocations_per_instance=max_concurrent_invocations,
            notification_config={
                "SuccessTopic": f"arn:aws:sns:{self.region}:{self.account_id}:{self.project_name}-async-success",
                "ErrorTopic": f"arn:aws:sns:{self.region}:{self.account_id}:{self.project_name}-async-error"
            }
        )
        
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            async_inference_config=async_config,
            wait=True
        )
        
        logger.info(f"Async endpoint deployed: {endpoint_name}")
        return predictor
    
    def create_serverless_endpoint(
        self,
        model: HuggingFaceModel,
        endpoint_name: Optional[str] = None,
        memory_size: int = 4096,
        max_concurrency: int = 10
    ):
        """Create serverless inference endpoint"""
        if endpoint_name is None:
            endpoint_name = f"{self.project_name}-serverless-endpoint"
        
        logger.info(f"Creating serverless endpoint: {endpoint_name}")
        
        from sagemaker.serverless import ServerlessInferenceConfig
        
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=memory_size,
            max_concurrency=max_concurrency
        )
        
        predictor = model.deploy(
            serverless_inference_config=serverless_config,
            endpoint_name=endpoint_name,
            wait=True
        )
        
        logger.info(f"Serverless endpoint deployed: {endpoint_name}")
        return predictor
    
    def test_endpoint(self, endpoint_name: str, test_input: Dict):
        """Test deployed endpoint"""
        logger.info(f"Testing endpoint: {endpoint_name}")
        
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=self.session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        response = predictor.predict(test_input)
        logger.info(f"Test response: {response}")
        return response
    
    def delete_endpoint(self, endpoint_name: str, delete_model: bool = True):
        """Delete SageMaker endpoint"""
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint deleted: {endpoint_name}")
            
            if delete_model:
                # Get endpoint config
                endpoint_config = self.sagemaker_client.describe_endpoint(
                    EndpointName=endpoint_name
                )['EndpointConfigName']
                
                # Delete endpoint config
                self.sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=endpoint_config
                )
                logger.info(f"Endpoint config deleted: {endpoint_config}")
        except Exception as e:
            logger.error(f"Error deleting endpoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="Deploy Wave Field LLM to SageMaker")
    parser.add_argument("--model-path", required=True, help="Path to model artifacts")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--endpoint-name", help="Custom endpoint name")
    parser.add_argument("--instance-type", default="ml.g5.xlarge", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--deployment-type", default="standard", 
                       choices=["standard", "multi-model", "async", "serverless"],
                       help="Deployment type")
    parser.add_argument("--enable-autoscaling", action="store_true", help="Enable autoscaling")
    parser.add_argument("--min-capacity", type=int, default=1, help="Min autoscaling capacity")
    parser.add_argument("--max-capacity", type=int, default=5, help="Max autoscaling capacity")
    parser.add_argument("--test", action="store_true", help="Test endpoint after deployment")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = SageMakerDeployer(
        model_path=args.model_path,
        role_arn=args.role_arn,
        region=args.region
    )
    
    # Upload model
    model_data = deployer.upload_model_to_s3()
    
    # Create model
    model = deployer.create_model(model_data, instance_type=args.instance_type)
    
    # Deploy based on type
    if args.deployment_type == "standard":
        predictor = deployer.deploy_endpoint(
            model,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            enable_autoscaling=args.enable_autoscaling,
            min_capacity=args.min_capacity,
            max_capacity=args.max_capacity
        )
    elif args.deployment_type == "multi-model":
        predictor = deployer.create_multi_model_endpoint(
            model_data_prefix=f"s3://{deployer.bucket}/models/",
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count
        )
    elif args.deployment_type == "async":
        predictor = deployer.create_async_endpoint(
            model,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type
        )
    elif args.deployment_type == "serverless":
        predictor = deployer.create_serverless_endpoint(
            model,
            endpoint_name=args.endpoint_name
        )
    
    # Test endpoint
    if args.test:
        test_input = {
            "inputs": "Once upon a time",
            "parameters": {
                "max_length": 100,
                "temperature": 0.7
            }
        }
        deployer.test_endpoint(predictor.endpoint_name, test_input)
    
    logger.info("Deployment completed successfully!")
    logger.info(f"Endpoint name: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()

# Made with Bob
