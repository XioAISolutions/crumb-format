#!/bin/bash
set -euo pipefail

# Wave Field LLM - AWS Deployment Script
# Complete AWS infrastructure and application deployment

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT="production"
REGION="us-east-1"
CLUSTER_NAME=""
VPC_CIDR="10.0.0.0/16"
ENABLE_CDN=true
ENABLE_WAF=true
DB_INSTANCE_CLASS="db.r5.large"
REDIS_NODE_TYPE="cache.r5.large"
EKS_NODE_TYPE="m5.xlarge"
EKS_NODE_COUNT=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}\n"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Wave Field LLM to AWS

Options:
    --environment ENV       Environment (production, staging, development)
    --region REGION         AWS region (default: us-east-1)
    --cluster-name NAME     EKS cluster name (default: wavefield-ENV)
    --vpc-cidr CIDR         VPC CIDR block (default: 10.0.0.0/16)
    --no-cdn                Disable CloudFront CDN
    --no-waf                Disable AWS WAF
    --db-instance CLASS     RDS instance class (default: db.r5.large)
    --redis-node TYPE       ElastiCache node type (default: cache.r5.large)
    --eks-node-type TYPE    EKS node instance type (default: m5.xlarge)
    --eks-node-count N      Number of EKS nodes (default: 3)
    -h, --help              Show this help

Examples:
    # Production deployment
    $0 --environment production --region us-east-1

    # Staging with smaller instances
    $0 --environment staging --db-instance db.t3.medium --redis-node cache.t3.medium

EOF
    exit 1
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment) ENVIRONMENT="$2"; shift 2 ;;
            --region) REGION="$2"; shift 2 ;;
            --cluster-name) CLUSTER_NAME="$2"; shift 2 ;;
            --vpc-cidr) VPC_CIDR="$2"; shift 2 ;;
            --no-cdn) ENABLE_CDN=false; shift ;;
            --no-waf) ENABLE_WAF=false; shift ;;
            --db-instance) DB_INSTANCE_CLASS="$2"; shift 2 ;;
            --redis-node) REDIS_NODE_TYPE="$2"; shift 2 ;;
            --eks-node-type) EKS_NODE_TYPE="$2"; shift 2 ;;
            --eks-node-count) EKS_NODE_COUNT="$2"; shift 2 ;;
            -h|--help) usage ;;
            *) log_error "Unknown option: $1"; usage ;;
        esac
    done

    [[ -z "$CLUSTER_NAME" ]] && CLUSTER_NAME="wavefield-${ENVIRONMENT}"
}

check_aws_cli() {
    log_step "Checking AWS CLI"

    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not installed"
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi

    log_success "AWS CLI configured"
}

create_vpc() {
    log_step "Creating VPC and Networking"

    local vpc_id=$(aws ec2 describe-vpcs \
        --filters "Name=tag:Name,Values=wavefield-${ENVIRONMENT}" \
        --query 'Vpcs[0].VpcId' --output text 2>/dev/null || echo "")

    if [[ "$vpc_id" != "None" ]] && [[ -n "$vpc_id" ]]; then
        log_info "VPC already exists: $vpc_id"
        export VPC_ID="$vpc_id"
        return 0
    fi

    log_info "Creating VPC with CIDR ${VPC_CIDR}..."
    vpc_id=$(aws ec2 create-vpc \
        --cidr-block "$VPC_CIDR" \
        --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=wavefield-${ENVIRONMENT}}]" \
        --query 'Vpc.VpcId' --output text)

    export VPC_ID="$vpc_id"

    # Enable DNS
    aws ec2 modify-vpc-attribute --vpc-id "$vpc_id" --enable-dns-hostnames
    aws ec2 modify-vpc-attribute --vpc-id "$vpc_id" --enable-dns-support

    # Create Internet Gateway
    log_info "Creating Internet Gateway..."
    local igw_id=$(aws ec2 create-internet-gateway \
        --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=wavefield-${ENVIRONMENT}-igw}]" \
        --query 'InternetGateway.InternetGatewayId' --output text)
    
    aws ec2 attach-internet-gateway --vpc-id "$vpc_id" --internet-gateway-id "$igw_id"

    # Create subnets
    log_info "Creating subnets..."
    local az_count=3
    local subnet_ids=()

    for i in $(seq 0 $((az_count - 1))); do
        local az="${REGION}$(echo {a..z} | cut -d' ' -f$((i+1)))"
        local cidr="10.0.$((i * 16)).0/20"
        
        local subnet_id=$(aws ec2 create-subnet \
            --vpc-id "$vpc_id" \
            --cidr-block "$cidr" \
            --availability-zone "$az" \
            --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=wavefield-${ENVIRONMENT}-public-${az}}]" \
            --query 'Subnet.SubnetId' --output text 2>/dev/null || echo "")
        
        if [[ -n "$subnet_id" ]]; then
            subnet_ids+=("$subnet_id")
            aws ec2 modify-subnet-attribute --subnet-id "$subnet_id" --map-public-ip-on-launch
        fi
    done

    export SUBNET_IDS="${subnet_ids[*]}"

    log_success "VPC and networking created"
}

create_security_groups() {
    log_step "Creating Security Groups"

    # EKS cluster security group
    log_info "Creating EKS security group..."
    local eks_sg_id=$(aws ec2 create-security-group \
        --group-name "wavefield-${ENVIRONMENT}-eks-sg" \
        --description "Security group for EKS cluster" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
            --filters "Name=group-name,Values=wavefield-${ENVIRONMENT}-eks-sg" \
            --query 'SecurityGroups[0].GroupId' --output text)

    export EKS_SG_ID="$eks_sg_id"

    # RDS security group
    log_info "Creating RDS security group..."
    local rds_sg_id=$(aws ec2 create-security-group \
        --group-name "wavefield-${ENVIRONMENT}-rds-sg" \
        --description "Security group for RDS" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
            --filters "Name=group-name,Values=wavefield-${ENVIRONMENT}-rds-sg" \
            --query 'SecurityGroups[0].GroupId' --output text)

    # Allow EKS to access RDS
    aws ec2 authorize-security-group-ingress \
        --group-id "$rds_sg_id" \
        --protocol tcp \
        --port 5432 \
        --source-group "$eks_sg_id" 2>/dev/null || true

    export RDS_SG_ID="$rds_sg_id"

    log_success "Security groups created"
}

create_rds() {
    log_step "Creating RDS Database"

    local db_identifier="wavefield-${ENVIRONMENT}"

    # Check if DB exists
    if aws rds describe-db-instances --db-instance-identifier "$db_identifier" &> /dev/null; then
        log_info "RDS instance already exists"
        return 0
    fi

    # Create DB subnet group
    log_info "Creating DB subnet group..."
    aws rds create-db-subnet-group \
        --db-subnet-group-name "wavefield-${ENVIRONMENT}-subnet-group" \
        --db-subnet-group-description "Subnet group for Wave Field LLM" \
        --subnet-ids $SUBNET_IDS 2>/dev/null || true

    # Create RDS instance
    log_info "Creating RDS instance (this may take 10-15 minutes)..."
    aws rds create-db-instance \
        --db-instance-identifier "$db_identifier" \
        --db-instance-class "$DB_INSTANCE_CLASS" \
        --engine postgres \
        --engine-version 15.3 \
        --master-username wavefield \
        --master-user-password "$(openssl rand -base64 32)" \
        --allocated-storage 100 \
        --storage-type gp3 \
        --vpc-security-group-ids "$RDS_SG_ID" \
        --db-subnet-group-name "wavefield-${ENVIRONMENT}-subnet-group" \
        --backup-retention-period 7 \
        --preferred-backup-window "03:00-04:00" \
        --preferred-maintenance-window "mon:04:00-mon:05:00" \
        --multi-az \
        --storage-encrypted \
        --enable-performance-insights \
        --tags "Key=Environment,Value=${ENVIRONMENT}" "Key=Application,Value=wavefield-llm"

    # Wait for DB to be available
    log_info "Waiting for RDS instance to be available..."
    aws rds wait db-instance-available --db-instance-identifier "$db_identifier"

    log_success "RDS database created"
}

create_elasticache() {
    log_step "Creating ElastiCache Redis"

    local cache_cluster_id="wavefield-${ENVIRONMENT}"

    # Check if cluster exists
    if aws elasticache describe-cache-clusters --cache-cluster-id "$cache_cluster_id" &> /dev/null; then
        log_info "ElastiCache cluster already exists"
        return 0
    fi

    # Create cache subnet group
    log_info "Creating cache subnet group..."
    aws elasticache create-cache-subnet-group \
        --cache-subnet-group-name "wavefield-${ENVIRONMENT}-cache-subnet" \
        --cache-subnet-group-description "Cache subnet for Wave Field LLM" \
        --subnet-ids $SUBNET_IDS 2>/dev/null || true

    # Create Redis cluster
    log_info "Creating Redis cluster..."
    aws elasticache create-cache-cluster \
        --cache-cluster-id "$cache_cluster_id" \
        --cache-node-type "$REDIS_NODE_TYPE" \
        --engine redis \
        --engine-version 7.0 \
        --num-cache-nodes 1 \
        --cache-subnet-group-name "wavefield-${ENVIRONMENT}-cache-subnet" \
        --security-group-ids "$EKS_SG_ID" \
        --tags "Key=Environment,Value=${ENVIRONMENT}" "Key=Application,Value=wavefield-llm"

    log_success "ElastiCache Redis created"
}

create_s3_buckets() {
    log_step "Creating S3 Buckets"

    local model_bucket="wavefield-${ENVIRONMENT}-models-${REGION}"
    local logs_bucket="wavefield-${ENVIRONMENT}-logs-${REGION}"

    # Create model bucket
    log_info "Creating model bucket..."
    aws s3 mb "s3://${model_bucket}" --region "$REGION" 2>/dev/null || true
    aws s3api put-bucket-versioning \
        --bucket "$model_bucket" \
        --versioning-configuration Status=Enabled
    aws s3api put-bucket-encryption \
        --bucket "$model_bucket" \
        --server-side-encryption-configuration '{
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        }'

    # Create logs bucket
    log_info "Creating logs bucket..."
    aws s3 mb "s3://${logs_bucket}" --region "$REGION" 2>/dev/null || true
    aws s3api put-bucket-lifecycle-configuration \
        --bucket "$logs_bucket" \
        --lifecycle-configuration '{
            "Rules": [{
                "Id": "DeleteOldLogs",
                "Status": "Enabled",
                "Expiration": {"Days": 90}
            }]
        }'

    export MODEL_BUCKET="$model_bucket"
    export LOGS_BUCKET="$logs_bucket"

    log_success "S3 buckets created"
}

create_ecr_repository() {
    log_step "Creating ECR Repository"

    local repo_name="wavefield-llm"

    if aws ecr describe-repositories --repository-names "$repo_name" &> /dev/null; then
        log_info "ECR repository already exists"
        return 0
    fi

    log_info "Creating ECR repository..."
    aws ecr create-repository \
        --repository-name "$repo_name" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        --tags "Key=Environment,Value=${ENVIRONMENT}"

    # Set lifecycle policy
    aws ecr put-lifecycle-policy \
        --repository-name "$repo_name" \
        --lifecycle-policy-text '{
            "rules": [{
                "rulePriority": 1,
                "description": "Keep last 10 images",
                "selection": {
                    "tagStatus": "any",
                    "countType": "imageCountMoreThan",
                    "countNumber": 10
                },
                "action": {"type": "expire"}
            }]
        }'

    log_success "ECR repository created"
}

create_eks_cluster() {
    log_step "Creating EKS Cluster"

    if aws eks describe-cluster --name "$CLUSTER_NAME" &> /dev/null; then
        log_info "EKS cluster already exists"
        return 0
    fi

    # Create IAM role for EKS
    log_info "Creating EKS IAM role..."
    local role_name="wavefield-${ENVIRONMENT}-eks-role"
    
    aws iam create-role \
        --role-name "$role_name" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "eks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' 2>/dev/null || true

    aws iam attach-role-policy \
        --role-name "$role_name" \
        --policy-arn "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy" 2>/dev/null || true

    local role_arn=$(aws iam get-role --role-name "$role_name" --query 'Role.Arn' --output text)

    # Create EKS cluster
    log_info "Creating EKS cluster (this may take 10-15 minutes)..."
    aws eks create-cluster \
        --name "$CLUSTER_NAME" \
        --role-arn "$role_arn" \
        --resources-vpc-config "subnetIds=$(echo $SUBNET_IDS | tr ' ' ','),securityGroupIds=${EKS_SG_ID}" \
        --kubernetes-version 1.28 \
        --tags "Environment=${ENVIRONMENT},Application=wavefield-llm"

    # Wait for cluster to be active
    log_info "Waiting for EKS cluster to be active..."
    aws eks wait cluster-active --name "$CLUSTER_NAME"

    # Update kubeconfig
    aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"

    log_success "EKS cluster created"
}

create_node_group() {
    log_step "Creating EKS Node Group"

    local node_group_name="wavefield-${ENVIRONMENT}-nodes"

    if aws eks describe-nodegroup \
        --cluster-name "$CLUSTER_NAME" \
        --nodegroup-name "$node_group_name" &> /dev/null; then
        log_info "Node group already exists"
        return 0
    fi

    # Create IAM role for nodes
    log_info "Creating node IAM role..."
    local node_role_name="wavefield-${ENVIRONMENT}-node-role"
    
    aws iam create-role \
        --role-name "$node_role_name" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' 2>/dev/null || true

    for policy in AmazonEKSWorkerNodePolicy AmazonEKS_CNI_Policy AmazonEC2ContainerRegistryReadOnly; do
        aws iam attach-role-policy \
            --role-name "$node_role_name" \
            --policy-arn "arn:aws:iam::aws:policy/${policy}" 2>/dev/null || true
    done

    local node_role_arn=$(aws iam get-role --role-name "$node_role_name" --query 'Role.Arn' --output text)

    # Create node group
    log_info "Creating node group..."
    aws eks create-nodegroup \
        --cluster-name "$CLUSTER_NAME" \
        --nodegroup-name "$node_group_name" \
        --node-role "$node_role_arn" \
        --subnets $SUBNET_IDS \
        --instance-types "$EKS_NODE_TYPE" \
        --scaling-config "minSize=2,maxSize=10,desiredSize=${EKS_NODE_COUNT}" \
        --disk-size 100 \
        --tags "Environment=${ENVIRONMENT},Application=wavefield-llm"

    # Wait for node group to be active
    log_info "Waiting for node group to be active..."
    aws eks wait nodegroup-active \
        --cluster-name "$CLUSTER_NAME" \
        --nodegroup-name "$node_group_name"

    log_success "Node group created"
}

setup_load_balancer() {
    log_step "Setting Up Load Balancer"

    # Install AWS Load Balancer Controller
    log_info "Installing AWS Load Balancer Controller..."
    
    kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"
    
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update
    
    helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
        -n kube-system \
        --set clusterName="$CLUSTER_NAME" \
        --set serviceAccount.create=true \
        --set region="$REGION" \
        --set vpcId="$VPC_ID"

    log_success "Load balancer configured"
}

setup_cloudfront() {
    if [[ "$ENABLE_CDN" != true ]]; then
        log_warning "CloudFront CDN disabled"
        return 0
    fi

    log_step "Setting Up CloudFront CDN"

    log_info "CloudFront setup requires manual configuration"
    log_info "See: deployment/guides/CLOUD_DEPLOYMENT.md"

    log_success "CloudFront setup instructions provided"
}

print_summary() {
    log_step "AWS Deployment Complete!"

    cat << EOF
${GREEN}Wave Field LLM AWS infrastructure deployed successfully!${NC}

Resources Created:
  VPC ID:              ${VPC_ID}
  EKS Cluster:         ${CLUSTER_NAME}
  Region:              ${REGION}
  Model Bucket:        ${MODEL_BUCKET}
  Logs Bucket:         ${LOGS_BUCKET}

Next Steps:
1. Deploy application:
   cd ${PROJECT_ROOT}
   ./deployment/automation/deploy.sh \\
     --environment ${ENVIRONMENT} \\
     --cloud aws \\
     --region ${REGION} \\
     --model wavefield-small

2. Configure DNS and SSL certificates
3. Setup monitoring and alerting
4. Run security audit

For more information, see: deployment/guides/CLOUD_DEPLOYMENT.md
EOF
}

main() {
    log_info "Starting AWS Deployment"
    
    parse_args "$@"
    check_aws_cli
    create_vpc
    create_security_groups
    create_rds
    create_elasticache
    create_s3_buckets
    create_ecr_repository
    create_eks_cluster
    create_node_group
    setup_load_balancer
    setup_cloudfront
    print_summary

    log_success "AWS deployment completed!"
}

main "$@"

# Made with Bob
