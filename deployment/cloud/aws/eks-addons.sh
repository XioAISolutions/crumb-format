#!/bin/bash
# Wave Field LLM - EKS Addons Installation Script
# Installs essential Kubernetes addons for production EKS cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="${CLUSTER_NAME:-wavefield-llm-cluster}"
AWS_REGION="${AWS_REGION:-us-east-1}"
NAMESPACE="kube-system"

echo -e "${GREEN}Installing EKS Addons for cluster: ${CLUSTER_NAME}${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
for cmd in kubectl helm aws; do
    if ! command_exists "$cmd"; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        exit 1
    fi
done

# Update kubeconfig
echo -e "${YELLOW}Updating kubeconfig...${NC}"
aws eks update-kubeconfig --region "$AWS_REGION" --name "$CLUSTER_NAME"

# Verify cluster access
echo -e "${YELLOW}Verifying cluster access...${NC}"
kubectl cluster-info

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
OIDC_PROVIDER=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.identity.oidc.issuer" --output text | sed -e "s/^https:\/\///")

echo -e "${GREEN}AWS Account ID: ${AWS_ACCOUNT_ID}${NC}"
echo -e "${GREEN}OIDC Provider: ${OIDC_PROVIDER}${NC}"

# ============================================================================
# 1. AWS Load Balancer Controller
# ============================================================================
echo -e "${GREEN}Installing AWS Load Balancer Controller...${NC}"

# Add EKS Helm repository
helm repo add eks https://aws.github.io/eks-charts
helm repo update

# Create service account
kubectl create serviceaccount aws-load-balancer-controller -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Annotate service account with IAM role
kubectl annotate serviceaccount aws-load-balancer-controller \
    -n "$NAMESPACE" \
    eks.amazonaws.com/role-arn="arn:aws:iam::${AWS_ACCOUNT_ID}:role/wavefield-llm-aws-lb-controller-role" \
    --overwrite

# Install AWS Load Balancer Controller
helm upgrade --install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n "$NAMESPACE" \
    --set clusterName="$CLUSTER_NAME" \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller \
    --set region="$AWS_REGION" \
    --set vpcId=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$AWS_REGION" --query "cluster.resourcesVpcConfig.vpcId" --output text) \
    --wait

echo -e "${GREEN}AWS Load Balancer Controller installed successfully${NC}"

# ============================================================================
# 2. External DNS
# ============================================================================
echo -e "${GREEN}Installing External DNS...${NC}"

# Add Bitnami Helm repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Create service account
kubectl create serviceaccount external-dns -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Annotate service account with IAM role
kubectl annotate serviceaccount external-dns \
    -n "$NAMESPACE" \
    eks.amazonaws.com/role-arn="arn:aws:iam::${AWS_ACCOUNT_ID}:role/wavefield-llm-external-dns-role" \
    --overwrite

# Install External DNS
helm upgrade --install external-dns bitnami/external-dns \
    -n "$NAMESPACE" \
    --set provider=aws \
    --set aws.region="$AWS_REGION" \
    --set serviceAccount.create=false \
    --set serviceAccount.name=external-dns \
    --set policy=sync \
    --set txtOwnerId="$CLUSTER_NAME" \
    --wait

echo -e "${GREEN}External DNS installed successfully${NC}"

# ============================================================================
# 3. Cluster Autoscaler
# ============================================================================
echo -e "${GREEN}Installing Cluster Autoscaler...${NC}"

# Add autoscaler Helm repository
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm repo update

# Create service account
kubectl create serviceaccount cluster-autoscaler -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Annotate service account with IAM role
kubectl annotate serviceaccount cluster-autoscaler \
    -n "$NAMESPACE" \
    eks.amazonaws.com/role-arn="arn:aws:iam::${AWS_ACCOUNT_ID}:role/wavefield-llm-cluster-autoscaler-role" \
    --overwrite

# Install Cluster Autoscaler
helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
    -n "$NAMESPACE" \
    --set autoDiscovery.clusterName="$CLUSTER_NAME" \
    --set awsRegion="$AWS_REGION" \
    --set rbac.serviceAccount.create=false \
    --set rbac.serviceAccount.name=cluster-autoscaler \
    --set extraArgs.balance-similar-node-groups=true \
    --set extraArgs.skip-nodes-with-system-pods=false \
    --wait

echo -e "${GREEN}Cluster Autoscaler installed successfully${NC}"

# ============================================================================
# 4. Metrics Server
# ============================================================================
echo -e "${GREEN}Installing Metrics Server...${NC}"

helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
helm repo update

helm upgrade --install metrics-server metrics-server/metrics-server \
    -n "$NAMESPACE" \
    --set args[0]=--kubelet-preferred-address-types=InternalIP \
    --wait

echo -e "${GREEN}Metrics Server installed successfully${NC}"

# ============================================================================
# 5. Cert Manager
# ============================================================================
echo -e "${GREEN}Installing Cert Manager...${NC}"

helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install CRDs
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.crds.yaml

# Install Cert Manager
helm upgrade --install cert-manager jetstack/cert-manager \
    --namespace cert-manager \
    --create-namespace \
    --version v1.13.2 \
    --wait

echo -e "${GREEN}Cert Manager installed successfully${NC}"

# ============================================================================
# 6. External Secrets Operator
# ============================================================================
echo -e "${GREEN}Installing External Secrets Operator...${NC}"

helm repo add external-secrets https://charts.external-secrets.io
helm repo update

helm upgrade --install external-secrets external-secrets/external-secrets \
    --namespace external-secrets-system \
    --create-namespace \
    --wait

echo -e "${GREEN}External Secrets Operator installed successfully${NC}"

# ============================================================================
# 7. NVIDIA Device Plugin (for GPU nodes)
# ============================================================================
echo -e "${GREEN}Installing NVIDIA Device Plugin...${NC}"

kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.3/nvidia-device-plugin.yml

echo -e "${GREEN}NVIDIA Device Plugin installed successfully${NC}"

# ============================================================================
# 8. Kubernetes Dashboard (Optional)
# ============================================================================
read -p "Do you want to install Kubernetes Dashboard? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Installing Kubernetes Dashboard...${NC}"
    
    helm repo add kubernetes-dashboard https://kubernetes.github.io/dashboard/
    helm repo update
    
    helm upgrade --install kubernetes-dashboard kubernetes-dashboard/kubernetes-dashboard \
        --namespace kubernetes-dashboard \
        --create-namespace \
        --wait
    
    echo -e "${GREEN}Kubernetes Dashboard installed successfully${NC}"
    echo -e "${YELLOW}To access the dashboard, run:${NC}"
    echo -e "${YELLOW}kubectl proxy${NC}"
    echo -e "${YELLOW}Then visit: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/${NC}"
fi

# ============================================================================
# 9. Prometheus & Grafana (Optional)
# ============================================================================
read -p "Do you want to install Prometheus & Grafana monitoring stack? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Installing Prometheus & Grafana...${NC}"
    
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.adminPassword=admin \
        --wait
    
    echo -e "${GREEN}Prometheus & Grafana installed successfully${NC}"
    echo -e "${YELLOW}To access Grafana, run:${NC}"
    echo -e "${YELLOW}kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80${NC}"
    echo -e "${YELLOW}Then visit: http://localhost:3000 (admin/admin)${NC}"
fi

# ============================================================================
# 10. Verify Installations
# ============================================================================
echo -e "${GREEN}Verifying installations...${NC}"

echo -e "${YELLOW}Checking pod status in kube-system namespace...${NC}"
kubectl get pods -n "$NAMESPACE"

echo -e "${YELLOW}Checking all deployments...${NC}"
kubectl get deployments --all-namespaces

echo -e "${YELLOW}Checking all services...${NC}"
kubectl get services --all-namespaces

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EKS Addons Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installed components:${NC}"
echo -e "  ✓ AWS Load Balancer Controller"
echo -e "  ✓ External DNS"
echo -e "  ✓ Cluster Autoscaler"
echo -e "  ✓ Metrics Server"
echo -e "  ✓ Cert Manager"
echo -e "  ✓ External Secrets Operator"
echo -e "  ✓ NVIDIA Device Plugin"
echo -e ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Deploy your application workloads"
echo -e "  2. Configure ingress resources"
echo -e "  3. Set up monitoring and alerting"
echo -e "  4. Configure backup and disaster recovery"
echo -e ""
echo -e "${YELLOW}Useful commands:${NC}"
echo -e "  kubectl get nodes"
echo -e "  kubectl get pods --all-namespaces"
echo -e "  kubectl top nodes"
echo -e "  kubectl top pods --all-namespaces"
echo -e ""

# Made with Bob
