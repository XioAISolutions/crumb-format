#!/bin/bash
set -euo pipefail

# Wave Field LLM - CI/CD Pipeline Setup Script
# Configures automated testing and deployment pipelines

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CI_PLATFORM=""
GITHUB_REPO=""
GITLAB_PROJECT=""
JENKINS_URL=""
DRY_RUN=false
VERBOSE=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${GREEN}==>${NC} ${BLUE}$1${NC}\n"
}

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup CI/CD pipelines for Wave Field LLM

OPTIONS:
    --platform PLATFORM      CI/CD platform (github, gitlab, jenkins)
    --github-repo REPO       GitHub repository (owner/repo)
    --gitlab-project ID      GitLab project ID
    --jenkins-url URL        Jenkins server URL
    --dry-run               Show what would be done without executing
    --verbose               Enable verbose output
    -h, --help              Show this help message

EXAMPLES:
    # Setup GitHub Actions
    $0 --platform github --github-repo myorg/wavefield-llm

    # Setup GitLab CI
    $0 --platform gitlab --gitlab-project 12345

    # Setup Jenkins
    $0 --platform jenkins --jenkins-url https://jenkins.example.com

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            CI_PLATFORM="$2"
            shift 2
            ;;
        --github-repo)
            GITHUB_REPO="$2"
            shift 2
            ;;
        --gitlab-project)
            GITLAB_PROJECT="$2"
            shift 2
            ;;
        --jenkins-url)
            JENKINS_URL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [[ -z "$CI_PLATFORM" ]]; then
    log_error "CI/CD platform is required"
    usage
fi

# Setup GitHub Actions
setup_github_actions() {
    log_step "Setting up GitHub Actions"
    
    local workflows_dir="${PROJECT_ROOT}/.github/workflows"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create GitHub Actions workflows in $workflows_dir"
        return 0
    fi
    
    mkdir -p "$workflows_dir"
    
    # Create CI workflow
    cat > "${workflows_dir}/ci.yml" << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run linting
      run: |
        flake8 crumb_llm tests
        black --check crumb_llm tests
        mypy crumb_llm
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=crumb_llm --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build Docker image
      run: |
        docker build -t wavefield-llm:${{ github.sha }} -f deployment/docker/Dockerfile .
    
    - name: Run security scan
      run: |
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy image wavefield-llm:${{ github.sha }}
EOF
    
    # Create CD workflow
    cat > "${workflows_dir}/cd.yml" << 'EOF'
name: CD

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to ECR
      run: |
        aws ecr get-login-password --region us-east-1 | \
          docker login --username AWS --password-stdin ${{ secrets.ECR_REGISTRY }}
    
    - name: Build and push
      run: |
        docker build -t ${{ secrets.ECR_REGISTRY }}/wavefield-llm:staging -f deployment/docker/Dockerfile .
        docker push ${{ secrets.ECR_REGISTRY }}/wavefield-llm:staging
    
    - name: Deploy to staging
      run: |
        cd deployment/automation
        ./deploy.sh --environment staging --cloud aws --region us-east-1

  deploy-production:
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Login to ECR
      run: |
        aws ecr get-login-password --region us-east-1 | \
          docker login --username AWS --password-stdin ${{ secrets.ECR_REGISTRY }}
    
    - name: Build and push
      run: |
        VERSION=${GITHUB_REF#refs/tags/v}
        docker build -t ${{ secrets.ECR_REGISTRY }}/wavefield-llm:$VERSION -f deployment/docker/Dockerfile .
        docker push ${{ secrets.ECR_REGISTRY }}/wavefield-llm:$VERSION
    
    - name: Deploy to production
      run: |
        VERSION=${GITHUB_REF#refs/tags/v}
        cd deployment/automation
        ./deploy.sh --environment production --cloud aws --region us-east-1 --version $VERSION
EOF
    
    log_success "GitHub Actions workflows created"
    log_info "Workflows location: $workflows_dir"
    log_info "Required secrets:"
    log_info "  - AWS_ACCESS_KEY_ID"
    log_info "  - AWS_SECRET_ACCESS_KEY"
    log_info "  - ECR_REGISTRY"
}

# Setup GitLab CI
setup_gitlab_ci() {
    log_step "Setting up GitLab CI"
    
    local gitlab_ci="${PROJECT_ROOT}/.gitlab-ci.yml"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create GitLab CI configuration at $gitlab_ci"
        return 0
    fi
    
    cat > "$gitlab_ci" << 'EOF'
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.10
  script:
    - pip install -e ".[dev]"
    - flake8 crumb_llm tests
    - black --check crumb_llm tests
    - mypy crumb_llm
    - pytest tests/ -v --cov=crumb_llm --cov-report=xml
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -f deployment/docker/Dockerfile .
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
    - tags

deploy:staging:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl bash
    - curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    - chmod +x kubectl
    - mv kubectl /usr/local/bin/
  script:
    - cd deployment/automation
    - ./deploy.sh --environment staging --cloud aws --region us-east-1
  only:
    - main
  environment:
    name: staging
    url: https://staging.wavefield-llm.example.com

deploy:production:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl bash
    - curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    - chmod +x kubectl
    - mv kubectl /usr/local/bin/
  script:
    - cd deployment/automation
    - ./deploy.sh --environment production --cloud aws --region us-east-1 --version $CI_COMMIT_TAG
  only:
    - tags
  when: manual
  environment:
    name: production
    url: https://wavefield-llm.example.com
EOF
    
    log_success "GitLab CI configuration created"
    log_info "Configuration location: $gitlab_ci"
    log_info "Required CI/CD variables:"
    log_info "  - AWS_ACCESS_KEY_ID"
    log_info "  - AWS_SECRET_ACCESS_KEY"
    log_info "  - KUBECONFIG (base64 encoded)"
}

# Setup Jenkins
setup_jenkins() {
    log_step "Setting up Jenkins"
    
    local jenkinsfile="${PROJECT_ROOT}/Jenkinsfile"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create Jenkinsfile at $jenkinsfile"
        return 0
    fi
    
    cat > "$jenkinsfile" << 'EOF'
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = credentials('docker-registry')
        AWS_CREDENTIALS = credentials('aws-credentials')
        KUBECONFIG = credentials('kubeconfig')
    }
    
    stages {
        stage('Test') {
            agent {
                docker {
                    image 'python:3.10'
                }
            }
            steps {
                sh '''
                    pip install -e ".[dev]"
                    flake8 crumb_llm tests
                    black --check crumb_llm tests
                    mypy crumb_llm
                    pytest tests/ -v --cov=crumb_llm --cov-report=xml
                '''
            }
            post {
                always {
                    junit 'test-results/*.xml'
                    cobertura coberturaReportFile: 'coverage.xml'
                }
            }
        }
        
        stage('Build') {
            steps {
                script {
                    docker.build("wavefield-llm:${env.BUILD_ID}", "-f deployment/docker/Dockerfile .")
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                sh '''
                    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
                        aquasec/trivy image wavefield-llm:${BUILD_ID}
                '''
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    cd deployment/automation
                    ./deploy.sh --environment staging --cloud aws --region us-east-1
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                tag pattern: "v\\d+\\.\\d+\\.\\d+", comparator: "REGEXP"
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                sh '''
                    cd deployment/automation
                    ./deploy.sh --environment production --cloud aws --region us-east-1 --version ${TAG_NAME}
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            slackSend color: 'good', message: "Build ${env.BUILD_NUMBER} succeeded"
        }
        failure {
            slackSend color: 'danger', message: "Build ${env.BUILD_NUMBER} failed"
        }
    }
}
EOF
    
    log_success "Jenkinsfile created"
    log_info "Jenkinsfile location: $jenkinsfile"
    log_info "Required Jenkins credentials:"
    log_info "  - docker-registry (username/password)"
    log_info "  - aws-credentials (AWS access key)"
    log_info "  - kubeconfig (secret file)"
}

# Main execution
main() {
    log_step "Wave Field LLM CI/CD Setup"
    
    case "$CI_PLATFORM" in
        github)
            if [[ -z "$GITHUB_REPO" ]]; then
                log_error "GitHub repository is required for GitHub Actions"
                exit 1
            fi
            setup_github_actions
            ;;
        gitlab)
            if [[ -z "$GITLAB_PROJECT" ]]; then
                log_error "GitLab project ID is required for GitLab CI"
                exit 1
            fi
            setup_gitlab_ci
            ;;
        jenkins)
            if [[ -z "$JENKINS_URL" ]]; then
                log_error "Jenkins URL is required for Jenkins setup"
                exit 1
            fi
            setup_jenkins
            ;;
        *)
            log_error "Unknown CI/CD platform: $CI_PLATFORM"
            log_error "Supported platforms: github, gitlab, jenkins"
            exit 1
            ;;
    esac
    
    log_step "CI/CD Setup Complete"
    log_success "CI/CD pipeline configured for $CI_PLATFORM"
    
    log_info "\nNext steps:"
    log_info "1. Review the generated configuration files"
    log_info "2. Configure required secrets/credentials"
    log_info "3. Push changes to trigger the pipeline"
    log_info "4. Monitor the first pipeline run"
}

# Run main function
main

# Made with Bob
