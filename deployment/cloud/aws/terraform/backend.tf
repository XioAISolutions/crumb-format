# Wave Field LLM - Terraform Backend Configuration
# S3 backend with DynamoDB locking for state management

terraform {
  backend "s3" {
    # Backend configuration should be provided via backend config file or CLI
    # Example: terraform init -backend-config=backend.hcl
    
    # bucket         = "wavefield-llm-terraform-state"
    # key            = "prod/terraform.tfstate"
    # region         = "us-east-1"
    # encrypt        = true
    # dynamodb_table = "wavefield-llm-terraform-locks"
    # kms_key_id     = "arn:aws:kms:us-east-1:ACCOUNT_ID:key/KEY_ID"
    
    # Additional security settings
    # acl            = "private"
    # versioning     = true
  }
}

# Backend Setup Instructions:
# 
# 1. Create S3 bucket for state:
#    aws s3api create-bucket \
#      --bucket wavefield-llm-terraform-state \
#      --region us-east-1
#
#    aws s3api put-bucket-versioning \
#      --bucket wavefield-llm-terraform-state \
#      --versioning-configuration Status=Enabled
#
#    aws s3api put-bucket-encryption \
#      --bucket wavefield-llm-terraform-state \
#      --server-side-encryption-configuration '{
#        "Rules": [{
#          "ApplyServerSideEncryptionByDefault": {
#            "SSEAlgorithm": "aws:kms"
#          }
#        }]
#      }'
#
#    aws s3api put-public-access-block \
#      --bucket wavefield-llm-terraform-state \
#      --public-access-block-configuration \
#        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
#
# 2. Create DynamoDB table for locking:
#    aws dynamodb create-table \
#      --table-name wavefield-llm-terraform-locks \
#      --attribute-definitions AttributeName=LockID,AttributeType=S \
#      --key-schema AttributeName=LockID,KeyType=HASH \
#      --billing-mode PAY_PER_REQUEST \
#      --region us-east-1
#
# 3. Create backend.hcl file:
#    cat > backend.hcl <<EOF
#    bucket         = "wavefield-llm-terraform-state"
#    key            = "prod/terraform.tfstate"
#    region         = "us-east-1"
#    encrypt        = true
#    dynamodb_table = "wavefield-llm-terraform-locks"
#    EOF
#
# 4. Initialize Terraform with backend:
#    terraform init -backend-config=backend.hcl
#
# For multiple environments, use different state keys:
# - dev/terraform.tfstate
# - staging/terraform.tfstate
# - prod/terraform.tfstate