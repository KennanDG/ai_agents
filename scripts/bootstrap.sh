#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:=us-east-1}"
: "${ENV:=dev}"                 # dev | prod
: "${PROJECT:=ai-agents}"       

STATE_BUCKET="${PROJECT}-tfstate-${ENV}"
LOCK_TABLE="${PROJECT}-tflock-${ENV}"

echo "Bootstrapping Terraform backend..."
echo "  AWS_REGION  = ${AWS_REGION}"
echo "  ENV         = ${ENV}"
echo "  STATE_BUCKET= ${STATE_BUCKET}"
echo "  LOCK_TABLE  = ${LOCK_TABLE}"
echo

# Ensure AWS identity works
aws sts get-caller-identity >/dev/null

# --------
# S3 bucket (idempotent)
# --------
if aws s3api head-bucket --bucket "${STATE_BUCKET}" 2>/dev/null; then
  echo "S3 bucket already exists: ${STATE_BUCKET}"
else
  echo "Creating S3 bucket: ${STATE_BUCKET}"
  if [[ "${AWS_REGION}" == "us-east-1" ]]; then
    aws s3api create-bucket \
      --bucket "${STATE_BUCKET}" \
      --region "${AWS_REGION}"
  else
    aws s3api create-bucket \
      --bucket "${STATE_BUCKET}" \
      --region "${AWS_REGION}" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}"
  fi
fi

# Versioning
aws s3api put-bucket-versioning \
  --bucket "${STATE_BUCKET}" \
  --versioning-configuration Status=Enabled

# Block public access
aws s3api put-public-access-block \
  --bucket "${STATE_BUCKET}" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true


# Default encryption (AES256).
aws s3api put-bucket-encryption \
  --bucket "${STATE_BUCKET}" \
  --server-side-encryption-configuration '{
    "Rules": [
      {
        "ApplyServerSideEncryptionByDefault": { "SSEAlgorithm": "AES256" }
      }
    ]
  }'


# --------
# DynamoDB lock table (idempotent)
# --------
if aws dynamodb describe-table --table-name "${LOCK_TABLE}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "DynamoDB table already exists: ${LOCK_TABLE}"
else
  echo "Creating DynamoDB table: ${LOCK_TABLE}"
  aws dynamodb create-table \
    --table-name "${LOCK_TABLE}" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "${AWS_REGION}"
fi

echo
echo "✅ Bootstrap complete."
echo "Use these in backend.tf:"
echo "State bucket: ${STATE_BUCKET}"
echo "Lock table:  ${LOCK_TABLE}"
echo "region:  ${AWS_REGION}"