# --- API ---
output "api_url" {
  value       = module.api_gateway.api_url
  description = "HTTP API Gateway base URL"
}

output "lambda_function_name" {
  value       = module.lambda.lambda_function_name
  description = "Lambda function name (used by deploy workflow)"
}



# --- ECR ---
output "api_repo_url" {
  value       = module.ecr.api_repo_url
  description = "ECR repository URL for API image"
}

output "worker_repo_url" {
  value       = module.ecr.worker_repo_url
  description = "ECR repository URL for worker image"
}



# --- ECS ---
output "ecs_cluster_name" {
  value       = module.ecs.cluster_name
  description = "ECS cluster name"
}

output "ecs_service_name" {
  value       = module.ecs.service_name
  description = "ECS service name for worker"
}



# --- Data plane resources ---
output "ingest_queue_url" {
  value       = module.sqs.ingest_queue_url
  description = "SQS queue URL for ingestion jobs"
}

output "raw_bucket" {
  value       = module.s3.raw_bucket
  description = "S3 bucket for raw files"
}

output "derived_bucket" {
  value       = module.s3.derived_bucket
  description = "S3 bucket for derived artifacts"
}



# --- DB (useful for debugging; keep secrets in Secrets Manager) ---
# output "db_host" {
#   value       = module.db.db_host
#   description = "RDS endpoint hostname"
# }

# output "db_port" {
#   value       = module.db.db_port
#   description = "RDS port"
# }




# --- IAM for CI/CD ---
output "github_deploy_role_arn" {
  value       = module.iam.github_deploy_role_arn
  description = "Role ARN GitHub Actions assumes via OIDC"
}
