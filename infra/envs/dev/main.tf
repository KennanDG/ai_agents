module "api_gateway" {
  source = "../../modules/api_gateway"
  name   = var.project_name

  lambda_invoke_arn  = module.lambda.lambda_invoke_arn
  lambda_function_name = module.lambda.lambda_function_name
}


module "db" {
  source = "../../modules/db"
  name   = var.project_name

  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  db_sg_id           = module.network.db_sg_id

  db_secret_arn      = module.secrets.db_secret_arn
}


module "ecr" {
  source = "../../modules/ecr"
  name   = var.project_name
}


module "ecs" {
  source = "../../modules/ecs"
  name   = var.project_name

  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  worker_sg_id       = module.network.worker_sg_id

  task_role_arn      = module.iam.ecs_task_role_arn
  execution_role_arn = module.iam.ecs_execution_role_arn

  worker_image_uri   = "${module.ecr.worker_repo_url}:${var.image_tag}"

  qdrant_url         = var.qdrant_url
  ingest_queue_url   = module.sqs.ingest_queue_url

  raw_bucket         = module.s3.raw_bucket
  derived_bucket     = module.s3.derived_bucket

  groq_secret_arn    = module.secrets.groq_secret_arn
  db_secret_arn      = module.secrets.db_secret_arn
}


module "iam" {
  source = "../../modules/iam"
  name   = var.project_name

  ingest_queue_arn = module.sqs.ingest_queue_arn
  raw_bucket_arn   = module.s3.raw_bucket_arn
  derived_bucket_arn = module.s3.derived_bucket_arn

  groq_secret_arn = module.secrets.groq_secret_arn
  db_secret_arn   = module.secrets.db_secret_arn
}


module "lambda" {
  source = "../../modules/lambda"
  name   = var.project_name

  private_subnet_ids = module.network.private_subnet_ids
  lambda_sg_id       = module.network.lambda_sg_id

  lambda_role_arn    = module.iam.lambda_role_arn

  image_uri          = "${module.ecr.api_repo_url}:${var.image_tag}"

  qdrant_url         = var.qdrant_url
  ingest_queue_url   = module.sqs.ingest_queue_url

  raw_bucket         = module.s3.raw_bucket
  derived_bucket     = module.s3.derived_bucket

  groq_secret_arn    = module.secrets.groq_secret_arn
  db_secret_arn      = module.secrets.db_secret_arn
}


module "network" {
  source     = "../../modules/network"
  name       = var.project_name
  vpc_cidr   = var.vpc_cidr
  aws_region = var.aws_region
}


module "s3" {
  source = "../../modules/s3"
  name   = var.project_name
}


module "secrets" {
  source = "../../modules/secrets"
  name   = var.project_name

  groq_api_key = var.groq_api_key

  db_username = var.db_username
  db_password = var.db_password
  db_name     = var.db_name
}


module "sqs" {
  source = "../../modules/sqs"
  name   = var.project_name
}

