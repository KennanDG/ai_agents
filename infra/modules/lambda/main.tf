resource "aws_lambda_function" "this" {
  function_name = "${var.name}-api"
  role          = var.lambda_role_arn

  package_type = "Image"
  image_uri    = var.image_uri

  timeout     = 30
  memory_size = 1024

  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [var.lambda_sg_id]
  }

  environment {
    variables = {
      QDRANT_URL        = var.qdrant_url
      INGEST_QUEUE_URL  = var.ingest_queue_url
      RAW_BUCKET        = var.raw_bucket
      DERIVED_BUCKET    = var.derived_bucket
      GROQ_SECRET_ARN   = var.groq_secret_arn
      QDRANT_SECRET_ARN = var.qdrant_secret_arn
      DB_SECRET_ARN     = var.db_secret_arn
    }
  }
}
