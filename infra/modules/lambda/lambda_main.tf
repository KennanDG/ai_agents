resource "aws_lambda_function" "this" {
  function_name = "${var.name}-api"
  role          = var.lambda_role_arn

  package_type = "Image"
  image_uri    = var.image_uri

  timeout     = 60
  memory_size = 2048

  ephemeral_storage {
    size = 4096 # (4GB)
  }

  environment {
    variables = {
      QDRANT_URL           = var.qdrant_url
      GROQ_URL             = var.groq_url
      JINA_URL             = var.jina_url
      LANGSMITH_URL        = var.langsmith_url
      INGEST_QUEUE_URL     = var.ingest_queue_url
      RAW_BUCKET           = var.raw_bucket
      DERIVED_BUCKET       = var.derived_bucket
      SOURCES_TABLE        = var.sources_table_name
      JOBS_TABLE           = var.jobs_table_name
      LANGCHAIN_PROJECT    = var.langchain_project
      LANGCHAIN_TRACING_V2 = tostring(var.langchain_tracing_v2)

      GROQ_SECRET_ARN      = var.groq_secret_arn
      QDRANT_SECRET_ARN    = var.qdrant_secret_arn
      LANGCHAIN_SECRET_ARN = var.langchain_secret_arn
      JINA_SECRET_ARN      = var.jina_secret_arn
      AI_AGENTS_SECRET_ARN       = var.app_secret_arn

    }
  }
}
