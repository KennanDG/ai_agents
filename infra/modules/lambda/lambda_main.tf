resource "aws_lambda_function" "this" {
  function_name = "${var.name}-api"
  role          = var.lambda_role_arn

  package_type = "Image"
  image_uri    = var.image_uri

  timeout     = 30
  memory_size = 1024

  ephemeral_storage {
    size = 4096 # (4GB)
  }

  environment {
    variables = {
      QDRANT_URL           = var.qdrant_url
      INGEST_QUEUE_URL     = var.ingest_queue_url
      LANGCHAIN_ENDPOINT   = var.langsmith_url
      RAW_BUCKET           = var.raw_bucket
      DERIVED_BUCKET       = var.derived_bucket
      GROQ_SECRET_ARN      = var.groq_secret_arn
      QDRANT_SECRET_ARN    = var.qdrant_secret_arn
      LANGCHAIN_SECRET_ARN = var.langchain_secret_arn
      SOURCES_TABLE        = var.sources_table_name
      JOBS_TABLE           = var.jobs_table_name

      LANGCHAIN_TRACING_V2 = tostring(var.langchain_tracing_v2)
      LANGCHAIN_PROJECT    = var.langchain_project

      FASTEMBED_CACHE_PATH  = "/tmp/fastembed_cache"
      HF_HOME               = "/tmp/hf"
      HUGGINGFACE_HUB_CACHE = "/tmp/hf/hub"
      TRANSFORMERS_CACHE    = "/tmp/hf/transformers"

    }
  }
}
