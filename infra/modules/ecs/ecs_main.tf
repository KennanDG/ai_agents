resource "aws_ecs_cluster" "this" {
  name = "${var.name}-cluster"
}

resource "aws_cloudwatch_log_group" "cloudwatch_worker" {
  name              = "/ecs/${var.name}-worker"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${var.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048

  execution_role_arn = var.execution_role_arn
  task_role_arn      = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = var.worker_image_uri
      essential = true

      command = ["celery", "-A", "ai_agents.jobs.celery", "worker", "--loglevel=INFO"]

      environment = [
        { name = "QDRANT_URL", value = var.qdrant_url },
        { name = "LANGSMITH_URL", value = var.langsmith_url },
        { name = "INGEST_QUEUE_URL", value = var.ingest_queue_url },
        { name = "RAW_BUCKET", value = var.raw_bucket },
        { name = "DERIVED_BUCKET", value = var.derived_bucket },
        { name = "GROQ_SECRET_ARN", value = var.groq_secret_arn },
        { name = "LANGCHAIN_SECRET_ARN", value = var.langchain_secret_arn },
        { name = "SOURCES_TABLE", value = var.sources_table_name },
        { name = "JOBS_TABLE", value = var.jobs_table_name },
        { name = "LANGCHAIN_PROJECT", value = var.langchain_project },
        { name = "LANGCHAIN_TRACING_V2", value = tostring(var.langchain_tracing_v2) },
      ]

      secrets = [
        {
          name      = "GROQ_API_KEY"
          valueFrom = "${var.groq_secret_arn}:GROQ_API_KEY::"
        },
        {
          name      = "QDRANT_API_KEY"
          valueFrom = "${var.qdrant_secret_arn}:QDRANT_API_KEY::"
        },
        {
          name      = "LANGCHAIN_API_KEY"
          valueFrom = "${var.langchain_secret_arn}:LANGCHAIN_API_KEY::"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.cloudwatch_worker.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "worker" {
  name            = "${var.name}-worker"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.worker.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.public_subnet_ids
    security_groups  = [var.worker_sg_id]
    assign_public_ip = true
  }
}
