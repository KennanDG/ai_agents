resource "aws_ecs_cluster" "this" {
  name = "${var.name}-cluster"
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${var.name}-worker"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "worker" {
  family                   = "${var.name}-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024

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
        { name = "INGEST_QUEUE_URL", value = var.ingest_queue_url },
        { name = "RAW_BUCKET", value = var.raw_bucket },
        { name = "DERIVED_BUCKET", value = var.derived_bucket },
        { name = "GROQ_SECRET_ARN", value = var.groq_secret_arn },
        { name = "DB_SECRET_ARN", value = var.db_secret_arn }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.worker.name
          awslogs-region        = "us-east-1"
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
    subnets         = var.private_subnet_ids
    security_groups = [var.worker_sg_id]
  }
}
