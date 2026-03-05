resource "aws_ecs_cluster" "this" {
  name = "${var.name}-cluster"
}



###################################
############ Worker ###############
###################################


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

  volume {
    name = "models"

    efs_volume_configuration {
      file_system_id = aws_efs_file_system.models.id
      root_directory = "/"
    }
  }

  container_definitions = jsonencode([
    {
      name      = "worker"
      image     = var.worker_image_uri
      essential = true

      command = ["celery", "-A", "ai_agents.jobs.celery", "worker", "--loglevel=INFO"]

      mountPoints = [
        { sourceVolume = "models", containerPath = "/mnt/models", readOnly = false }
      ]

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

        # Persistent cache for FastEmbed
        { name = "FASTEMBED_CACHE_PATH", value = "/mnt/models/fastembed_cache" },
        { name = "HF_HOME", value = "/mnt/models/hf" },
        { name = "HUGGINGFACE_HUB_CACHE", value = "/mnt/models/hf/hub" },
        { name = "TRANSFORMERS_CACHE", value = "/mnt/models/hf/transformers" },
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



###################################
############ API ##################
###################################

resource "aws_cloudwatch_log_group" "cloudwatch_api" {
  name              = "/ecs/${var.name}-api"
  retention_in_days = 14
}


resource "aws_security_group" "efs" {
  name   = "${var.name}-sg-efs"
  vpc_id = var.vpc_id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [var.api_sg_id, var.worker_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_efs_file_system" "models" {
  creation_token = "${var.name}-models"
  encrypted      = true
}

resource "aws_efs_mount_target" "models" {
  count           = length(var.public_subnet_ids)
  file_system_id  = aws_efs_file_system.models.id
  subnet_id       = var.public_subnet_ids[count.index]
  security_groups = [aws_security_group.efs.id]
}



resource "aws_lb" "api" {
  name               = "${var.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.alb_sg_id]
  subnets            = var.public_subnet_ids
}

resource "aws_lb_target_group" "api" {
  name        = "${var.name}-tg-api"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}




resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 1024
  memory                   = 2048

  execution_role_arn = var.execution_role_arn
  task_role_arn      = var.task_role_arn

  volume {
    name = "models"

    efs_volume_configuration {
      file_system_id = aws_efs_file_system.models.id
      root_directory = "/"
    }
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.api_image_uri
      essential = true

      portMappings = [
        { containerPort = 8000, hostPort = 8000, protocol = "tcp" }
      ]

      command = [
        "/bin/sh", "-lc",
        "python -m ai_agents.scripts.warm_models_2 && uvicorn ai_agents.api.main:app --host 0.0.0.0 --port 8000"
      ]

      mountPoints = [
        { sourceVolume = "models", containerPath = "/mnt/models", readOnly = false }
      ]

      environment = [
        { name = "QDRANT_URL", value = var.qdrant_url },
        { name = "LANGSMITH_URL", value = var.langsmith_url },
        { name = "INGEST_QUEUE_URL", value = var.ingest_queue_url },
        { name = "RAW_BUCKET", value = var.raw_bucket },
        { name = "DERIVED_BUCKET", value = var.derived_bucket },
        { name = "SOURCES_TABLE", value = var.sources_table_name },
        { name = "JOBS_TABLE", value = var.jobs_table_name },
        { name = "LANGCHAIN_PROJECT", value = var.langchain_project },
        { name = "LANGCHAIN_TRACING_V2", value = tostring(var.langchain_tracing_v2) },

        # Persistent cache for FastEmbed
        { name = "FASTEMBED_CACHE_PATH", value = "/mnt/models/fastembed_cache" },
        { name = "HF_HOME", value = "/mnt/models/hf" },
        { name = "HUGGINGFACE_HUB_CACHE", value = "/mnt/models/hf/hub" },
        { name = "TRANSFORMERS_CACHE", value = "/mnt/models/hf/transformers" },
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
          awslogs-group         = aws_cloudwatch_log_group.cloudwatch_api.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}


resource "aws_ecs_service" "api" {
  name                   = "${var.name}-api"
  cluster                = aws_ecs_cluster.this.id
  task_definition        = aws_ecs_task_definition.api.arn
  desired_count          = 1
  launch_type            = "FARGATE"
  enable_execute_command = true

  network_configuration {
    subnets          = var.public_subnet_ids
    security_groups  = [var.api_sg_id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.http]
}
