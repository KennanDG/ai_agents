data "aws_caller_identity" "this" {}

# ---- Lambda Role ----
data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.name}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_vpc" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

data "aws_iam_policy_document" "lambda_inline" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.groq_secret_arn, var.db_secret_arn]
  }

  statement {
    actions   = ["sqs:SendMessage"]
    resources = [var.ingest_queue_arn]
  }

  statement {
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket"
    ]
    resources = [
      var.raw_bucket_arn,
      "${var.raw_bucket_arn}/*",
      var.derived_bucket_arn,
      "${var.derived_bucket_arn}/*",
    ]
  }
}

resource "aws_iam_policy" "lambda_app" {
  name   = "${var.name}-lambda-app"
  policy = data.aws_iam_policy_document.lambda_inline.json
}

resource "aws_iam_role_policy_attachment" "lambda_app" {
  role       = aws_iam_role.lambda.name
  policy_arn = aws_iam_policy.lambda_app.arn
}



# ---- ECS Task Execution Role (pull image, write logs) ----
data "aws_iam_policy_document" "ecs_exec_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${var.name}-ecs-exec-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_exec_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_exec" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---- ECS Task Role (your worker permissions) ----
resource "aws_iam_role" "ecs_task" {
  name               = "${var.name}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_exec_assume.json
}

data "aws_iam_policy_document" "ecs_task_inline" {
  statement {
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      var.groq_secret_arn,
      var.db_secret_arn
    ]
  }

  statement {
    actions = [
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:ChangeMessageVisibility"
    ]
    resources = [var.ingest_queue_arn]
  }

  statement {
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      var.raw_bucket_arn,
      "${var.raw_bucket_arn}/*",
      var.derived_bucket_arn,
      "${var.derived_bucket_arn}/*",
    ]
  }
}

resource "aws_iam_policy" "ecs_task" {
  name   = "${var.name}-ecs-task"
  policy = data.aws_iam_policy_document.ecs_task_inline.json
}

resource "aws_iam_role_policy_attachment" "ecs_task_attach" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = aws_iam_policy.ecs_task.arn
}
