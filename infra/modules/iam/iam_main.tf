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

# resource "aws_iam_role_policy_attachment" "lambda_vpc" {
#   role       = aws_iam_role.lambda.name
#   policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
# }

data "aws_iam_policy_document" "lambda_inline" {
  statement {
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetResourcePolicy",
      "secretsmanager:ListSecrets",
    ]
    resources = [
      var.groq_secret_arn,
      var.qdrant_secret_arn
    ]
  }

  # statement {
  #   actions   = [
  #     "sqs:SendMessage",
  #     "sqs:DeleteMessage",
  #     "sqs:GetQueueAttributes",
  #     "sqs:ChangeMessageVisibility",
  #     "sqs:GetQueueURL"
  #   ]
  #   resources = [var.ingest_queue_arn]
  # }

  statement {
    actions = [
      "sqs:ListQueues",
      "sqs:CreateQueue",
      "sqs:TagQueue",
      "sqs:DeleteQueue",
      "sqs:PurgeQueue",
      "sqs:ListQueueTags",
      "sqs:SendMessage",
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:SetQueueAttributes",
      "sqs:ChangeMessageVisibility",
      "sqs:GetQueueURL"
    ]
    resources = ["*"]
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


  statement {
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:Query",
      "dynamodb:Scan"
    ]
    resources = [
      var.sources_table_arn,
      var.jobs_table_arn
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


data "aws_iam_policy_document" "ecs_exec_secrets" {
  statement {
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret"
    ]
    resources = [
      var.groq_secret_arn,
      var.qdrant_secret_arn
    ]
  }
}

resource "aws_iam_policy" "ecs_exec_secrets" {
  name   = "${var.name}-ecs-exec-secrets"
  policy = data.aws_iam_policy_document.ecs_exec_secrets.json
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${var.name}-ecs-exec-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_exec_assume.json
}

resource "aws_iam_role_policy_attachment" "ecs_exec" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy_attachment" "ecs_exec_secrets" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = aws_iam_policy.ecs_exec_secrets.arn
}


# ---- ECS Task Role (your worker permissions) ----
resource "aws_iam_role" "ecs_task" {
  name               = "${var.name}-ecs-task-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_exec_assume.json
}

data "aws_iam_policy_document" "ecs_task_inline" {
  statement {
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetResourcePolicy",
      "secretsmanager:ListSecrets"
    ]
    resources = [
      var.groq_secret_arn,
      var.qdrant_secret_arn
    ]
  }

  # statement {
  #   actions   = [
  #     "sqs:SendMessage",
  #     "sqs:DeleteMessage",
  #     "sqs:GetQueueAttributes",
  #     "sqs:ChangeMessageVisibility",
  #     "sqs:GetQueueURL"
  #   ]
  #   resources = [var.ingest_queue_arn]
  # }

  statement {
    actions = [
      "sqs:ListQueues",
      "sqs:CreateQueue",
      "sqs:TagQueue",
      "sqs:DeleteQueue",
      "sqs:PurgeQueue",
      "sqs:ListQueueTags",
      "sqs:SendMessage",
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:SetQueueAttributes",
      "sqs:ChangeMessageVisibility",
      "sqs:GetQueueURL"
    ]
    resources = ["*"]
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


  statement {
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:Query",
      "dynamodb:Scan"
    ]
    resources = [
      var.sources_table_arn,
      var.jobs_table_arn
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
