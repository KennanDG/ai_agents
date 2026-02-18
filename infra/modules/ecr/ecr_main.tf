resource "aws_ecr_repository" "api" {
  name = "${var.name}-api"
}

resource "aws_ecr_repository" "worker" {
  name = "${var.name}-worker"
}
