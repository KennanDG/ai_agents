data "aws_secretsmanager_secret_version" "db" {
  secret_id = var.db_secret_arn
}

# Retrieves username/name/password from secrets module
locals {
  db = jsondecode(data.aws_secretsmanager_secret_version.db.secret_string)
}

resource "aws_db_subnet_group" "this" {
  name       = "${var.name}-db-subnets"
  subnet_ids = var.private_subnet_ids
}

resource "aws_db_instance" "this" {
  identifier        = "${var.name}-pg"
  engine            = "postgres"
  engine_version    = "16.3"
  instance_class    = "db.t4g.micro"
  allocated_storage = 20
  storage_encrypted = true

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [var.db_sg_id]
  publicly_accessible    = false

  username = local.db.username
  password = local.db.password
  db_name  = local.db.dbname

  skip_final_snapshot = true
}
