resource "aws_secretsmanager_secret" "groq" {
  name = "${var.name}/groq"
}

resource "aws_secretsmanager_secret_version" "groq" {
  secret_id     = aws_secretsmanager_secret.groq.id
  secret_string = jsonencode({ GROQ_API_KEY = var.groq_api_key })
}

resource "aws_secretsmanager_secret" "qdrant" {
  name = "${var.name}/qdrant"
}

resource "aws_secretsmanager_secret_version" "qdrant" {
  secret_id     = aws_secretsmanager_secret.qdrant.id
  secret_string = jsonencode({ QDRANT_API_KEY = var.qdrant_api_key })
}

# resource "aws_secretsmanager_secret" "db" {
#   name = "${var.name}/db"
# }


# resource "aws_secretsmanager_secret_version" "db" {
#   secret_id = aws_secretsmanager_secret.db.id
#   secret_string = jsonencode({
#     username = var.db_username
#     password = var.db_password
#     dbname   = var.db_name
#   })
# }

