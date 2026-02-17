output "groq_secret_arn" { value = aws_secretsmanager_secret.groq.arn }
# output "db_secret_arn" { value = aws_secretsmanager_secret.db.arn }
output "qdrant_secret_arn" { value = aws_secretsmanager_secret.qdrant.arn }
