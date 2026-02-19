output "groq_secret_arn" { value = aws_secretsmanager_secret.groq.arn }
output "langchain_secret_arn" { value = aws_secretsmanager_secret.langchain.arn }
output "qdrant_secret_arn" { value = aws_secretsmanager_secret.qdrant.arn }
