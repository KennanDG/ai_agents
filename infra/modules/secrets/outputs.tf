output "groq_secret_arn" { value = aws_secretsmanager_secret.groq.arn }
output "langchain_secret_arn" { value = aws_secretsmanager_secret.langchain.arn }
output "qdrant_secret_arn" { value = aws_secretsmanager_secret.qdrant.arn }
output "jina_secret_arn" { value = aws_secretsmanager_secret.jina.arn }
output "app_secret_arn" { value = aws_secretsmanager_secret.app.arn }
