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


resource "aws_secretsmanager_secret" "langchain" {
  name = "${var.name}/langchain"
}

resource "aws_secretsmanager_secret_version" "langchain" {
  secret_id     = aws_secretsmanager_secret.langchain.id
  secret_string = jsonencode({ LANGCHAIN_API_KEY = var.langchain_api_key })
}


resource "aws_secretsmanager_secret" "jina" {
  name = "${var.name}/jina"
}

resource "aws_secretsmanager_secret_version" "jina" {
  secret_id     = aws_secretsmanager_secret.jina.id
  secret_string = jsonencode({ JINA_API_KEY = var.jina_api_key })
}


resource "aws_secretsmanager_secret" "app" {
  name = "${var.name}/app"
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id     = aws_secretsmanager_secret.app.id
  secret_string = jsonencode({ AI_AGENTS_API_KEY = var.ai_agents_api_key })
}
