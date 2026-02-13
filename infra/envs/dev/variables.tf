variable "project_name" {
  type    = string
  default = "ai-agents-dev"
}

variable "env" {
  type    = string
  default = "dev"
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "vpc_cidr" {
  type    = string
  default = "10.50.0.0/16"
}

# Allows for API testing from local IP
variable "allowed_cidr" {
  type    = string
  default = "0.0.0.0/0"
}

variable "image_tag" {
  type        = string
  description = "ECR image tag to deploy"
  default     = "latest"
}


variable "qdrant_url" {
  type        = string
  description = "Base URL for Qdrant cloud"
}

variable "groq_url" {
  type        = string
  description = "Base URL for Groq API"
}


variable "github_owner" {
  type        = string
  description = "GitHub account"
}
variable "github_repo" {
  type        = string
  description = "GitHub Repo"
}

variable "github_branch" {
  type    = string
  default = "main"
}

# Secrets
variable "groq_api_key" {
  type      = string
  sensitive = true
}

variable "qdrant_api_key" {
  type      = string
  sensitive = true
}

variable "db_username" {
  type    = string
  default = "ai_agents"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "db_name" {
  type    = string
  default = "ai_agents"
}

