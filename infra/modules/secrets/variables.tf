variable "name" { type = string }

variable "groq_api_key" {
  type      = string
  sensitive = true
}

variable "qdrant_api_key" {
  type      = string
  sensitive = true
}

variable "langchain_api_key" {
  type      = string
  sensitive = true
}
