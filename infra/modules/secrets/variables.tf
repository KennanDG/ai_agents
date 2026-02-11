variable "name" { type = string }

variable "groq_api_key" {
  type      = string
  sensitive = true
}

variable "db_username" { type = string }
variable "db_name" { type = string }

variable "db_password" {
  type      = string
  sensitive = true
}

