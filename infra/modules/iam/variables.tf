variable "name" { type = string }

variable "ingest_queue_arn" { type = string }

variable "raw_bucket_arn" { type = string }
variable "derived_bucket_arn" { type = string }

variable "groq_secret_arn" { type = string }
variable "qdrant_secret_arn" { type = string }
variable "langchain_secret_arn" { type = string }

variable "sources_table_arn" { type = string }
variable "jobs_table_arn" { type = string }

variable "github_owner" { type = string }
variable "github_repo" { type = string }

variable "github_branch" {
  type    = string
  default = "main"
}