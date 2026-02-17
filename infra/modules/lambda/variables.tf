variable "name" { type = string }

# variable "private_subnet_ids" { type = list(string) }
variable "lambda_sg_id" { type = string }

variable "lambda_role_arn" { type = string }
variable "image_uri" { type = string }

variable "qdrant_url" { type = string }
variable "ingest_queue_url" { type = string }

variable "raw_bucket" { type = string }
variable "derived_bucket" { type = string }

variable "groq_secret_arn" { type = string }
variable "qdrant_secret_arn" { type = string }
# variable "db_secret_arn" { type = string }

variable "sources_table_name" { type = string }
variable "jobs_table_name" { type = string }
