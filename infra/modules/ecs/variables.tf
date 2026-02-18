variable "name" { type = string }
variable "region" { type = string }

variable "vpc_id" { type = string }
# variable "private_subnet_ids" { type = list(string) }
variable "public_subnet_ids" { type = list(string) }
variable "worker_sg_id" { type = string }

variable "execution_role_arn" { type = string }
variable "task_role_arn" { type = string }

variable "worker_image_uri" { type = string }

variable "qdrant_url" { type = string }
variable "ingest_queue_url" { type = string }

variable "raw_bucket" { type = string }
variable "derived_bucket" { type = string }

variable "groq_secret_arn" { type = string }
variable "qdrant_secret_arn" { type = string }
# variable "db_secret_arn" { type = string }

variable "sources_table_name" { type = string }
variable "jobs_table_name" { type = string }
