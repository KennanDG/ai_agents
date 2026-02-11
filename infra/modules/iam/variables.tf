variable "name" { type = string }

variable "ingest_queue_arn" { type = string }

variable "raw_bucket_arn" { type = string }
variable "derived_bucket_arn" { type = string }

variable "groq_secret_arn" { type = string }
variable "db_secret_arn" { type = string }
