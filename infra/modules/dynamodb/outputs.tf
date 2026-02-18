output "sources_table_name" { value = aws_dynamodb_table.sources.name }
output "sources_table_arn" { value = aws_dynamodb_table.sources.arn }

output "jobs_table_name" { value = aws_dynamodb_table.jobs.name }
output "jobs_table_arn" { value = aws_dynamodb_table.jobs.arn }