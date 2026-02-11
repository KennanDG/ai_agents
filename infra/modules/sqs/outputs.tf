output "ingest_queue_url" { value = aws_sqs_queue.ingest.url }
output "ingest_queue_arn" { value = aws_sqs_queue.ingest.arn }

output "dlq_url" { value = aws_sqs_queue.dlq.url }
output "dlq_arn" { value = aws_sqs_queue.dlq.arn }
