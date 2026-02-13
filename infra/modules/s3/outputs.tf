output "raw_bucket" { value = aws_s3_bucket.raw.bucket }
output "derived_bucket" { value = aws_s3_bucket.derived.bucket }

output "raw_bucket_arn" { value = aws_s3_bucket.raw.arn }
output "derived_bucket_arn" { value = aws_s3_bucket.derived.arn }
