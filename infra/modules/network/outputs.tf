output "vpc_id" { value = aws_vpc.this.id }
output "public_subnet_ids" { value = [for s in aws_subnet.public : s.id] }

output "lambda_sg_id" { value = aws_security_group.lambda.id }
output "worker_sg_id" { value = aws_security_group.worker.id }


output "alb_sg_id" { value = aws_security_group.alb.id }
output "api_sg_id" { value = aws_security_group.api.id }