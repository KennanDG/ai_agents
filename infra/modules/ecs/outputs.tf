output "cluster_name" { value = aws_ecs_cluster.this.name }
output "service_name" { value = aws_ecs_service.worker.name }
output "api_base_url" { value = "http://${aws_lb.api.dns_name}" }

