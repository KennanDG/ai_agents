terraform {
  backend "s3" {
    bucket         = "ai-agents-tfstate-dev"
    key            = "infra/dev/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "ai-agents-tflock-dev"
    encrypt        = true
  }
}