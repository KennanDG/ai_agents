terraform {
  backend "s3" {
    bucket         = "ai-agents-tfstate-dev"
    key            = "infra/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "ai-agents-tf-lock-dev"
    encrypt        = true
  }
}