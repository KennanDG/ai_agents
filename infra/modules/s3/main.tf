resource "aws_s3_bucket" "raw" {
  bucket = "${var.name}-raw"
}

resource "aws_s3_bucket" "derived" {
  bucket = "${var.name}-derived"
}

resource "aws_s3_bucket_versioning" "raw" {
  bucket = aws_s3_bucket.raw.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_versioning" "derived" {
  bucket = aws_s3_bucket.derived.id
  versioning_configuration { status = "Enabled" }
}
