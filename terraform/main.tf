# main.tf
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0.0"
    }
    cloudinit = {
      source  = "hashicorp/cloudinit"
      version = ">= 2.0.0"
    }
  }
}
provider "aws" { region = var.region }

variable "region" { default = "us-east-1" }     # 가용성/스팟 유리 리전
variable "subnet_id" {}                         # 기존 퍼블릭 서브넷
variable "security_group" {}                    # SSH 22, 필요시 6006 등
variable "key_name" {}                          # EC2 키페어
variable "bucket_name" {}                       # S3 버킷명 (전역 유일)
variable "gpu_ami" { default = "ami-xxxxxxxx" } # Deep Learning AMI (GPU, PyTorch) 최신
variable "cpu_ami" { default = "ami-xxxxxxxx" } # Amazon Linux 2023 등
variable "instance_profile_name" { default = "AscenderEC2Profile" }

# S3 bucket (lifecycle: 30일 후 IA, 90일 후 Glacier Instant Retrieval)
resource "aws_s3_bucket" "asc" { bucket = var.bucket_name }
resource "aws_s3_bucket_lifecycle_configuration" "asc_lc" {
  bucket = aws_s3_bucket.asc.id
  rule {
    id     = "tiering"
    status = "Enabled"
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    transition {
      days          = 90
      storage_class = "GLACIER_IR"
    }
  }
}

# IAM role for EC2 to access S3 + CW logs
resource "aws_iam_role" "ec2_role" {
  name = "AscenderEC2Role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17",
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}
resource "aws_iam_policy" "ec2_s3_cw" {
  name = "AscenderS3CWPolicy"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      { Effect = "Allow", Action = ["s3:*"], Resource = [
        "arn:aws:s3:::${var.bucket_name}", "arn:aws:s3:::${var.bucket_name}/*"
      ] },
      { Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "*" }
    ]
  })
}
resource "aws_iam_role_policy_attachment" "attach" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.ec2_s3_cw.arn
}
resource "aws_iam_instance_profile" "ec2_profile" {
  name = var.instance_profile_name
  role = aws_iam_role.ec2_role.name
}

# GPU Launch Template (DLAMI + user_data)
data "cloudinit_config" "gpu_userdata" {
  gzip          = false
  base64_encode = true
  part {
    content_type = "text/x-shellscript"
    content = templatefile("${path.module}/userdata_gpu.sh", {
      BUCKET = var.bucket_name
    })
  }
}
resource "aws_launch_template" "gpu_lt" {
  name_prefix   = "asc-gpu-"
  image_id      = var.gpu_ami
  instance_type = "g5.xlarge"
  key_name      = var.key_name
  iam_instance_profile { name = var.instance_profile_name }
  network_interfaces {
    subnet_id                   = var.subnet_id
    security_groups             = [var.security_group]
    associate_public_ip_address = true
  }
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 200
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }
  user_data = data.cloudinit_config.gpu_userdata.rendered
}

# Spot Fleet: capacity-optimized
resource "aws_autoscaling_group" "gpu_asg" {
  name                = "asc-gpu-asg"
  max_size            = 1
  min_size            = 0
  desired_capacity    = 1
  vpc_zone_identifier = [var.subnet_id]
  health_check_type   = "EC2"
  mixed_instances_policy {
    instances_distribution { spot_allocation_strategy = "capacity-optimized" }
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.gpu_lt.id
        version            = "$Latest"
      }
      override { instance_type = "g5.xlarge" }
    }
  }
}

# Orchestrator (on-demand)
data "cloudinit_config" "cpu_userdata" {
  gzip          = false
  base64_encode = true
  part {
    content_type = "text/x-shellscript"
    content      = file("${path.module}/userdata_cpu.sh")
  }
}
resource "aws_instance" "orchestrator" {
  ami                    = var.cpu_ami
  instance_type          = "c7i.large"
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group]
  key_name               = var.key_name
  iam_instance_profile   = var.instance_profile_name
  user_data_base64       = data.cloudinit_config.cpu_userdata.rendered
  tags                   = { Name = "asc-orchestrator" }
}
