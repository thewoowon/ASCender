# Terraform Configuration for Parallel ASCender Experiments
# Launches individual spot instances for each experiment

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# ============================================================================
# Variables
# ============================================================================

variable "region" { default = "us-east-1" }
variable "subnet_id" {}
variable "security_group" {}
variable "key_name" {}
variable "bucket_name" {}
variable "gpu_ami" { default = "ami-003b184e823d3e894" }
variable "instance_profile_name" { default = "AscenderEC2Profile" }

# ============================================================================
# Experiment Configurations
# ============================================================================

locals {
  seeds = [42, 123, 456, 789, 2024]
  hidden_dims = [32, 48, 64, 80]

  # Generate all 40 main experiments
  main_experiments = flatten([
    for h in local.hidden_dims : [
      for seed in local.seeds : [
        {
          name = "h${h}_baseline_seed${seed}"
          hidden_dim = h
          use_ascender = ""
          seed = seed
          exp_type = "main"
        },
        {
          name = "h${h}_ascender_seed${seed}"
          hidden_dim = h
          use_ascender = "--use-ascender"
          seed = seed
          exp_type = "main"
        }
      ]
    ]
  ])

  # Add 2 gradient analysis experiments
  gradient_experiments = [
    {
      name = "gradient_h48_seed42"
      hidden_dim = 48
      use_ascender = ""
      seed = 42
      exp_type = "gradient"
    },
    {
      name = "gradient_h64_seed42"
      hidden_dim = 64
      use_ascender = ""
      seed = 42
      exp_type = "gradient"
    }
  ]

  all_experiments = concat(local.main_experiments, local.gradient_experiments)
}

# ============================================================================
# Individual Spot Instance for Each Experiment
# ============================================================================

resource "aws_spot_instance_request" "experiment" {
  count = length(local.all_experiments)

  ami                    = var.gpu_ami
  instance_type          = "g4dn.xlarge"
  key_name               = var.key_name
  subnet_id              = var.subnet_id
  vpc_security_group_ids = [var.security_group]
  iam_instance_profile   = var.instance_profile_name

  # Spot instance configuration
  spot_price                      = "0.60"  # Max price
  wait_for_fulfillment           = false
  spot_type                      = "one-time"
  instance_interruption_behavior = "terminate"

  # User data for this specific experiment
  user_data = templatefile("${path.module}/userdata_single_exp.sh", {
    BUCKET       = var.bucket_name
    EXP_NAME     = local.all_experiments[count.index].name
    HIDDEN_DIM   = local.all_experiments[count.index].hidden_dim
    USE_ASCENDER = local.all_experiments[count.index].use_ascender
    SEED         = local.all_experiments[count.index].seed
    EXP_TYPE     = local.all_experiments[count.index].exp_type
  })

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name       = "asc-${local.all_experiments[count.index].name}"
    Project    = "ASCender"
    Experiment = local.all_experiments[count.index].name
    ExpType    = local.all_experiments[count.index].exp_type
  }
}

# ============================================================================
# Outputs
# ============================================================================

output "total_experiments" {
  value = length(local.all_experiments)
}

output "spot_request_ids" {
  value = aws_spot_instance_request.experiment[*].id
}

output "experiment_names" {
  value = [for exp in local.all_experiments : exp.name]
}

output "s3_results_path" {
  value = "s3://${var.bucket_name}/results/"
}

output "monitoring_commands" {
  value = <<-EOT
    # Check spot instance status
    aws ec2 describe-spot-instance-requests --region ${var.region}

    # Monitor S3 results
    watch -n 10 'aws s3 ls s3://${var.bucket_name}/results/completion_markers/ | wc -l'

    # Download results when complete
    aws s3 sync s3://${var.bucket_name}/results/ ./results/
  EOT
}
