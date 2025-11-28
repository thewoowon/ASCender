# AWS Parallel Experiments - Quick Start Guide

## 🚀 Complete Statistical Validation in 3 Hours

This guide will help you run **42 parallel GPU experiments** on AWS to complete the statistical validation for ASCender v2.0.

### 📊 What Will Be Run

| Experiment Type | Count | Description |
|----------------|-------|-------------|
| h=32 (baseline + ASCender) | 10 | 5 seeds × 2 configs |
| h=48 (baseline + ASCender) | 10 | 5 seeds × 2 configs |
| h=64 (baseline + ASCender) | 10 | 5 seeds × 2 configs |
| h=80 (baseline + ASCender) | 10 | 5 seeds × 2 configs |
| Gradient Analysis (h=48, h=64) | 2 | Medium-scale degradation analysis |
| **TOTAL** | **42** | |

### 💰 Cost Estimate

- **Instance type**: g4dn.xlarge (1x NVIDIA T4 GPU, 16GB)
- **Spot price**: ~$0.16/hour (vs $0.526 on-demand)
- **Duration**: 2-3 hours per experiment
- **Total cost**: **$20-35** (42 instances × 2.5 hrs × $0.16)

---

## ⚡ Quick Start (3 Steps)

### Step 1: Prepare Data

```bash
# Make sure preprocessed data exists
ls data/ModelNet40/preprocessed/train/sample_0000.pt

# If not, run preprocessing
python experiments/preprocess_modelnet40.py
```

### Step 2: Launch Experiments

```bash
# Simple one-command launch
./launch_aws_experiments.sh
```

This will:
1. ✅ Upload data/code to S3 (~5 min)
2. ✅ Launch 42 GPU spot instances
3. ✅ Run experiments in parallel
4. ✅ Auto-shutdown when complete
5. ✅ Download results

### Step 3: Review Results

```bash
# Check completion status
aws s3 ls s3://ascender-research-20251005/results/completion_markers/ | wc -l

# View statistical summary
cat results/statistical_validation/summary.json

# Update paper with new numbers
python experiments/update_paper_stats.py
```

---

## 📡 Monitoring Progress

### Check How Many Experiments Completed

```bash
watch -n 10 'aws s3 ls s3://ascender-research-20251005/results/completion_markers/ | wc -l'
```

Shows: `X / 42` completed

### View Individual Results

```bash
# List completed experiments
aws s3 ls s3://ascender-research-20251005/results/completion_markers/

# Download specific result
aws s3 cp s3://ascender-research-20251005/results/statistical_validation/h32_ascender_seed42.json .

# Download all results
aws s3 sync s3://ascender-research-20251005/results/ ./results/
```

### Check Spot Instance Status

```bash
# Active instances
aws ec2 describe-spot-instance-requests \
  --region us-east-1 \
  --filters Name=state,Values=active,open \
  --query 'SpotInstanceRequests[*].[SpotInstanceRequestId,State,InstanceId]' \
  --output table

# View console logs (for debugging)
aws ec2 get-console-output --instance-id <INSTANCE_ID> --region us-east-1
```

---

## 🔧 Manual Control (Advanced)

### Launch Using Terraform Directly

```bash
cd terraform/

# Initialize
terraform init

# Plan (preview)
terraform plan -var-file=terraform.tfvars main_parallel_v2.tf

# Apply (launch instances)
terraform apply -var-file=terraform.tfvars main_parallel_v2.tf

# Destroy (cleanup)
terraform destroy -var-file=terraform.tfvars main_parallel_v2.tf
```

### Run Single Experiment Locally (Testing)

```bash
# Test h=32 baseline with seed 42
python experiments/run_single_seed.py \
  --hidden-dim 32 \
  --seed 42 \
  --preprocessed-dir data/ModelNet40/preprocessed \
  --output-dir results/test \
  --epochs 5  # Quick test

# Test gradient analysis
python experiments/run_gradient_analysis.py \
  --hidden-dim 48 \
  --seed 42 \
  --preprocessed-dir data/ModelNet40/preprocessed \
  --output-dir results/test \
  --epochs 5
```

---

## 📈 Expected Results

### After Completion

You should have:

```
results/
├── statistical_validation/
│   ├── h32_baseline_seed42.json
│   ├── h32_ascender_seed42.json
│   ├── h32_baseline_seed123.json
│   ├── h32_ascender_seed123.json
│   ├── ... (40 total)
│   └── summary.json
├── gradient_analysis/
│   ├── gradient_analysis_h48_seed42.json
│   └── gradient_analysis_h64_seed42.json
└── completion_markers/
    ├── h32_baseline_seed42.json
    ├── ... (42 total)
```

### Summary Statistics

The `summary.json` will contain:

```json
{
  "h32": {
    "baseline": {
      "mean": 0.7484,
      "std": 0.0080,
      "individual_results": [...]
    },
    "ascender": {
      "mean": 0.7705,
      "std": 0.0095,
      "individual_results": [...]
    },
    "paired_ttest": {
      "p_value": 0.038,
      "cohens_d": 1.36
    }
  },
  "h48": { ... },
  "h64": { ... },
  "h80": { ... }
}
```

---

## 🛠️ Troubleshooting

### Spot Instance Interrupted

Spot instances may be interrupted if AWS needs capacity. If this happens:

```bash
# Check which experiments failed
python experiments/check_missing_experiments.py

# Re-launch only missing experiments
terraform apply -target=aws_spot_instance_request.experiment[X]
```

### Instance Won't Start

Common issues:

1. **Quota limit**: Request quota increase in AWS Console
   - Service Quotas → EC2 → Spot instances
   - Increase vCPU limit for G instances

2. **Spot capacity unavailable**: Try different region
   ```bash
   terraform apply -var="region=us-west-2"
   ```

3. **AMI not found**: Update GPU AMI ID in `terraform.tfvars`
   ```bash
   # Find latest Deep Learning AMI
   aws ec2 describe-images \
     --owners amazon \
     --filters "Name=name,Values=Deep Learning AMI GPU PyTorch*" \
     --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId'
   ```

### Data Upload Slow

If S3 upload is slow:

```bash
# Enable transfer acceleration
aws s3api put-bucket-accelerate-configuration \
  --bucket ascender-research-20251005 \
  --accelerate-configuration Status=Enabled

# Upload with acceleration
aws s3 sync data/ s3://ascender-research-20251005/data/ \
  --endpoint-url https://ascender-research-20251005.s3-accelerate.amazonaws.com
```

---

## 🧹 Cleanup

### After Experiments Complete

```bash
# 1. Download results
aws s3 sync s3://ascender-research-20251005/results/ ./results/

# 2. Destroy instances (if not auto-shutdown)
cd terraform/
terraform destroy -var-file=terraform.tfvars main_parallel_v2.tf

# 3. (Optional) Delete S3 data to save cost
aws s3 rm s3://ascender-research-20251005/data/ --recursive
```

### Cost Optimization

- ✅ Instances auto-shutdown after completion
- ✅ Spot instances (~70% cheaper than on-demand)
- ✅ S3 Lifecycle: STANDARD_IA after 30 days
- ✅ EBS volumes deleted on termination

---

## 📞 Support

If you encounter issues:

1. Check logs: `cat /var/log/user-data.log` on instance
2. Review S3 completion markers for error details
3. Re-run failed experiments individually

---

**Ready to launch?**

```bash
./launch_aws_experiments.sh
```

Sit back and grab a coffee ☕ - your 42 experiments will complete in ~3 hours!
