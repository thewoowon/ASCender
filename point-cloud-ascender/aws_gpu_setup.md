# AWS GPU Setup for ModelNet40 Training

## Recommended Instance Types

### Option 1: g4dn.xlarge (Most Cost-Effective)
- **GPU**: 1x NVIDIA T4 (16GB VRAM)
- **vCPU**: 4
- **RAM**: 16GB
- **Cost**: ~$0.526/hour (us-east-1 on-demand)
- **Best for**: Small-medium models, testing

### Option 2: g5.xlarge (Better Performance)
- **GPU**: 1x NVIDIA A10G (24GB VRAM)
- **vCPU**: 4
- **RAM**: 16GB
- **Cost**: ~$1.006/hour (us-east-1 on-demand)
- **Best for**: Larger models, faster training

### Option 3: p3.2xlarge (High Performance)
- **GPU**: 1x NVIDIA V100 (16GB VRAM)
- **vCPU**: 8
- **RAM**: 61GB
- **Cost**: ~$3.06/hour (us-east-1 on-demand)
- **Best for**: Large-scale experiments

## Quick Launch Commands

### 1. Check Available Instances
```bash
aws ec2 describe-instance-type-offerings \
    --location-type availability-zone \
    --filters Name=instance-type,Values=g4dn.xlarge \
    --region us-east-1 \
    --query "InstanceTypeOfferings[].Location" \
    --output table
```

### 2. Launch g4dn.xlarge Instance
```bash
# Using Deep Learning AMI (Ubuntu 20.04)
aws ec2 run-instances \
    --image-id ami-0c7217cdde317cfec \
    --instance-type g4dn.xlarge \
    --key-name YOUR_KEY_NAME \
    --security-group-ids YOUR_SG_ID \
    --subnet-id YOUR_SUBNET_ID \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ASCender-GPU}]' \
    --region us-east-1
```

### 3. Connect to Instance
```bash
# Get instance public IP
INSTANCE_ID=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=ASCender-GPU" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].InstanceId" \
    --output text \
    --region us-east-1)

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text \
    --region us-east-1)

# SSH into instance
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@$PUBLIC_IP
```

## Setup Script for GPU Instance

Once connected, run:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone repository
cd ~
git clone https://github.com/YOUR_USERNAME/ASCender.git
cd ASCender/point-cloud-ascender

# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip3 install -r requirements.txt

# Verify GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run experiment
python3 experiments/modelnet40_experiment.py
```

## Cost Estimates

For ModelNet40 training (~2-3 hours on GPU):

| Instance Type | Training Time | Cost per Run | Monthly (1 run/day) |
|---------------|---------------|--------------|---------------------|
| g4dn.xlarge   | ~2 hours      | ~$1.05       | ~$31.50             |
| g5.xlarge     | ~1.5 hours    | ~$1.51       | ~$45.30             |
| p3.2xlarge    | ~1 hour       | ~$3.06       | ~$91.80             |

## Alternative: Spot Instances (70% Cheaper)

Use spot instances for even lower cost:

```bash
aws ec2 request-spot-instances \
    --spot-price "0.20" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification '{
        "ImageId": "ami-0c7217cdde317cfec",
        "InstanceType": "g4dn.xlarge",
        "KeyName": "YOUR_KEY_NAME",
        "SecurityGroupIds": ["YOUR_SG_ID"],
        "SubnetId": "YOUR_SUBNET_ID"
    }' \
    --region us-east-1
```

**Recommendation**: Start with **g4dn.xlarge** spot instance (~$0.16/hour) for cost-effective testing.

## Cleanup

Don't forget to terminate instances when done:

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
```
