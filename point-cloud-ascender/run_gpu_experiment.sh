#!/bin/bash
# One-command script to run GPU experiment using existing Terraform setup

set -e

TERRAFORM_DIR="/Users/aepeul/ASCender/terraform"
BUCKET="ascender-research-20251005"
PROJECT_DIR="/Users/aepeul/ASCender/point-cloud-ascender"

echo "🚀 ASCender GPU Experiment Launcher"
echo "="*70
echo ""

# Step 1: Upload code to S3
echo "📤 Step 1/4: Uploading code to S3..."
aws s3 sync "$PROJECT_DIR/" "s3://$BUCKET/code/point-cloud-ascender/" \
    --exclude ".git/*" \
    --exclude "__pycache__/*" \
    --exclude "*.pyc" \
    --exclude "data/*" \
    --exclude "results/*" \
    --exclude "logs/*" \
    --quiet

echo "✅ Code uploaded"
echo ""

# Step 2: Check Terraform
echo "🔧 Step 2/4: Checking Terraform setup..."
cd "$TERRAFORM_DIR"

if ! terraform validate -no-color > /dev/null 2>&1; then
    echo "❌ Terraform validation failed. Running terraform init..."
    terraform init
fi

echo "✅ Terraform ready"
echo ""

# Step 3: Launch GPU instance
echo "🖥️  Step 3/4: Launching GPU instance..."
terraform apply -auto-approve -no-color

echo "✅ Instance launched"
echo ""

# Step 4: Get instance info
echo "📍 Step 4/4: Getting instance information..."
sleep 10  # Wait for instance to be fully registered

INSTANCE_INFO=$(aws ec2 describe-instances \
    --filters "Name=tag-key,Values=aws:autoscaling:groupName" \
              "Name=tag-value,Values=asc-gpu-asg" \
              "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].[InstanceId,PublicIpAddress,InstanceType,State.Name]" \
    --output text 2>/dev/null || echo "")

if [ -z "$INSTANCE_INFO" ]; then
    echo "⏳ Instance is starting... Please wait 2-3 minutes then check:"
    echo ""
    echo "   aws ec2 describe-instances \\"
    echo "       --filters 'Name=tag:aws:autoscaling:groupName,Values=asc-gpu-asg' \\"
    echo "                 'Name=instance-state-name,Values=running' \\"
    echo "       --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \\"
    echo "       --output text"
    echo ""
else
    INSTANCE_ID=$(echo "$INSTANCE_INFO" | awk '{print $1}')
    PUBLIC_IP=$(echo "$INSTANCE_INFO" | awk '{print $2}')
    INSTANCE_TYPE=$(echo "$INSTANCE_INFO" | awk '{print $3}')

    echo "✅ Instance running!"
    echo ""
    echo "📋 Instance Details:"
    echo "   ID:   $INSTANCE_ID"
    echo "   IP:   $PUBLIC_IP"
    echo "   Type: $INSTANCE_TYPE"
    echo ""
    echo "🔗 To connect (wait ~2 minutes for setup to complete):"
    echo "   ssh -i ~/.ssh/ascender-key.pem ubuntu@$PUBLIC_IP"
    echo ""
    echo "🏃 To run experiment:"
    echo "   cd /home/ubuntu/asc/code/point-cloud-ascender"
    echo "   conda activate asc"
    echo "   pip install scikit-learn h5py"
    echo "   python experiments/modelnet40_gpu_experiment.py"
    echo ""
    echo "📊 To monitor:"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "💾 Results will be auto-saved to S3 on spot interruption"
    echo ""

    # Save info for cleanup script
    cat > .gpu_instance_info << EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
INSTANCE_TYPE=$INSTANCE_TYPE
TERRAFORM_DIR=$TERRAFORM_DIR
BUCKET=$BUCKET
EOF

    echo "💡 Info saved to .gpu_instance_info"
    echo ""
fi

echo "🛑 To stop instance when done:"
echo "   bash stop_gpu_experiment.sh"
echo ""
echo "✨ All done!"
