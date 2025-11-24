#!/bin/bash
# Stop GPU experiment and cleanup

set -e

if [ ! -f ".gpu_instance_info" ]; then
    echo "❌ No instance info found. Using Terraform directly..."
    echo ""
    read -p "Stop all GPU instances? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Cancelled"
        exit 1
    fi

    cd /Users/aepeul/ASCender/terraform
    terraform apply -var="gpu_asg_desired_capacity=0" -auto-approve
    echo "✅ GPU Auto Scaling Group scaled to 0"
    exit 0
fi

source .gpu_instance_info

echo "🛑 Stopping GPU Experiment"
echo "="*70
echo ""
echo "📋 Instance to stop:"
echo "   ID:   $INSTANCE_ID"
echo "   IP:   $PUBLIC_IP"
echo "   Type: $INSTANCE_TYPE"
echo ""

read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Step 1: Download results (if not already done)
echo "💾 Step 1/3: Downloading results from S3..."
aws s3 sync "s3://$BUCKET/results/modelnet40/" results/ --quiet || true
aws s3 sync "s3://$BUCKET/logs/modelnet40/" logs/ --quiet || true
echo "✅ Results downloaded"
echo ""

# Step 2: Stop instance via Terraform
echo "🛑 Step 2/3: Stopping instance..."
cd "$TERRAFORM_DIR"

# Scale ASG to 0
cat > temp_vars.tf << EOF
variable "gpu_asg_desired_capacity" {
  default = 0
}
EOF

terraform apply -var="gpu_asg_desired_capacity=0" -auto-approve -no-color
rm -f temp_vars.tf

echo "✅ Auto Scaling Group scaled to 0"
echo "⏳ Waiting for instance to terminate..."
sleep 5

# Step 3: Verify termination
echo "🔍 Step 3/3: Verifying termination..."
STATE=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].State.Name" \
    --output text 2>/dev/null || echo "terminated")

echo "   Instance state: $STATE"

if [ "$STATE" = "terminated" ] || [ "$STATE" = "shutting-down" ]; then
    echo "✅ Instance terminated successfully"
    rm .gpu_instance_info
    echo "🧹 Cleaned up instance info"
else
    echo "⚠️  Instance is in state: $STATE"
    echo "   It may take a few minutes to fully terminate."
fi

echo ""
echo "📊 Results location:"
echo "   Local:  $(pwd)/results/"
echo "   S3:     s3://$BUCKET/results/modelnet40/"
echo ""
echo "💰 Estimated cost saved by stopping now!"
echo ""
echo "✨ Done!"
