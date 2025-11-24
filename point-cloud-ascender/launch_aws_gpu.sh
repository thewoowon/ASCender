#!/bin/bash
# Launch AWS GPU instance for ASCender training

set -e

echo "🚀 Launching AWS GPU Instance for ASCender..."

# Configuration
INSTANCE_TYPE="g4dn.xlarge"
AMI_ID="ami-0c7217cdde317cfec"  # Deep Learning AMI (Ubuntu 20.04)
REGION="us-east-1"
KEY_NAME="${AWS_KEY_NAME:-your-key-name}"  # Set AWS_KEY_NAME env var or replace
SECURITY_GROUP="${AWS_SG_ID:-sg-xxxxxxxx}"  # Set AWS_SG_ID env var or replace
VOLUME_SIZE=100

echo "📋 Configuration:"
echo "  Instance Type: $INSTANCE_TYPE"
echo "  Region: $REGION"
echo "  Key Name: $KEY_NAME"
echo ""

# Check if key name and SG are set
if [[ "$KEY_NAME" == "your-key-name" ]] || [[ "$SECURITY_GROUP" == "sg-xxxxxxxx" ]]; then
    echo "❌ Error: Please set AWS_KEY_NAME and AWS_SG_ID environment variables"
    echo "   export AWS_KEY_NAME=your-actual-key-name"
    echo "   export AWS_SG_ID=sg-xxxxxxxxxxxxx"
    exit 1
fi

# Launch instance
echo "🔄 Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SECURITY_GROUP \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ASCender-GPU-$(date +%Y%m%d-%H%M%S)}]" \
    --region $REGION \
    --query "Instances[0].InstanceId" \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"
echo "⏳ Waiting for instance to be running..."

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text \
    --region $REGION)

echo "✅ Instance running!"
echo ""
echo "📍 Instance ID: $INSTANCE_ID"
echo "🌐 Public IP: $PUBLIC_IP"
echo ""
echo "⏳ Waiting 30 seconds for SSH to be ready..."
sleep 30

# Save instance info
cat > .aws_instance_info << EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
REGION=$REGION
KEY_NAME=$KEY_NAME
EOF

echo "✅ Instance info saved to .aws_instance_info"
echo ""
echo "🔗 To connect:"
echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo ""
echo "📦 Setup script:"
echo "   bash setup_gpu_instance.sh"
echo ""
echo "🛑 To terminate:"
echo "   bash terminate_aws_gpu.sh"
