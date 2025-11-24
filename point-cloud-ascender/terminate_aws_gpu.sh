#!/bin/bash
# Terminate AWS GPU instance

set -e

if [ ! -f ".aws_instance_info" ]; then
    echo "❌ Error: No instance info found (.aws_instance_info)"
    echo "   Please provide instance ID manually:"
    echo "   aws ec2 terminate-instances --instance-ids i-xxxxx --region us-east-1"
    exit 1
fi

# Load instance info
source .aws_instance_info

echo "🛑 Terminating AWS GPU Instance..."
echo "  Instance ID: $INSTANCE_ID"
echo "  Region: $REGION"
echo ""

read -p "Are you sure you want to terminate this instance? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Termination cancelled"
    exit 1
fi

# Terminate instance
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION > /dev/null

echo "✅ Instance $INSTANCE_ID termination requested"
echo "⏳ Waiting for termination..."

# Wait for termination
aws ec2 wait instance-terminated --instance-ids $INSTANCE_ID --region $REGION

echo "✅ Instance terminated successfully!"

# Remove instance info
rm .aws_instance_info

echo "🧹 Cleaned up instance info file"
