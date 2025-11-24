#!/bin/bash
# Setup script to run ON the AWS GPU instance

set -e

echo "🔧 Setting up ASCender on GPU instance..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install basic tools
echo "🛠️  Installing build tools..."
sudo apt-get install -y git wget build-essential > /dev/null

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip3 install --quiet --upgrade pip

# Install PyTorch with CUDA 11.8
echo "🔥 Installing PyTorch with CUDA..."
pip3 install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install scientific computing packages
echo "📊 Installing scientific packages..."
pip3 install --quiet numpy scipy scikit-learn h5py tqdm

# Verify GPU
echo ""
echo "🎮 GPU Information:"
python3 -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Clone or update repository
echo ""
echo "📥 Setting up ASCender repository..."
if [ -d "ASCender" ]; then
    echo "  Repository exists, updating..."
    cd ASCender/point-cloud-ascender
    git pull
else
    echo "  Cloning repository..."
    echo "  ⚠️  Note: You'll need to upload your local changes manually"
    echo "  Use: scp -r point-cloud-ascender ubuntu@PUBLIC_IP:~/ASCender/"
    mkdir -p ASCender/point-cloud-ascender
    cd ASCender/point-cloud-ascender
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📍 Current directory: $(pwd)"
echo ""
echo "🚀 To upload your local code:"
echo "   # From your local machine:"
echo "   scp -r /Users/aepeul/ASCender/point-cloud-ascender ubuntu@\$PUBLIC_IP:~/ASCender/"
echo ""
echo "🏃 To run experiment:"
echo "   python3 experiments/modelnet40_experiment.py"
