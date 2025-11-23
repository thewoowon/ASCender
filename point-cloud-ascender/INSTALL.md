# 🛠️ Installation Guide

## Quick Setup (Recommended)

### Option 1: Using existing conda environment

```bash
# Check if you have conda
which conda

# If yes, create new environment
conda create -n point-ascender python=3.10 -y
conda activate point-ascender

# Install PyTorch (CPU version for quick testing)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### Option 2: Using system Python with venv

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### Option 3: Quick test without full install

**Just install PyTorch for now:**

```bash
# Using pip directly
pip3 install torch numpy

# Or if you have conda
conda install pytorch -c pytorch
```

---

## Verification

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed!')"
```

Expected output:
```
PyTorch 2.x.x installed!
```

---

## Run Tests

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender

# Test 1: Minimal (already passed ✅)
python3 experiments/test_ascender_minimal.py

# Test 2: RBP Learning (next)
python3 experiments/test_rbp_learning.py
```

---

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip3 install torch
```

### Issue: "No module named 'src'"

**Solution:** Make sure you're running from project root:
```bash
cd /Users/aepeul/ASCender/point-cloud-ascender
python3 experiments/test_rbp_learning.py
```

### Issue: Conda not found

**Solution:** Install Miniconda or use venv (Option 2)

---

## Minimal Requirements (for testing only)

```
torch>=2.0.0
numpy>=1.24.0
```

## Full Requirements (for MSRAction3D experiments)

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
open3d>=0.17.0
h5py>=3.8.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## Next Steps After Installation

1. ✅ Verify PyTorch: `python3 -c "import torch; print(torch.__version__)"`
2. ✅ Run minimal test: `python3 experiments/test_ascender_minimal.py`
3. 🚀 Run RBP learning: `python3 experiments/test_rbp_learning.py`
4. 📊 Analyze results
5. 🎯 Decide next direction (MSRAction3D or adjust parameters)
