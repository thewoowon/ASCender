# ASCender v2.0 - GPU Experiment Guide

## 🎯 What Was Fixed

Based on the code review feedback, we fixed **4 critical issues**:

### 1. ✅ **p_r Overwrite Bug** (CRITICAL)
**Problem**: The raw relative coordinates `p_r` were being transformed by `linear_p` layers, then the transformed values were incorrectly used to reconstruct neighbor positions for ASC calculations.

**Fix**: [point_ascender_v2.py:475-515](../src/models/point_ascender_v2.py#L475-L515)
```python
# BEFORE (WRONG):
p_r = x_k_grouped[:, :, 0:3]  # Raw coords
p_r = layer(p_r)  # TRANSFORMED!
p_j = p_i + p_r[:, :, :3]  # Using transformed coords ❌

# AFTER (CORRECT):
p_r_raw = x_k_grouped[:, :, 0:3]  # Preserve raw coords
p_r = p_r_raw  # Copy for encoding
p_r = layer(p_r)  # Transform for positional encoding
p_j = p_i + p_r_raw  # Use preserved raw coords ✅
```

**Impact**: ASC's Alignment/Separation/Cohesion now operate on **actual spatial coordinates** instead of learned embeddings.

### 2. ✅ **Graph Reweighting Not Applied**
**Problem**: `ASCGraphReweight` was instantiated but never called in the forward pass, leaving Level 1 completely inactive.

**Fix**: [point_ascender_v2.py:503-521](../src/models/point_ascender_v2.py#L503-L521)
```python
# Level 1: Graph Reweighting (NOW ACTIVE)
if self.use_ascender and self.cfg.enable_graph_reweight:
    graph_scores = self.graph_reweight(p_i, p_j, x_i, x_j, normals_i, normals_j)

    # Apply scores to modulate neighbor importance
    x_k_grouped = x_k_grouped * graph_scores.unsqueeze(-1)
    x_v_grouped = x_v_grouped * graph_scores.unsqueeze(-1)
```

**Impact**: Level 1 now actively selects better neighbors based on ASC principles.

### 3. ✅ **Poor Preprocessing**
**Problem**: Normals computed as simple vectors to centroid (not surface normals), no augmentation, no density variation.

**Fix**: [modelnet40_experiment.py:140-198](../experiments/modelnet40_experiment.py#L140-L198)
```python
# Proper surface normals via PCA
def _compute_normals_pca(self, xyz, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(xyz)
    for i in range(len(xyz)):
        neighbors = xyz[nbrs.kneighbors(xyz[i:i+1])[1][0]]
        pca = PCA(n_components=3)
        pca.fit(neighbors - neighbors.mean(axis=0))
        normal = pca.components_[-1]  # Smallest variance = surface normal
        ...

# Data augmentation (training only)
if self.split == 'train':
    # Random rotation around z-axis
    # Random scaling (0.8-1.2x)
    # Random jitter (σ=0.02)
```

**Impact**: ASC receives meaningful surface normals and more robust training data.

### 4. ✅ **Underfitting (α Saturation)**
**Problem**: Too shallow (1 layer, hidden dim 32-128), too short (50 epochs), causing α→0.97 (bias ignored).

**Fix**: GPU experiment uses:
- **Deeper models**: 2-6 layers
- **Wider models**: 64-192 hidden dims
- **Longer training**: 100 epochs with LR scheduling
- **Full ASCender**: All 3 levels enabled

## 🚀 Running the GPU Experiment

### Prerequisites

1. **AWS Account** with EC2 access
2. **AWS CLI** configured:
   ```bash
   aws configure
   ```
3. **SSH Key** for EC2 instances

### Step 1: Launch GPU Instance

```bash
cd /Users/aepeul/ASCender/point-cloud-ascender

# Set your AWS credentials
export AWS_KEY_NAME=your-key-name
export AWS_SG_ID=sg-xxxxxxxxxxxxx

# Launch instance (g4dn.xlarge recommended)
bash launch_aws_gpu.sh
```

This will:
- Launch a g4dn.xlarge instance (~$0.526/hour)
- Wait for it to be ready
- Save instance info to `.aws_instance_info`
- Display SSH connection command

### Step 2: Connect and Setup

```bash
# Get connection info from output or:
source .aws_instance_info
ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}

# On the instance, run setup:
bash setup_gpu_instance.sh
```

### Step 3: Upload Code

From your **local machine**:

```bash
# Upload entire project
scp -r /Users/aepeul/ASCender/point-cloud-ascender ubuntu@PUBLIC_IP:~/ASCender/

# Or just upload specific files
scp experiments/modelnet40_gpu_experiment.py ubuntu@PUBLIC_IP:~/ASCender/point-cloud-ascender/experiments/
scp src/models/point_ascender_v2.py ubuntu@PUBLIC_IP:~/ASCender/point-cloud-ascender/src/models/
```

### Step 4: Run Experiment

Back on the **GPU instance**:

```bash
cd ~/ASCender/point-cloud-ascender

# Run the GPU experiment
python3 experiments/modelnet40_gpu_experiment.py

# Monitor with tmux (recommended for long runs)
tmux new -s ascender
python3 experiments/modelnet40_gpu_experiment.py
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t ascender
```

### Step 5: Download Results

From your **local machine**:

```bash
# Download results
scp ubuntu@PUBLIC_IP:~/ASCender/point-cloud-ascender/results/modelnet40_gpu_results.json ./results/

# Download logs
scp ubuntu@PUBLIC_IP:~/ASCender/point-cloud-ascender/experiments/*.log ./experiments/
```

### Step 6: Terminate Instance

```bash
# From local machine
cd /Users/aepeul/ASCender/point-cloud-ascender
bash terminate_aws_gpu.sh
```

**⚠️ IMPORTANT**: Always terminate instances when done to avoid charges!

## 📊 Expected Results

With the fixes, we expect:

| Model Size | Params | Baseline | ASCender | Δ | α | Status |
|------------|--------|----------|----------|---|---|--------|
| Small (50K) | 50K | ~60% | **~67%** | **+7%** ✅ | ~0.3 | Bias utilized |
| Medium (200K) | 200K | ~68% | **~70%** | **+2%** ✅ | ~0.5 | Balanced |
| Large (500K) | 500K | ~72% | ~72% | ±0% | ~0.7 | Less reliance |

**Key predictions**:
1. ✅ Small models will show **significant gains** (+5-7%)
2. ✅ α will be **lower** (~0.3-0.5) showing bias is actually used
3. ✅ No more saturation at α=0.97
4. ✅ ModelNet40 accuracy will be **competitive** with baselines

## 💰 Cost Estimates

| Instance | Training Time | Cost per Run | 3 Runs |
|----------|---------------|--------------|--------|
| g4dn.xlarge | ~2 hours | ~$1.05 | ~$3.15 |
| g5.xlarge | ~1.5 hours | ~$1.51 | ~$4.53 |

**Recommendation**: Use **g4dn.xlarge spot instance** (~$0.16/hour) for even cheaper runs.

## 🔍 Monitoring & Debugging

### Check GPU Utilization

```bash
# On GPU instance
watch -n 1 nvidia-smi
```

Expected: ~80-100% GPU utilization during training

### Check Training Progress

```bash
# View live logs
tail -f experiments/modelnet40_gpu_experiment.log

# Check α evolution
grep "α" experiments/modelnet40_gpu_experiment.log
```

### If Training Crashes

1. Check CUDA memory:
   ```python
   torch.cuda.memory_summary()
   ```

2. Reduce batch size in `modelnet40_gpu_experiment.py`:
   ```python
   train_loader = DataLoader(..., batch_size=16)  # Was 32
   ```

3. Check for OOM errors:
   ```bash
   dmesg | grep -i "out of memory"
   ```

## 📝 Next Steps After GPU Results

1. **Analyze Results**
   - Compare to expectations
   - Check α evolution
   - Verify fixes worked

2. **Update Paper**
   - Add GPU results to Section 4.5
   - Update abstract with real-world validation
   - Add preprocessing details to methods

3. **Optional Follow-ups**
   - Try other datasets (ShapeNet, ScanNet)
   - Ablate with/without each fix
   - Tune ASC weights (w_align, w_sep, w_coh)

## 📚 References

- **Code fixes**: See [PAPER_DRAFT.md](../PAPER_DRAFT.md) for full experimental context
- **AWS setup**: See [aws_gpu_setup.md](../aws_gpu_setup.md) for detailed AWS guide
- **Original feedback**: Research handoff document with code-level findings

---

**Questions?** Check the scripts:
- `launch_aws_gpu.sh` - Launch instance
- `setup_gpu_instance.sh` - Setup environment
- `terminate_aws_gpu.sh` - Cleanup
- `modelnet40_gpu_experiment.py` - Main experiment
