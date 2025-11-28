# ModelNet40 Preprocessing Status

## Overview

Solving the critical bottleneck that prevented ModelNet40 experiments: **PCA and k-NN graph computation in training loop** caused 7-8 sec/batch, making experiments infeasible.

## Solution: Preprocessing Pipeline

Pre-compute expensive operations once and save to disk:

1. **PCA-based surface normals** (k=10 neighbors)
2. **k-NN neighbor indices** (k=8 neighbors)
3. **Local density** (log-based, stable)
4. **Point cloud normalization**

## Current Progress

### Verification Tests ✅

- [x] Individual function tests (normals, k-NN, density) - PASSED
- [x] Full pipeline test on synthetic data - PASSED
- [x] ModelNet40 data loading - PASSED
- [x] Real sample preprocessing (5 samples) - PASSED
- [x] Speed verification: 0.217s per sample ✅

### Full Preprocessing 🔄

**Status**: Running in background (started 2025-11-25)

**Dataset**:
- Train: 9,840 samples
- Test: 2,468 samples
- Total: 12,308 samples

**Speed**: ~4.6 samples/sec

**Estimated Time**:
- Train: ~36 minutes
- Test: ~9 minutes
- **Total: ~45 minutes**

**Current**: Processing training data (316/9840 completed at last check)

## Expected Speedup

### Before (with in-loop computation):
- Batch processing: **7-8 seconds/batch**
- Single epoch (ModelNet40, batch_size=32): ~40 minutes
- 50 epochs: **~33 hours** ❌ INFEASIBLE

### After (with preprocessing):
- Batch processing: **<1 second/batch**
- Single epoch: ~5 minutes
- 50 epochs: **~4 hours** ✅ FEASIBLE

**Speedup: 10x improvement** 🚀

## Storage Cost

**Per sample**: ~0.15 MB (estimated)
**Total storage**: ~1.8 GB for full dataset

Breakdown:
- xyz: (1024, 3) float32 = 12 KB
- normals: (1024, 3) float32 = 12 KB
- neighbor_indices: (1024, 8) int64 = 65 KB
- neighbor_distances: (1024, 8) float32 = 33 KB
- density: (1024,) float32 = 4 KB
- Total: ~126 KB per sample

## Next Steps

1. ✅ Verification complete
2. 🔄 Full preprocessing (in progress)
3. ⏳ Verify preprocessed data structure
4. ⏳ Run fast training baseline (1 epoch test)
5. ⏳ Run full experiments (50 epochs, with/without ASCender)
6. ⏳ Compare ModelNet40 results with synthetic data results
7. ⏳ Update paper with real-world validation

## Commands

### Run preprocessing (currently running):
```bash
python experiments/preprocess_modelnet40.py \
  --data-dir data/modelnet40_ply_hdf5_2048 \
  --output-dir data/ModelNet40/preprocessed \
  --num-points 1024 \
  --k-neighbors 8 \
  --k-normal 10
```

### Verify preprocessed data:
```bash
# Check output structure
ls -la data/ModelNet40/preprocessed/train/ | head

# Load and inspect sample
python -c "
import torch
sample = torch.load('data/ModelNet40/preprocessed/train/sample_0000.pt')
print('Keys:', list(sample.keys()))
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        print(f'{k}: {v.shape}, {v.dtype}')
"
```

### Run fast training experiments:
```bash
# Test with 1 epoch (verify speed)
python experiments/fast_modelnet40_experiment.py \
  --preprocessed-dir data/ModelNet40/preprocessed \
  --epochs 1 \
  --batch-size 32

# Baseline (no ASCender)
python experiments/fast_modelnet40_experiment.py \
  --preprocessed-dir data/ModelNet40/preprocessed \
  --epochs 50 \
  --batch-size 32 \
  --output results/modelnet40_baseline.json

# With ASCender v2.0
python experiments/fast_modelnet40_experiment.py \
  --preprocessed-dir data/ModelNet40/preprocessed \
  --epochs 50 \
  --batch-size 32 \
  --use-ascender \
  --output results/modelnet40_ascender.json
```

## Technical Details

### Preprocessing Functions

**PCA-based normals**:
```python
def compute_normals_pca(xyz, k=10):
    # k-NN for each point
    # PCA on neighbors → smallest eigenvector = surface normal
    # Consistent orientation toward centroid
```

**k-NN graph**:
```python
def compute_knn_indices(xyz, k=8):
    # Pre-compute neighbor indices (fixed graph)
    # Excludes self (returns k neighbors, not k+1)
```

**Density estimation**:
```python
def compute_density(xyz, k=8):
    # Log-based: -log(mean_dist + 1e-6)
    # Stable for gradient flow (no 1/dist explosion)
```

### DataLoader Changes

**Before** (slow):
```python
def __getitem__(self, idx):
    xyz = load_point_cloud(idx)
    normals = compute_normals_pca(xyz, k=10)  # SLOW: 0.2s
    neighbors = compute_knn(xyz, k=8)          # SLOW
    density = compute_density(xyz)             # SLOW
    return xyz, normals, neighbors, density
```

**After** (fast):
```python
def __getitem__(self, idx):
    sample = torch.load(self.samples[idx])  # FAST: <0.01s
    return sample  # All pre-computed!
```

## Impact on Paper

This preprocessing pipeline enables:

1. **Real-world validation** on ModelNet40 (previously infeasible)
2. **Fair comparison** between baseline and ASCender on same dataset
3. **Credibility** for reviewers (not just synthetic data)
4. **Scalability demonstration** for larger datasets

### Paper Section Update

Will update [PAPER_DRAFT.md](PAPER_DRAFT.md) Section 6.3 "ModelNet40 Experiments (Preliminary)" with:

- Complete accuracy results (train/test)
- Learning curves (50 epochs)
- Comparison: Baseline vs ASCender v2.0
- Computational cost analysis
- Discussion of real-world applicability

## Files Created

1. `experiments/preprocess_modelnet40.py` - Preprocessing pipeline
2. `experiments/fast_modelnet40_experiment.py` - Fast training script
3. `experiments/test_preprocessing.py` - Verification tests
4. `PREPROCESSING_STATUS.md` - This status document

---

**Last Updated**: 2025-11-25 22:12 KST
**Status**: ✅ Preprocessing COMPLETE! Now running experiments.

## Current Experiments

### ✅ Speed Verification (1 epoch)
- **Batch speed**: ~9.5 it/s (improved from 7-8 sec/batch bottleneck!)
- **Epoch time**: ~35 seconds training + ~5 seconds eval = **40 seconds/epoch**
- **Test accuracy (1 epoch)**: 20.58% (baseline starting point)
- **Conclusion**: Preprocessing pipeline successful, 10x speedup achieved ✅

### 🔄 Baseline Experiment (50 epochs, no ASCender)
- **Status**: Running in background
- **Current**: Epoch 1/50 completed
- **Estimated time**: ~33 minutes for 50 epochs
- **Output**: `results/modelnet40_baseline.json`

### ⏳ ASCender Experiment (50 epochs, with ASCender)
- **Status**: Pending (will run after baseline completes)
- **Configuration**: A+C (best from synthetic ablation)
- **Output**: `results/modelnet40_ascender.json`
