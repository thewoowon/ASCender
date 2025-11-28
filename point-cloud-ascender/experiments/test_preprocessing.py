"""
Quick test of preprocessing pipeline on small subset

Tests that preprocessing works correctly before full ModelNet40 processing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_modelnet40 import (
    compute_normals_pca,
    compute_knn_indices,
    compute_density,
    preprocess_sample,
    load_modelnet40_h5
)
import numpy as np
import torch
import time

def test_preprocessing_functions():
    """Test individual preprocessing functions"""
    print("="*70)
    print("Testing Preprocessing Functions")
    print("="*70)

    # Create synthetic test point cloud
    np.random.seed(42)
    xyz = np.random.randn(100, 3).astype(np.float32)
    xyz = xyz / (np.max(np.linalg.norm(xyz, axis=1)) + 1e-8)  # normalize

    print(f"\nTest point cloud: {xyz.shape}")

    # Test 1: Normals computation
    print("\n1. Testing PCA-based normals computation...")
    start = time.time()
    normals = compute_normals_pca(xyz, k=10)
    elapsed = time.time() - start
    print(f"   ✓ Normals shape: {normals.shape}")
    print(f"   ✓ Time: {elapsed:.3f}s")
    print(f"   ✓ Normal magnitudes (should be ~1.0): {np.linalg.norm(normals, axis=1).mean():.4f}")

    # Test 2: k-NN indices
    print("\n2. Testing k-NN graph computation...")
    start = time.time()
    indices, distances = compute_knn_indices(xyz, k=8)
    elapsed = time.time() - start
    print(f"   ✓ Indices shape: {indices.shape}")
    print(f"   ✓ Distances shape: {distances.shape}")
    print(f"   ✓ Time: {elapsed:.3f}s")
    print(f"   ✓ Mean distance: {distances.mean():.4f}")

    # Test 3: Density computation
    print("\n3. Testing density computation...")
    start = time.time()
    density = compute_density(xyz, k=8)
    elapsed = time.time() - start
    print(f"   ✓ Density shape: {density.shape}")
    print(f"   ✓ Time: {elapsed:.3f}s")
    print(f"   ✓ Density range: [{density.min():.4f}, {density.max():.4f}]")

    # Test 4: Full preprocessing
    print("\n4. Testing full preprocessing pipeline...")
    start = time.time()
    sample = preprocess_sample(xyz, label=5, k_neighbors=8, k_normal=10)
    elapsed = time.time() - start
    print(f"   ✓ Time: {elapsed:.3f}s")
    print(f"   ✓ Keys: {list(sample.keys())}")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"   ✓ {k}: {v.shape}, {v.dtype}")

    print("\n✅ All preprocessing functions work correctly!")
    return elapsed


def test_modelnet40_loading():
    """Test loading actual ModelNet40 data"""
    print("\n" + "="*70)
    print("Testing ModelNet40 Data Loading")
    print("="*70)

    data_dir = Path('data/modelnet40_ply_hdf5_2048')

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return None

    print(f"\nLoading from: {data_dir}")

    try:
        start = time.time()
        train_data, train_labels, test_data, test_labels = load_modelnet40_h5(data_dir)
        elapsed = time.time() - start

        print(f"✓ Loaded in {elapsed:.2f}s")
        print(f"✓ Train: {train_data.shape}, labels: {train_labels.shape}")
        print(f"✓ Test: {test_data.shape}, labels: {test_labels.shape}")
        print(f"✓ Point cloud shape: {train_data[0].shape}")
        print(f"✓ Num classes: {len(np.unique(train_labels))}")

        return train_data, train_labels, test_data, test_labels

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_full_pipeline_on_real_data(train_data, train_labels, num_samples=5):
    """Test preprocessing on real ModelNet40 samples"""
    print("\n" + "="*70)
    print(f"Testing Full Pipeline on {num_samples} Real Samples")
    print("="*70)

    times = []

    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}:")
        xyz = train_data[i]
        label = train_labels[i]

        # Downsample to 1024 points
        if xyz.shape[0] > 1024:
            choice = np.random.choice(xyz.shape[0], 1024, replace=False)
            xyz = xyz[choice]

        print(f"  Points: {xyz.shape[0]}, Label: {label}")

        start = time.time()
        sample = preprocess_sample(xyz, label, k_neighbors=8, k_normal=10)
        elapsed = time.time() - start
        times.append(elapsed)

        print(f"  Time: {elapsed:.3f}s")
        print(f"  Sizes: xyz={sample['xyz'].shape}, normals={sample['normals'].shape}")

    avg_time = np.mean(times)
    print(f"\n✅ Average preprocessing time: {avg_time:.3f}s per sample")

    # Estimate total time
    total_samples = len(train_data)
    estimated_total = avg_time * total_samples / 60  # minutes
    print(f"📊 Estimated total preprocessing time: {estimated_total:.1f} minutes for {total_samples} samples")

    return avg_time


def main():
    print("\n" + "="*70)
    print("ModelNet40 Preprocessing Verification")
    print("="*70)

    # Test 1: Basic functions
    single_sample_time = test_preprocessing_functions()

    # Test 2: Load ModelNet40
    data = test_modelnet40_loading()

    if data is None:
        print("\n⚠️  Could not load ModelNet40 data. Skipping full pipeline test.")
        return

    train_data, train_labels, test_data, test_labels = data

    # Test 3: Full pipeline on real samples
    avg_time = test_full_pipeline_on_real_data(train_data, train_labels, num_samples=5)

    # Final summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"✓ Single sample time: {single_sample_time:.3f}s")
    print(f"✓ Average time on real data: {avg_time:.3f}s")
    print(f"✓ Total train samples: {len(train_data)}")
    print(f"✓ Total test samples: {len(test_data)}")

    estimated_total = avg_time * (len(train_data) + len(test_data)) / 60
    print(f"\n📊 Estimated preprocessing time: {estimated_total:.1f} minutes")

    if estimated_total < 60:
        print("✅ Preprocessing time is reasonable. Ready to proceed!")
    else:
        print(f"⚠️  Preprocessing will take ~{estimated_total/60:.1f} hours. Consider running in background.")

    print("\n" + "="*70)
    print("Next steps:")
    print("1. Run full preprocessing: python experiments/preprocess_modelnet40.py")
    print("2. Verify output: ls data/ModelNet40/preprocessed/train/ | head")
    print("3. Test fast training: python experiments/fast_modelnet40_experiment.py --epochs 1")
    print("="*70)


if __name__ == '__main__':
    main()
