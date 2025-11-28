"""
ModelNet40 Preprocessing Script

Pre-computes expensive operations for fast training:
1. PCA-based surface normals (expensive k-NN + PCA per point)
2. Boids features (density, etc.)
3. k-NN neighbor indices (fixed graph)

This reduces 7-8 sec/batch to <1 sec/batch by moving computation from
training loop to one-time preprocessing.

Usage:
    python experiments/preprocess_modelnet40.py

Output:
    data/ModelNet40/preprocessed/
        train/sample_0000.pt
        train/sample_0001.pt
        ...
        test/sample_0000.pt
        ...
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import argparse


def compute_normals_pca(xyz, k=10):
    """
    Compute surface normals using PCA on k-NN

    Args:
        xyz: (N, 3) point cloud
        k: number of neighbors for PCA

    Returns:
        normals: (N, 3) surface normals
    """
    N = xyz.shape[0]
    normals = np.zeros_like(xyz)

    # Build k-NN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(xyz)

    for i in range(N):
        distances, indices = nbrs.kneighbors(xyz[i:i+1])
        neighbors = xyz[indices[0]]

        # PCA on neighbors
        pca = PCA(n_components=3)
        pca.fit(neighbors - neighbors.mean(axis=0))
        normal = pca.components_[-1]  # smallest eigenvalue = surface normal

        # Consistent orientation (toward centroid)
        centroid = xyz.mean(axis=0)
        if np.dot(normal, xyz[i] - centroid) < 0:
            normal = -normal

        normals[i] = normal

    return normals


def compute_knn_indices(xyz, k=8):
    """
    Pre-compute k-NN neighbor indices

    Args:
        xyz: (N, 3) point cloud
        k: number of neighbors

    Returns:
        indices: (N, k) neighbor indices
        distances: (N, k) distances
    """
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)

    # Exclude self (first neighbor)
    return indices[:, 1:], distances[:, 1:]


def compute_density(xyz, k=8):
    """
    Compute local density (log-based for stability)

    Args:
        xyz: (N, 3) point cloud
        k: number of neighbors

    Returns:
        density: (N,) local density values
    """
    _, distances = compute_knn_indices(xyz, k)
    mean_dist = distances.mean(axis=1)
    density = -np.log(mean_dist + 1e-6)  # Log-based (stable)
    return density


def preprocess_sample(xyz, label, k_neighbors=8, k_normal=10):
    """
    Pre-compute all expensive operations for one sample

    Args:
        xyz: (N, 3) raw point cloud
        label: int class label
        k_neighbors: k for neighbor graph
        k_normal: k for normal estimation

    Returns:
        dict with all pre-computed tensors
    """
    # 1. Normalize point cloud
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / (np.max(np.linalg.norm(xyz, axis=1)) + 1e-8)

    # 2. Compute surface normals (EXPENSIVE - do once!)
    normals = compute_normals_pca(xyz, k=k_normal)

    # 3. Compute k-NN graph (EXPENSIVE - do once!)
    neighbor_indices, neighbor_distances = compute_knn_indices(xyz, k=k_neighbors)

    # 4. Compute density (for Boids Separation)
    density = compute_density(xyz, k=k_neighbors)

    return {
        'xyz': torch.FloatTensor(xyz),           # (N, 3)
        'normals': torch.FloatTensor(normals),   # (N, 3)
        'neighbor_indices': torch.LongTensor(neighbor_indices),  # (N, k)
        'neighbor_distances': torch.FloatTensor(neighbor_distances),  # (N, k)
        'density': torch.FloatTensor(density),   # (N,)
        'label': torch.LongTensor([label])       # (1,)
    }


def load_modelnet40_h5(data_dir):
    """
    Load ModelNet40 HDF5 files

    Returns:
        train_data, train_labels, test_data, test_labels
    """
    data_dir = Path(data_dir)

    # Load training data
    train_files = sorted(data_dir.glob('ply_data_train*.h5'))
    train_data_list = []
    train_label_list = []

    for f in train_files:
        with h5py.File(f, 'r') as hf:
            train_data_list.append(hf['data'][:])
            train_label_list.append(hf['label'][:])

    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_label_list, axis=0)

    # Load test data
    test_files = sorted(data_dir.glob('ply_data_test*.h5'))
    test_data_list = []
    test_label_list = []

    for f in test_files:
        with h5py.File(f, 'r') as hf:
            test_data_list.append(hf['data'][:])
            test_label_list.append(hf['label'][:])

    test_data = np.concatenate(test_data_list, axis=0)
    test_labels = np.concatenate(test_label_list, axis=0)

    return train_data, train_labels.squeeze(), test_data, test_labels.squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                       default='data/modelnet40_ply_hdf5_2048',
                       help='Path to ModelNet40 HDF5 directory')
    parser.add_argument('--output-dir', type=str,
                       default='data/ModelNet40/preprocessed',
                       help='Output directory for preprocessed files')
    parser.add_argument('--num-points', type=int, default=1024,
                       help='Number of points to sample')
    parser.add_argument('--k-neighbors', type=int, default=8,
                       help='Number of neighbors for graph')
    parser.add_argument('--k-normal', type=int, default=10,
                       help='Number of neighbors for normal estimation')
    args = parser.parse_args()

    print("="*70)
    print("ModelNet40 Preprocessing Pipeline")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Num points: {args.num_points}")
    print(f"k neighbors: {args.k_neighbors}")
    print(f"k normals: {args.k_normal}")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test').mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading ModelNet40 HDF5 files...")
    train_data, train_labels, test_data, test_labels = load_modelnet40_h5(args.data_dir)

    print(f"Train: {train_data.shape[0]} samples")
    print(f"Test: {test_data.shape[0]} samples")
    print()

    # Process training data
    print("Preprocessing training data...")
    for i in tqdm(range(len(train_data)), desc="Train"):
        xyz = train_data[i]
        label = train_labels[i]

        # Random downsample to num_points
        if xyz.shape[0] > args.num_points:
            choice = np.random.choice(xyz.shape[0], args.num_points, replace=False)
            xyz = xyz[choice]

        # Preprocess
        sample = preprocess_sample(
            xyz, label,
            k_neighbors=args.k_neighbors,
            k_normal=args.k_normal
        )

        # Save
        output_path = output_dir / 'train' / f'sample_{i:04d}.pt'
        torch.save(sample, output_path)

    # Process test data
    print("Preprocessing test data...")
    for i in tqdm(range(len(test_data)), desc="Test"):
        xyz = test_data[i]
        label = test_labels[i]

        # Random downsample to num_points
        if xyz.shape[0] > args.num_points:
            choice = np.random.choice(xyz.shape[0], args.num_points, replace=False)
            xyz = xyz[choice]

        # Preprocess
        sample = preprocess_sample(
            xyz, label,
            k_neighbors=args.k_neighbors,
            k_normal=args.k_normal
        )

        # Save
        output_path = output_dir / 'test' / f'sample_{i:04d}.pt'
        torch.save(sample, output_path)

    print()
    print("="*70)
    print("Preprocessing complete!")
    print(f"Saved to: {output_dir}")
    print()

    # Calculate storage size
    import os
    total_size = sum(
        os.path.getsize(p)
        for p in output_dir.rglob('*.pt')
    ) / 1024**2  # MB

    print(f"Total storage: {total_size:.1f} MB")
    print(f"Average per sample: {total_size / (len(train_data) + len(test_data)):.2f} MB")
    print()

    # Show sample
    sample_path = output_dir / 'train' / 'sample_0000.pt'
    sample = torch.load(sample_path)
    print("Sample structure:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape} ({val.dtype})")
    print()
    print("✅ Ready for fast training!")


if __name__ == '__main__':
    main()
