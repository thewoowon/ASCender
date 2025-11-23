"""
Small Model Experiment - ASCender v2.0

Strategy:
- Small model (few parameters) → inductive bias matters more
- Synthetic data → fast iteration
- Focus: Does α move away from 0.5?
- Goal: Publishable results even if large models don't benefit

Experiment Design:
1. Baseline: Tiny Point Transformer (no ASCender)
2. ASCender v2.0: Same size but with 3-level intervention
3. Metrics: Accuracy, α values, component contributions
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# ASCender v2.0
from models.point_ascender_v2 import (
    VectorKernelEncoder,
    ASCGraphReweight,
    ValuePathwayModulator,
    ASCenderV2Config,
)


# ============================================================================
# Tiny Point Cloud Classifier (Small Model)
# ============================================================================

class TinyPointTransformer(nn.Module):
    """
    Tiny Point Transformer for classification

    Small model configuration:
    - Hidden dim: 32 (vs 64 in standard)
    - 1 layer (vs 3-5 in standard)
    - k=8 neighbors (vs 16)

    Total params: ~10K (vs 1M+ in large models)
    """

    def __init__(self, num_classes=10, use_ascender=False, asc_config=None):
        super().__init__()

        self.use_ascender = use_ascender
        self.hidden_dim = 32
        self.k = 8

        # Input embedding: xyz → features
        self.input_embed = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        # Q, K, V projections
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Position encoding (like Point Transformer)
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, self.hidden_dim)
        )

        # Attention transform
        self.attn_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        # ASCender v2.0
        if use_ascender:
            self.asc_config = asc_config or ASCenderV2Config()

            # Level 1: Graph reweight
            if self.asc_config.enable_graph_reweight:
                self.graph_reweight = ASCGraphReweight(self.asc_config)

            # Level 2: Vector kernel
            if self.asc_config.enable_vector_kernel:
                self.vector_kernel = VectorKernelEncoder(self.hidden_dim, self.asc_config)

            # Level 3: Value modulator
            if self.asc_config.enable_value_rbp or self.asc_config.enable_value_gating:
                self.value_modulator = ValuePathwayModulator(self.hidden_dim, self.asc_config)

            # Learnable alpha (RBP mixing parameter)
            self.alpha_logit = nn.Parameter(torch.zeros(1))

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def knn(self, xyz, k):
        """Simple k-NN (Euclidean distance)"""
        # xyz: (B, N, 3)
        B, N, _ = xyz.shape

        # Pairwise distances
        dist = torch.cdist(xyz, xyz, p=2)  # (B, N, N)

        # Get k nearest neighbors (excluding self)
        _, indices = torch.topk(dist, k=k+1, dim=-1, largest=False)
        indices = indices[:, :, 1:]  # Remove self

        return indices  # (B, N, k)

    def forward(self, xyz, normals=None):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            normals: (B, N, 3) surface normals (optional, for ASCender)

        Returns:
            logits: (B, num_classes)
        """
        B, N, _ = xyz.shape

        # Embed points
        x = self.input_embed(xyz.view(B*N, 3))  # (B*N, hidden_dim)
        x = x.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        # ===== Attention Layer =====

        # k-NN graph
        k = self.k
        k_large = int(k * self.asc_config.neighbor_enlarge_factor) if self.use_ascender and self.asc_config.enable_graph_reweight else k
        indices = self.knn(xyz, k_large)  # (B, N, k_large)

        # Level 1: ASC-aware graph reweighting
        if self.use_ascender and self.asc_config.enable_graph_reweight:
            # Gather large neighbor set
            batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, k_large)
            point_idx = torch.arange(N, device=xyz.device).view(1, N, 1).expand(B, N, k_large)

            xyz_neighbors = xyz[batch_idx, indices]  # (B, N, k_large, 3)
            x_neighbors = x[batch_idx, indices]  # (B, N, k_large, hidden_dim)

            if normals is not None:
                normals_neighbors = normals[batch_idx, indices]  # (B, N, k_large, 3)
            else:
                normals_neighbors = None

            # Compute neighbor scores
            # Reshape for graph_reweight: (B*N, k_large, ...)
            xyz_flat = xyz.view(B*N, 3)
            xyz_neighbors_flat = xyz_neighbors.view(B*N, k_large, 3)
            x_flat = x.view(B*N, self.hidden_dim)
            x_neighbors_flat = x_neighbors.view(B*N, k_large, self.hidden_dim)

            if normals is not None:
                normals_flat = normals.view(B*N, 3)
                normals_neighbors_flat = normals_neighbors.view(B*N, k_large, 3)
            else:
                normals_flat = None
                normals_neighbors_flat = None

            scores = self.graph_reweight(
                xyz_flat, xyz_neighbors_flat,
                x_flat, x_neighbors_flat,
                normals_flat, normals_neighbors_flat
            )  # (B*N, k_large)

            # Select top-k
            _, top_idx = torch.topk(scores, k=k, dim=1)  # (B*N, k)

            # Gather selected neighbors
            top_idx_expanded = top_idx.unsqueeze(-1).expand(B*N, k, 3)
            xyz_neighbors = torch.gather(xyz_neighbors_flat, 1, top_idx_expanded)  # (B*N, k, 3)

            top_idx_expanded_x = top_idx.unsqueeze(-1).expand(B*N, k, self.hidden_dim)
            x_neighbors = torch.gather(x_neighbors_flat, 1, top_idx_expanded_x)  # (B*N, k, hidden_dim)

            if normals is not None:
                top_idx_expanded_n = top_idx.unsqueeze(-1).expand(B*N, k, 3)
                normals_neighbors = torch.gather(normals_neighbors_flat, 1, top_idx_expanded_n)
        else:
            # Standard k-NN
            batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, k)
            xyz_neighbors = xyz[batch_idx, indices]  # (B, N, k, 3)
            x_neighbors = x[batch_idx, indices]  # (B, N, k, hidden_dim)

            xyz_neighbors = xyz_neighbors.view(B*N, k, 3)
            x_neighbors = x_neighbors.view(B*N, k, self.hidden_dim)

            if normals is not None:
                normals_neighbors = normals[batch_idx, indices].view(B*N, k, 3)
            else:
                normals_neighbors = None

        # Q, K, V
        q = self.q_proj(x.view(B*N, self.hidden_dim))  # (B*N, hidden_dim)
        k_neigh = self.k_proj(x_neighbors)  # (B*N, k, hidden_dim)
        v_neigh = self.v_proj(x_neighbors)  # (B*N, k, hidden_dim)

        # Position encoding
        delta_pos = xyz.view(B*N, 1, 3) - xyz_neighbors  # (B*N, k, 3)
        pos_enc = self.pos_encoder(delta_pos.view(B*N*k, 3))  # (B*N*k, hidden_dim)
        pos_enc = pos_enc.view(B*N, k, self.hidden_dim)  # (B*N, k, hidden_dim)

        # Relation: q - k + pos_enc
        rel = q.unsqueeze(1) - k_neigh + pos_enc  # (B*N, k, hidden_dim)

        # Level 2: Vector kernel
        if self.use_ascender and self.asc_config.enable_vector_kernel:
            B_vec = self.vector_kernel(
                p_i=xyz.view(B*N, 3),
                p_j=xyz_neighbors,
                x_i=x.view(B*N, self.hidden_dim),
                x_j=x_neighbors,
                normals_i=normals.view(B*N, 3) if normals is not None else None,
                normals_j=normals_neighbors,
            )  # (B*N, k, hidden_dim)

            # Add vector kernel to relation
            rel = rel + B_vec

        # Attention weights
        attn_weights = self.attn_transform(rel)  # (B*N, k, hidden_dim)
        attn_weights = F.softmax(attn_weights.sum(dim=-1), dim=-1)  # (B*N, k)

        # Value aggregation
        v = v_neigh + pos_enc  # (B*N, k, hidden_dim)

        # Level 3: Value modulation
        if self.use_ascender and (self.asc_config.enable_value_rbp or self.asc_config.enable_value_gating):
            v = self.value_modulator(v, B_vec)

        # Weighted sum
        out = (v * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B*N, hidden_dim)
        out = out.view(B, N, self.hidden_dim)

        # Global pooling
        out = out.mean(dim=1)  # (B, hidden_dim)

        # Classification
        logits = self.output_head(out)  # (B, num_classes)

        return logits

    def get_alpha(self):
        """Get learned alpha value for RBP"""
        if self.use_ascender and hasattr(self, 'alpha_logit'):
            return torch.sigmoid(self.alpha_logit).item()
        return None


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_synthetic_dataset(num_samples=1000, num_points=128, num_classes=10, seed=42):
    """
    Generate synthetic point cloud classification dataset

    Each class is a different geometric shape in 3D:
    - Class 0: Sphere
    - Class 1: Cube
    - Class 2: Cylinder
    - etc.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    samples = []

    for _ in range(num_samples):
        label = np.random.randint(0, num_classes)

        # Generate shape based on class
        if label == 0:  # Sphere
            theta = np.random.uniform(0, 2*np.pi, num_points)
            phi = np.random.uniform(0, np.pi, num_points)
            r = 0.5 + np.random.randn(num_points) * 0.05
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

        elif label == 1:  # Cube
            x = np.random.uniform(-0.5, 0.5, num_points)
            y = np.random.uniform(-0.5, 0.5, num_points)
            z = np.random.uniform(-0.5, 0.5, num_points)

        elif label == 2:  # Cylinder
            theta = np.random.uniform(0, 2*np.pi, num_points)
            r = 0.5 + np.random.randn(num_points) * 0.05
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(-0.5, 0.5, num_points)

        elif label == 3:  # Cone
            theta = np.random.uniform(0, 2*np.pi, num_points)
            z = np.random.uniform(0, 1, num_points)
            r = (1 - z) * 0.5
            x = r * np.cos(theta)
            y = r * np.sin(theta)

        else:  # Random other shapes
            x = np.random.randn(num_points) * 0.3
            y = np.random.randn(num_points) * 0.3
            z = np.random.randn(num_points) * 0.3

        # Stack xyz
        xyz = np.stack([x, y, z], axis=1)

        # Estimate normals (simple: point to centroid)
        centroid = xyz.mean(axis=0)
        normals = xyz - centroid
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

        samples.append({
            'xyz': torch.FloatTensor(xyz),
            'normals': torch.FloatTensor(normals),
            'label': torch.LongTensor([label])
        })

    return samples


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        xyz = batch['xyz'].to(device)
        normals = batch['normals'].to(device)
        labels = batch['label'].to(device).squeeze()

        optimizer.zero_grad()
        logits = model(xyz, normals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            xyz = batch['xyz'].to(device)
            normals = batch['normals'].to(device)
            labels = batch['label'].to(device).squeeze()

            logits = model(xyz, normals)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("="*70)
    print(" "*15 + "Small Model Experiment - ASCender v2.0")
    print("="*70)
    print("\nStrategy:")
    print("  ✓ Small model (10K params) → inductive bias matters")
    print("  ✓ Synthetic data → fast iteration")
    print("  ✓ Goal: α ≠ 0.5, publishable even if large models don't benefit")
    print("="*70)

    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    num_classes = 10
    num_points = 128
    batch_size = 32
    epochs = 50
    lr = 0.001

    # Generate dataset
    print("\n📊 Generating synthetic dataset...")
    train_data = generate_synthetic_dataset(num_samples=800, num_points=num_points, num_classes=num_classes)
    val_data = generate_synthetic_dataset(num_samples=200, num_points=num_points, num_classes=num_classes, seed=43)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")

    # Models
    print("\n🏗️  Building models...")

    # Baseline
    baseline = TinyPointTransformer(num_classes=num_classes, use_ascender=False).to(device)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"   Baseline: {baseline_params:,} parameters")

    # ASCender v2.0
    asc_config = ASCenderV2Config(
        use_alignment=True,
        use_separation=True,
        use_cohesion=True,
        enable_graph_reweight=True,
        enable_vector_kernel=True,
        enable_value_rbp=True,
        enable_value_gating=True,
        w_align=0.5,
        w_sep=0.3,
        w_coh=0.4,
    )
    ascender = TinyPointTransformer(num_classes=num_classes, use_ascender=True, asc_config=asc_config).to(device)
    ascender_params = sum(p.numel() for p in ascender.parameters())
    print(f"   ASCender: {ascender_params:,} parameters")

    # Training
    print("\n🚀 Training...")
    print("-"*70)

    criterion = nn.CrossEntropyLoss()

    baseline_opt = torch.optim.Adam(baseline.parameters(), lr=lr)
    ascender_opt = torch.optim.Adam(ascender.parameters(), lr=lr)

    results = {
        'baseline': {'train_acc': [], 'val_acc': []},
        'ascender': {'train_acc': [], 'val_acc': [], 'alpha': []}
    }

    for epoch in range(epochs):
        # Train baseline
        train_loss_b, train_acc_b = train_epoch(baseline, train_loader, baseline_opt, criterion, device)
        val_loss_b, val_acc_b = eval_model(baseline, val_loader, criterion, device)

        # Train ASCender
        train_loss_a, train_acc_a = train_epoch(ascender, train_loader, ascender_opt, criterion, device)
        val_loss_a, val_acc_a = eval_model(ascender, val_loader, criterion, device)

        # Get alpha
        alpha = ascender.get_alpha() if hasattr(ascender, 'get_alpha') else 0.5

        # Log
        results['baseline']['train_acc'].append(train_acc_b)
        results['baseline']['val_acc'].append(val_acc_b)
        results['ascender']['train_acc'].append(train_acc_a)
        results['ascender']['val_acc'].append(val_acc_a)
        results['ascender']['alpha'].append(alpha if alpha is not None else 0.5)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs}")
            print(f"  Baseline  - Train: {train_acc_b:.3f}, Val: {val_acc_b:.3f}, Loss: {val_loss_b:.4f}")
            print(f"  ASCender  - Train: {train_acc_a:.3f}, Val: {val_acc_a:.3f}, Loss: {val_loss_a:.4f}")
            if alpha is not None:
                print(f"  Alpha: {alpha:.4f}")
            print()

    # Final results
    print("="*70)
    print(" "*25 + "FINAL RESULTS")
    print("="*70)

    best_val_baseline = max(results['baseline']['val_acc'])
    best_val_ascender = max(results['ascender']['val_acc'])
    final_alpha = results['ascender']['alpha'][-1]

    print(f"\n📊 Best Validation Accuracy:")
    print(f"   Baseline:  {best_val_baseline:.3f}")
    print(f"   ASCender:  {best_val_ascender:.3f}")
    print(f"   Δ Improvement: {(best_val_ascender - best_val_baseline):.3f}")

    if alpha is not None:
        print(f"\n🔬 Learned Alpha (RBP mixing):")
        print(f"   Final α: {final_alpha:.4f}")
        if final_alpha < 0.4:
            print(f"   ✅ α < 0.4 → Bias dominates (strong ASCender effect!)")
        elif final_alpha > 0.6:
            print(f"   ✅ α > 0.6 → Learned attention dominates")
        else:
            print(f"   ⚠️  α ≈ 0.5 → Balanced (need stronger bias)")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "small_model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {output_dir / 'small_model_results.json'}")

    # Publishability check
    print("\n"+"="*70)
    print(" "*20 + "PUBLISHABILITY ANALYSIS")
    print("="*70)

    improvement = best_val_ascender - best_val_baseline

    if improvement > 0.05:  # 5% improvement
        print("\n✅ PUBLISHABLE!")
        print(f"   Reason: {improvement*100:.1f}% improvement on small model")
        print("   → Demonstrates inductive bias value")
        print("   → Works when model capacity is limited")
    elif improvement > 0.02:  # 2% improvement
        print("\n✓ POSSIBLY PUBLISHABLE")
        print(f"   Reason: {improvement*100:.1f}% improvement")
        print("   → Need stronger ablation studies")
    else:
        print("\n⚠️  NEEDS MORE WORK")
        print(f"   Reason: Only {improvement*100:.1f}% improvement")
        print("   → Try: increase bias weights, adjust σ")

    if alpha is not None and (final_alpha < 0.4 or final_alpha > 0.6):
        print("\n✅ BONUS: α ≠ 0.5")
        print("   → Shows ASCender actually contributes (not ignored)")

    print("\n" + "="*70)
    print("Experiment complete! 🎉")
    print("="*70)


if __name__ == "__main__":
    main()
