"""
Tiny Point Transformer with PROPER RBP α Implementation

Key: α * attn_learned + (1-α) * attn_bias
"""

import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from models.point_ascender_v2 import (
    VectorKernelEncoder,
    ASCGraphReweight,
    ValuePathwayModulator,
    ASCenderV2Config,
)


class TinyPointTransformerRBP(nn.Module):
    """
    Tiny Point Transformer with PROPER RBP

    Key difference:
    - Learned attention path: Q·K → attn_learned
    - Bias attention path: B_vec → attn_bias
    - RBP mixing: α * attn_learned + (1-α) * attn_bias
    """

    def __init__(self, num_classes=10, use_ascender=False, asc_config=None):
        super().__init__()

        self.use_ascender = use_ascender
        self.hidden_dim = 32
        self.k = 8

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        # Q, K, V
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, self.hidden_dim)
        )

        # ASCender components
        if use_ascender:
            self.asc_config = asc_config or ASCenderV2Config()

            # Vector kernel encoder
            if self.asc_config.enable_vector_kernel:
                self.vector_kernel = VectorKernelEncoder(self.hidden_dim, self.asc_config)

            # Graph reweight
            if self.asc_config.enable_graph_reweight:
                self.graph_reweight = ASCGraphReweight(self.asc_config)

            # Value modulator
            if self.asc_config.enable_value_rbp or self.asc_config.enable_value_gating:
                self.value_modulator = ValuePathwayModulator(self.hidden_dim, self.asc_config)

            # ===== THE KEY: Learnable α for RBP =====
            # α ∈ [0, 1] via sigmoid
            self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0) = 0.5

        # Output
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def knn(self, xyz, k):
        """k-NN"""
        B, N, _ = xyz.shape
        dist = torch.cdist(xyz, xyz, p=2)
        _, indices = torch.topk(dist, k=k+1, dim=-1, largest=False)
        return indices[:, :, 1:]  # Exclude self

    def forward(self, xyz, normals=None):
        B, N, _ = xyz.shape

        # Embed
        x = self.input_embed(xyz.view(B*N, 3))
        x = x.view(B, N, self.hidden_dim)

        # k-NN
        k = self.k
        k_large = int(k * self.asc_config.neighbor_enlarge_factor) if self.use_ascender and self.asc_config.enable_graph_reweight else k
        indices = self.knn(xyz, k_large)

        # ===== Level 1: Graph Reweighting =====
        if self.use_ascender and self.asc_config.enable_graph_reweight:
            batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, k_large)

            xyz_neighbors = xyz[batch_idx, indices]
            x_neighbors = x[batch_idx, indices]
            normals_neighbors = normals[batch_idx, indices] if normals is not None else None

            # Flatten for graph_reweight
            xyz_flat = xyz.view(B*N, 3)
            xyz_neighbors_flat = xyz_neighbors.view(B*N, k_large, 3)
            x_flat = x.view(B*N, self.hidden_dim)
            x_neighbors_flat = x_neighbors.view(B*N, k_large, self.hidden_dim)
            normals_flat = normals.view(B*N, 3) if normals is not None else None
            normals_neighbors_flat = normals_neighbors.view(B*N, k_large, 3) if normals is not None else None

            # Compute scores
            scores = self.graph_reweight(
                xyz_flat, xyz_neighbors_flat,
                x_flat, x_neighbors_flat,
                normals_flat, normals_neighbors_flat
            )

            # Select top-k
            _, top_idx = torch.topk(scores, k=k, dim=1)

            # Gather
            top_idx_xyz = top_idx.unsqueeze(-1).expand(B*N, k, 3)
            xyz_neighbors = torch.gather(xyz_neighbors_flat, 1, top_idx_xyz)

            top_idx_x = top_idx.unsqueeze(-1).expand(B*N, k, self.hidden_dim)
            x_neighbors = torch.gather(x_neighbors_flat, 1, top_idx_x)

            if normals is not None:
                top_idx_n = top_idx.unsqueeze(-1).expand(B*N, k, 3)
                normals_neighbors = torch.gather(normals_neighbors_flat, 1, top_idx_n)
        else:
            # Standard k-NN
            batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, k)
            xyz_neighbors = xyz[batch_idx, indices].view(B*N, k, 3)
            x_neighbors = x[batch_idx, indices].view(B*N, k, self.hidden_dim)
            normals_neighbors = normals[batch_idx, indices].view(B*N, k, 3) if normals is not None else None

        # Q, K, V
        q = self.q_proj(x.view(B*N, self.hidden_dim))  # (B*N, C)
        k_neigh = self.k_proj(x_neighbors)  # (B*N, k, C)
        v_neigh = self.v_proj(x_neighbors)  # (B*N, k, C)

        # Position encoding
        delta_pos = xyz.view(B*N, 1, 3) - xyz_neighbors
        pos_enc = self.pos_encoder(delta_pos.view(B*N*k, 3)).view(B*N, k, self.hidden_dim)

        # ===== LEARNED ATTENTION PATH =====
        # Standard dot-product attention
        attn_scores_learned = (q.unsqueeze(1) * k_neigh).sum(dim=-1) / np.sqrt(self.hidden_dim)  # (B*N, k)
        attn_weights_learned = F.softmax(attn_scores_learned, dim=-1)  # (B*N, k)

        # ===== BIAS ATTENTION PATH (ASCender) =====
        if self.use_ascender and self.asc_config.enable_vector_kernel:
            # Level 2: Vector kernel
            B_vec = self.vector_kernel(
                p_i=xyz.view(B*N, 3),
                p_j=xyz_neighbors,
                x_i=x.view(B*N, self.hidden_dim),
                x_j=x_neighbors,
                normals_i=normals.view(B*N, 3) if normals is not None else None,
                normals_j=normals_neighbors,
            )  # (B*N, k, C)

            # Aggregate B_vec to get bias attention scores
            # Simple: sum over channels
            attn_scores_bias = B_vec.sum(dim=-1)  # (B*N, k)
            attn_weights_bias = F.softmax(attn_scores_bias, dim=-1)  # (B*N, k)

            # ===== RBP MIXING with α =====
            alpha = torch.sigmoid(self.alpha_logit)  # [0, 1]
            attn_weights = alpha * attn_weights_learned + (1 - alpha) * attn_weights_bias
        else:
            attn_weights = attn_weights_learned
            B_vec = None

        # ===== Value Aggregation =====
        v = v_neigh + pos_enc

        # Level 3: Value modulation
        if self.use_ascender and B_vec is not None and (self.asc_config.enable_value_rbp or self.asc_config.enable_value_gating):
            v = self.value_modulator(v, B_vec)

        # Weighted sum
        out = (v * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B*N, C)
        out = out.view(B, N, self.hidden_dim)

        # Global pooling
        out = out.mean(dim=1)

        # Classification
        logits = self.output_head(out)
        return logits

    def get_alpha(self):
        """Get learned α"""
        if self.use_ascender and hasattr(self, 'alpha_logit'):
            return torch.sigmoid(self.alpha_logit).item()
        return None


# ============================================================================
# Dataset (reuse from original)
# ============================================================================

def generate_synthetic_dataset(num_samples=1000, num_points=128, num_classes=10, seed=42):
    """Generate synthetic point cloud dataset"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    samples = []

    for _ in range(num_samples):
        label = np.random.randint(0, num_classes)

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

        else:  # Random
            x = np.random.randn(num_points) * 0.3
            y = np.random.randn(num_points) * 0.3
            z = np.random.randn(num_points) * 0.3

        xyz = np.stack([x, y, z], axis=1)

        # Normals
        centroid = xyz.mean(axis=0)
        normals = xyz - centroid
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

        samples.append({
            'xyz': torch.FloatTensor(xyz),
            'normals': torch.FloatTensor(normals),
            'label': torch.LongTensor([label])
        })

    return samples


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
    print(" "*10 + "Tiny Model with PROPER RBP α Implementation")
    print("="*70)
    print("\nKey: α * attn_learned + (1-α) * attn_bias")
    print("     α is LEARNABLE → should move away from 0.5!")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Config
    num_classes = 10
    num_points = 128
    batch_size = 32
    epochs = 100  # More epochs to see α evolution
    lr = 0.001

    # Dataset
    print("\n📊 Generating dataset...")
    train_data = generate_synthetic_dataset(800, num_points, num_classes)
    val_data = generate_synthetic_dataset(200, num_points, num_classes, seed=43)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    # Models
    print("\n🏗️  Building models...")

    # Baseline
    baseline = TinyPointTransformerRBP(num_classes=num_classes, use_ascender=False).to(device)
    print(f"   Baseline: {sum(p.numel() for p in baseline.parameters()):,} params")

    # ASCender with different bias strengths
    configs = [
        ("Weak Bias", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False,  # Start simple
            enable_vector_kernel=True,
            enable_value_rbp=False,  # Start simple
            enable_value_gating=False,
            w_align=0.5, w_sep=0.3, w_coh=0.4,
        )),
        ("Strong Bias", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False,
            enable_vector_kernel=True,
            enable_value_rbp=False,
            enable_value_gating=False,
            w_align=2.0, w_sep=1.5, w_coh=2.0,
        )),
        ("Very Strong Bias", ASCenderV2Config(
            use_alignment=True, use_separation=True, use_cohesion=True,
            enable_graph_reweight=False,
            enable_vector_kernel=True,
            enable_value_rbp=False,
            enable_value_gating=False,
            w_align=5.0, w_sep=3.0, w_coh=4.0,
        )),
    ]

    # Train baseline
    print("\n🚀 Training Baseline...")
    baseline_opt = torch.optim.Adam(baseline.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_epoch(baseline, train_loader, baseline_opt, criterion, device)
        if (epoch + 1) % 20 == 0:
            _, val_acc = eval_model(baseline, val_loader, criterion, device)
            print(f"   Epoch {epoch+1}: Val={val_acc:.3f}")

    _, baseline_acc = eval_model(baseline, val_loader, criterion, device)
    print(f"   Final: {baseline_acc:.3f}")

    # Train ASCender variants
    results = {}

    for name, config in configs:
        print(f"\n🚀 Training {name}...")
        print(f"   w_a={config.w_align}, w_s={config.w_sep}, w_c={config.w_coh}")

        model = TinyPointTransformerRBP(num_classes=num_classes, use_ascender=True, asc_config=config).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        alpha_history = []
        acc_history = []

        for epoch in range(epochs):
            train_epoch(model, train_loader, opt, criterion, device)

            if (epoch + 1) % 20 == 0:
                _, val_acc = eval_model(model, val_loader, criterion, device)
                alpha = model.get_alpha()
                alpha_history.append(alpha)
                acc_history.append(val_acc)
                print(f"   Epoch {epoch+1}: Val={val_acc:.3f}, α={alpha:.4f}")

        _, final_acc = eval_model(model, val_loader, criterion, device)
        final_alpha = model.get_alpha()

        results[name] = {
            'accuracy': final_acc,
            'alpha': final_alpha,
            'alpha_history': alpha_history,
            'acc_history': acc_history,
            'improvement': final_acc - baseline_acc,
        }

    # Summary
    print("\n" + "="*70)
    print(" "*25 + "RESULTS")
    print("="*70)

    print(f"\n📊 Baseline: {baseline_acc:.3f}")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"   Accuracy: {res['accuracy']:.3f} (Δ {res['improvement']:+.3f})")
        print(f"   Final α:  {res['alpha']:.4f}")

        if res['alpha'] < 0.4:
            print(f"   ✅ SUCCESS! α < 0.4 → Bias dominates!")
        elif res['alpha'] > 0.6:
            print(f"   ✅ α > 0.6 → Learned dominates")
        else:
            print(f"   ⚠️  α ≈ 0.5 → Balanced")

        # Show α evolution
        if len(res['alpha_history']) > 0:
            print(f"   α evolution: {[f'{a:.3f}' for a in res['alpha_history']]}")

    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best = results[best_name]

    print(f"\n🏆 Best: {best_name}")
    print(f"   Accuracy: {best['accuracy']:.3f} (Δ {best['improvement']:+.3f})")
    print(f"   Alpha: {best['alpha']:.4f}")

    # Publishability
    print("\n" + "="*70)
    print(" "*20 + "PUBLISHABILITY CHECK")
    print("="*70)

    publishable = False

    if best['improvement'] > 0.05:
        print(f"\n✅ STRONG RESULT: {best['improvement']*100:.1f}% improvement!")
        publishable = True

    if best['alpha'] < 0.4 or best['alpha'] > 0.6:
        print(f"\n✅ α ≠ 0.5: α = {best['alpha']:.4f}")
        print("   → ASCender bias actually contributes!")
        publishable = True

    if publishable:
        print("\n🎉 PUBLISHABLE!")
    else:
        print("\n⚠️  Needs more work")

    # Save
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "rbp_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Saved to {output_dir / 'rbp_results.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
