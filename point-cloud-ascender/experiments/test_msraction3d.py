"""
MSRAction3D Real Data Experiment

Purpose: Test ASCender on REAL 3D point cloud sequences
Key Question: Does α learn away from 0.5 on real spatial data?

Hypothesis:
- Synthetic data (clusters): α ≈ 0.5 (no clear spatial structure)
- Real data (human actions): α < 0.4 (spatial bias valuable!)

Dataset: MSRAction3D
- 567 sequences, 20 actions
- Real depth sequences from Kinect
- Train: subjects 1-5, Test: subjects 6-10
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from src.models.point_ascender_bias import PointAscenderBias, PointAscenderConfig
from src.data.msraction3d_loader import MSRAction3DPointCloudDataset


class TemporalPointClassifier(nn.Module):
    """
    Temporal point cloud classifier for action recognition

    Architecture:
    - Per-frame self-attention with ASCender spatial bias
    - Temporal aggregation (mean pooling)
    - Classification head
    """
    def __init__(
        self,
        n_classes=20,
        d_model=128,
        n_heads=8,
        use_ascender=False,
        strong_bias=False
    ):
        super().__init__()
        self.use_ascender = use_ascender
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Per-frame embedding
        self.embedding = nn.Linear(3, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention (applied per frame)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # ASCender spatial bias
        if use_ascender:
            if strong_bias:
                cfg = PointAscenderConfig(
                    use_alignment=True,
                    use_separation=True,
                    use_cohesion=True,
                    per_head_scale=True,
                    per_head_gate=True,
                    w_align=0.80,  # Strong bias (2x)
                    w_sep=0.40,
                    w_coh=0.60,
                    sigma_sep=0.05,
                    sigma_coh=0.50,
                )
            else:
                cfg = PointAscenderConfig(
                    use_alignment=True,
                    use_separation=True,
                    use_cohesion=True,
                    per_head_scale=True,
                    per_head_gate=True,
                    w_align=0.40,
                    w_sep=0.20,
                    w_coh=0.30,
                    sigma_sep=0.05,
                    sigma_coh=0.50,
                )
            self.biaser = PointAscenderBias(cfg, n_heads=n_heads)

            # Learnable α for RBP (per head)
            self.alpha_logit = nn.Parameter(torch.zeros(n_heads))

        # Feedforward
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

        # Temporal aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, xyz, normals):
        """
        Args:
            xyz: (B, T, N, 3) temporal point cloud sequence
            normals: (B, T, N, 3) surface normals

        Returns:
            logits: (B, n_classes)
        """
        B, T, N, _ = xyz.shape

        # Process each frame independently
        frame_features = []

        for t in range(T):
            xyz_t = xyz[:, t]  # (B, N, 3)
            normals_t = normals[:, t]  # (B, N, 3)

            # Embed
            x = self.embedding(xyz_t)  # (B, N, d_model)
            x = self.norm1(x)

            # Self-attention
            q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
            v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

            # Learned attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

            # Apply ASCender bias with RBP
            if self.use_ascender:
                spatial_bias = self.biaser(q, k, xyz_t, xyz_t, normals_t, normals_t)
                alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1, 1)
                attn_scores = alpha * attn_scores + (1 - alpha) * spatial_bias

            # Attention
            attn_probs = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_probs, v)
            out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
            out = self.o_proj(out)

            # Residual
            x = x + out

            # Feedforward
            x = x + self.ffn(self.norm2(x))

            # Frame-level pooling
            frame_feat = x.mean(dim=1)  # (B, d_model)
            frame_features.append(frame_feat)

        # Temporal aggregation
        temporal_feats = torch.stack(frame_features, dim=2)  # (B, d_model, T)
        pooled = self.temporal_pool(temporal_feats).squeeze(2)  # (B, d_model)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def get_alpha(self):
        """Get current α values"""
        if self.use_ascender:
            return torch.sigmoid(self.alpha_logit).detach().cpu()
        return None


def train_and_evaluate(
    data_root: str,
    n_epochs: int = 30,
    batch_size: int = 4,
    n_points: int = 256,
    n_frames: int = 8,
    strong_bias: bool = False,
    device: str = 'cpu'
):
    """
    Train and evaluate on MSRAction3D
    """
    print("="*80)
    print("🎬 MSRAction3D Real Data Experiment")
    print("="*80)
    print()
    print(f"📁 Data root: {data_root}")
    print(f"📱 Device: {device}")
    print(f"🔢 Epochs: {n_epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔹 Points per frame: {n_points}")
    print(f"🎞️  Frames per sequence: {n_frames}")
    print(f"💪 Strong bias: {'Yes (2x)' if strong_bias else 'No (1x)'}")
    print()

    # Datasets
    print("📂 Loading datasets...")
    train_dataset = MSRAction3DPointCloudDataset(
        data_root=data_root,
        n_points=n_points,
        n_frames=n_frames,
        split='train',
        train_subjects=[1, 2, 3, 4, 5],
        test_subjects=[6, 7, 8, 9, 10]
    )

    test_dataset = MSRAction3DPointCloudDataset(
        data_root=data_root,
        n_points=n_points,
        n_frames=n_frames,
        split='test',
        train_subjects=[1, 2, 3, 4, 5],
        test_subjects=[6, 7, 8, 9, 10]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CPU only for now
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    print(f"   Train: {len(train_dataset)} sequences")
    print(f"   Test:  {len(test_dataset)} sequences")
    print()

    # Models
    print("🏗️  Building models...")
    baseline = TemporalPointClassifier(
        n_classes=20,
        d_model=128,
        n_heads=8,
        use_ascender=False
    ).to(device)

    ascender = TemporalPointClassifier(
        n_classes=20,
        d_model=128,
        n_heads=8,
        use_ascender=True,
        strong_bias=strong_bias
    ).to(device)

    n_params_baseline = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    n_params_ascender = sum(p.numel() for p in ascender.parameters() if p.requires_grad)

    print(f"   Baseline: {n_params_baseline:,} parameters")
    print(f"   ASCender: {n_params_ascender:,} parameters (+{n_params_ascender - n_params_baseline:,})")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    opt_baseline = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_ascender = torch.optim.AdamW(ascender.parameters(), lr=1e-3, weight_decay=1e-4)

    sched_baseline = torch.optim.lr_scheduler.CosineAnnealingLR(opt_baseline, T_max=n_epochs)
    sched_ascender = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ascender, T_max=n_epochs)

    print("🏋️  Training...\n")

    # Training history
    history = {
        'baseline_loss': [],
        'ascender_loss': [],
        'baseline_acc': [],
        'ascender_acc': [],
        'alpha_mean': [],
        'alpha_std': [],
        'alpha_min': [],
        'alpha_max': [],
    }

    best_acc_baseline = 0.0
    best_acc_ascender = 0.0

    for epoch in range(n_epochs):
        # Training
        baseline.train()
        ascender.train()

        train_loss_baseline = 0.0
        train_loss_ascender = 0.0
        train_acc_baseline = 0.0
        train_acc_ascender = 0.0
        n_batches = 0

        for batch in train_loader:
            xyz = batch['xyz'].to(device)  # (B, T, N, 3)
            normals = batch['normals'].to(device)  # (B, T, N, 3)
            labels = batch['label'].to(device)  # (B,)

            # Baseline
            opt_baseline.zero_grad()
            logits_baseline = baseline(xyz, normals)
            loss_baseline = criterion(logits_baseline, labels)
            loss_baseline.backward()
            opt_baseline.step()

            train_loss_baseline += loss_baseline.item()
            train_acc_baseline += (logits_baseline.argmax(dim=-1) == labels).float().mean().item()

            # ASCender
            opt_ascender.zero_grad()
            logits_ascender = ascender(xyz, normals)
            loss_ascender = criterion(logits_ascender, labels)
            loss_ascender.backward()
            opt_ascender.step()

            train_loss_ascender += loss_ascender.item()
            train_acc_ascender += (logits_ascender.argmax(dim=-1) == labels).float().mean().item()

            n_batches += 1

        train_loss_baseline /= n_batches
        train_loss_ascender /= n_batches
        train_acc_baseline /= n_batches
        train_acc_ascender /= n_batches

        sched_baseline.step()
        sched_ascender.step()

        # Get α
        alpha = ascender.get_alpha()

        # Record
        history['baseline_loss'].append(train_loss_baseline)
        history['ascender_loss'].append(train_loss_ascender)
        history['baseline_acc'].append(train_acc_baseline)
        history['ascender_acc'].append(train_acc_ascender)
        history['alpha_mean'].append(alpha.mean().item())
        history['alpha_std'].append(alpha.std().item())
        history['alpha_min'].append(alpha.min().item())
        history['alpha_max'].append(alpha.max().item())

        # Log
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:03d}/{n_epochs}")
            print(f"  Baseline: Loss={train_loss_baseline:.4f}, Acc={train_acc_baseline:.3f}")
            print(f"  ASCender: Loss={train_loss_ascender:.4f}, Acc={train_acc_ascender:.3f}")
            print(f"  α: mean={alpha.mean().item():.4f}, std={alpha.std().item():.4f}, "
                  f"min={alpha.min().item():.4f}, max={alpha.max().item():.4f}")
            print()

        if train_acc_baseline > best_acc_baseline:
            best_acc_baseline = train_acc_baseline
        if train_acc_ascender > best_acc_ascender:
            best_acc_ascender = train_acc_ascender

    # Final analysis
    print("="*80)
    print("📊 FINAL ANALYSIS")
    print("="*80)
    print()

    alpha_final = ascender.get_alpha()
    print("1️⃣  Final α Values (per head):")
    print(f"   {alpha_final.numpy()}")
    print()

    mean_alpha = alpha_final.mean().item()
    std_alpha = alpha_final.std().item()
    min_alpha = alpha_final.min().item()
    max_alpha = alpha_final.max().item()

    print("2️⃣  α Statistics:")
    print(f"   Mean: {mean_alpha:.4f}")
    print(f"   Std:  {std_alpha:.4f}")
    print(f"   Min:  {min_alpha:.4f}")
    print(f"   Max:  {max_alpha:.4f}")
    print()

    print("3️⃣  Best Training Performance:")
    print(f"   Baseline: Acc={best_acc_baseline:.3f}")
    print(f"   ASCender: Acc={best_acc_ascender:.3f}")
    if best_acc_ascender > best_acc_baseline:
        improvement = (best_acc_ascender - best_acc_baseline) * 100
        print(f"   ✅ ASCender improved by {improvement:.1f}%")
    print()

    # Interpretation
    print("="*80)
    print("🎯 INTERPRETATION")
    print("="*80)
    print()

    # Compare with synthetic
    print("📊 Comparison with Synthetic Data:")
    print("   Synthetic (clusters): α ≈ 0.5039")
    print(f"   Real (MSRAction3D):   α ≈ {mean_alpha:.4f}")
    print()

    if mean_alpha < 0.40:
        print("✅ Spatial bias DOMINATES on real data (α < 0.4)")
        print("   → Real 3D spatial structure is valuable!")
        print("   → ASCender successfully captures spatial patterns")
        print()
        print("🎉 SUCCESS!")
        print("   - Boids-inspired bias provides strong value on real data")
        print("   - Ready for full experiments & paper")

    elif mean_alpha < 0.5:
        delta = 0.5 - mean_alpha
        print(f"✅ Spatial bias STRONGER on real data (α decreased by {delta:.4f})")
        print("   → Real spatial structure more informative than synthetic")
        print("   → ASCender bias provides measurable value")
        print()
        print("💡 Next steps:")
        print("   - Train longer (more epochs)")
        print("   - Try strong bias (2x) if not already")
        print("   - Consider architectural improvements")

    else:
        print("⚠️  Still BALANCED mixing (α ≈ 0.5)")
        print("   → Real data also shows balanced preference")
        print()
        print("💡 Possible reasons:")
        print("   1. Need more training epochs")
        print("   2. Bias strength too weak (try 2x)")
        print("   3. Dataset characteristics (sparse depth)")
        print("   4. Model capacity insufficient")

    print()
    print("="*80)

    # α learning dynamics
    alpha_initial = history['alpha_mean'][0]
    alpha_final_mean = history['alpha_mean'][-1]
    alpha_drift = abs(alpha_final_mean - alpha_initial)

    print("4️⃣  α Learning Dynamics:")
    print(f"   Initial α: {alpha_initial:.4f}")
    print(f"   Final α:   {alpha_final_mean:.4f}")
    print(f"   Drift:     {alpha_drift:.4f}")
    print()

    if alpha_drift > 0.1:
        print("   ✅ α learned significantly (drift > 0.1)")
        print("   → Model actively adjusted preference")
    elif alpha_drift > 0.05:
        print("   ⚠️  α learned moderately (0.05 < drift < 0.1)")
        print("   → Some preference learning, but limited")
    else:
        print("   ⚠️  α stayed near initialization (drift < 0.05)")
        print("   → Need longer training or stronger bias")

    print()
    print("="*80)
    print("✅ Experiment completed!")
    print("="*80)
    print()

    return history, alpha_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='src/data/Depth/',
                        help='Path to extracted MSRAction3D Depth folder')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--n_points', type=int, default=256,
                        help='Number of points per frame')
    parser.add_argument('--n_frames', type=int, default=8,
                        help='Number of frames per sequence')
    parser.add_argument('--strong_bias', action='store_true',
                        help='Use 2x stronger bias')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    history, alpha = train_and_evaluate(
        data_root=args.data_root,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        n_points=args.n_points,
        n_frames=args.n_frames,
        strong_bias=args.strong_bias,
        device=device
    )
