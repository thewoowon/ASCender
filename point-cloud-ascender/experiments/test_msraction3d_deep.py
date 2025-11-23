"""
MSRAction3D Deep Architecture Experiment

Purpose: Test ASCender with DEEP multi-layer architecture
Goal: Get accuracy to 20-30% to see real α effects

Improvements:
1. Multi-layer attention (3 layers)
2. Better temporal modeling (Temporal Transformer)
3. Larger capacity (d_model=256)
4. Proper pre-normalization
5. Gradient clipping & better optimization
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


class PointAttentionLayer(nn.Module):
    """Single attention layer with optional ASCender bias"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_ascender: bool = False,
        strong_bias: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_ascender = use_ascender
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Pre-normalization
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ASCender bias
        if use_ascender:
            if strong_bias:
                cfg = PointAscenderConfig(
                    use_alignment=True,
                    use_separation=True,
                    use_cohesion=True,
                    per_head_scale=True,
                    per_head_gate=True,
                    w_align=0.80,
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
            self.alpha_logit = nn.Parameter(torch.zeros(n_heads))

        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, xyz, normals):
        """
        Args:
            x: (B, N, d_model) features
            xyz: (B, N, 3) coordinates
            normals: (B, N, 3) normals
        """
        B, N, _ = x.shape

        # Pre-norm
        x_norm = self.norm1(x)

        # Self-attention
        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # ASCender bias with RBP
        if self.use_ascender:
            spatial_bias = self.biaser(q, k, xyz, xyz, normals, normals)
            alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1, 1)
            attn_scores = alpha * attn_scores + (1 - alpha) * spatial_bias

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.o_proj(out)
        out = self.dropout(out)

        # Residual
        x = x + out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x

    def get_alpha(self):
        if self.use_ascender:
            return torch.sigmoid(self.alpha_logit).detach().cpu()
        return None


class TemporalTransformer(nn.Module):
    """Temporal Transformer for aggregating frame features"""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model) temporal sequence
        """
        B, T, _ = x.shape

        # Pre-norm
        x_norm = self.norm1(x)

        # Self-attention over time
        q = self.q_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out)
        out = self.dropout(out)

        x = x + out
        x = x + self.ffn(self.norm2(x))

        return x


class DeepTemporalPointClassifier(nn.Module):
    """
    Deep multi-layer architecture for action recognition

    Architecture:
    - Embedding layer
    - 3x Point Attention Layers (with optional ASCender)
    - Temporal Transformer
    - Classification head
    """

    def __init__(
        self,
        n_classes: int = 20,
        d_model: int = 256,
        n_heads: int = 16,
        n_layers: int = 3,
        use_ascender: bool = False,
        strong_bias: bool = False,
        dropout: float = 0.2
    ):
        super().__init__()
        self.use_ascender = use_ascender
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Multi-layer point attention
        self.point_layers = nn.ModuleList([
            PointAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                use_ascender=use_ascender,
                strong_bias=strong_bias,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Frame-level normalization
        self.frame_norm = nn.LayerNorm(d_model)

        # Temporal transformer
        self.temporal_transformer = TemporalTransformer(
            d_model=d_model,
            n_heads=n_heads // 2,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, xyz, normals):
        """
        Args:
            xyz: (B, T, N, 3)
            normals: (B, T, N, 3)
        """
        B, T, N, _ = xyz.shape

        # Process each frame
        frame_features = []

        for t in range(T):
            xyz_t = xyz[:, t]  # (B, N, 3)
            normals_t = normals[:, t]  # (B, N, 3)

            # Embed
            x = self.embedding(xyz_t)  # (B, N, d_model)

            # Multi-layer attention
            for layer in self.point_layers:
                x = layer(x, xyz_t, normals_t)

            # Frame pooling
            x = self.frame_norm(x)
            frame_feat = x.mean(dim=1)  # (B, d_model)
            frame_features.append(frame_feat)

        # Temporal sequence
        temporal_feats = torch.stack(frame_features, dim=1)  # (B, T, d_model)

        # Temporal transformer
        temporal_feats = self.temporal_transformer(temporal_feats)

        # Temporal pooling
        pooled = temporal_feats.mean(dim=1)  # (B, d_model)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def get_alpha(self):
        """Get α from all layers"""
        alphas = []
        for layer in self.point_layers:
            alpha = layer.get_alpha()
            if alpha is not None:
                alphas.append(alpha)

        if alphas:
            return torch.stack(alphas, dim=0).mean(dim=0)  # Average across layers
        return None


def train_and_evaluate(
    data_root: str,
    n_epochs: int = 50,
    batch_size: int = 8,
    n_points: int = 512,
    n_frames: int = 16,
    strong_bias: bool = False,
    device: str = 'cpu'
):
    """Train deep architecture"""

    print("="*80)
    print("🚀 MSRAction3D DEEP Architecture Experiment")
    print("="*80)
    print()
    print(f"📁 Data root: {data_root}")
    print(f"📱 Device: {device}")
    print(f"🔢 Epochs: {n_epochs}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔹 Points per frame: {n_points}")
    print(f"🎞️  Frames per sequence: {n_frames}")
    print(f"💪 Strong bias: {'Yes (2x)' if strong_bias else 'No (1x)'}")
    print(f"🏗️  Architecture: 3-layer + Temporal Transformer + d_model=256")
    print()

    # Datasets
    print("📂 Loading datasets...")
    train_dataset = MSRAction3DPointCloudDataset(
        data_root=data_root,
        n_points=n_points,
        n_frames=n_frames,
        split='train'
    )

    test_dataset = MSRAction3DPointCloudDataset(
        data_root=data_root,
        n_points=n_points,
        n_frames=n_frames,
        split='test'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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
    baseline = DeepTemporalPointClassifier(
        n_classes=20,
        d_model=256,
        n_heads=16,
        n_layers=3,
        use_ascender=False,
        dropout=0.2
    ).to(device)

    ascender = DeepTemporalPointClassifier(
        n_classes=20,
        d_model=256,
        n_heads=16,
        n_layers=3,
        use_ascender=True,
        strong_bias=strong_bias,
        dropout=0.2
    ).to(device)

    n_params_baseline = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    n_params_ascender = sum(p.numel() for p in ascender.parameters() if p.requires_grad)

    print(f"   Baseline: {n_params_baseline:,} parameters")
    print(f"   ASCender: {n_params_ascender:,} parameters (+{n_params_ascender - n_params_baseline:,})")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    opt_baseline = torch.optim.AdamW(baseline.parameters(), lr=2e-4, weight_decay=1e-4)
    opt_ascender = torch.optim.AdamW(ascender.parameters(), lr=2e-4, weight_decay=1e-4)

    sched_baseline = torch.optim.lr_scheduler.CosineAnnealingLR(opt_baseline, T_max=n_epochs)
    sched_ascender = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ascender, T_max=n_epochs)

    print("🏋️  Training...\n")

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
        baseline.train()
        ascender.train()

        train_loss_baseline = 0.0
        train_loss_ascender = 0.0
        train_acc_baseline = 0.0
        train_acc_ascender = 0.0
        n_batches = 0

        for batch in train_loader:
            xyz = batch['xyz'].to(device)
            normals = batch['normals'].to(device)
            labels = batch['label'].to(device)

            # Baseline
            opt_baseline.zero_grad()
            logits_baseline = baseline(xyz, normals)
            loss_baseline = criterion(logits_baseline, labels)
            loss_baseline.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            opt_baseline.step()

            train_loss_baseline += loss_baseline.item()
            train_acc_baseline += (logits_baseline.argmax(dim=-1) == labels).float().mean().item()

            # ASCender
            opt_ascender.zero_grad()
            logits_ascender = ascender(xyz, normals)
            loss_ascender = criterion(logits_ascender, labels)
            loss_ascender.backward()
            torch.nn.utils.clip_grad_norm_(ascender.parameters(), 1.0)
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
    print("2️⃣  α Statistics:")
    print(f"   Mean: {mean_alpha:.4f}")
    print(f"   Std:  {alpha_final.std().item():.4f}")
    print(f"   Min:  {alpha_final.min().item():.4f}")
    print(f"   Max:  {alpha_final.max().item():.4f}")
    print()

    print("3️⃣  Best Training Performance:")
    print(f"   Baseline: Acc={best_acc_baseline:.3f}")
    print(f"   ASCender: Acc={best_acc_ascender:.3f}")
    if best_acc_ascender > best_acc_baseline:
        improvement = (best_acc_ascender - best_acc_baseline) * 100
        print(f"   ✅ ASCender improved by {improvement:.1f}%")
    print()

    print("="*80)
    print("🎯 INTERPRETATION")
    print("="*80)
    print()

    if mean_alpha < 0.40:
        print("🎉 SUCCESS! Spatial bias DOMINATES (α < 0.4)")
        print("   → Deep architecture + real data reveals spatial structure!")
    elif mean_alpha < 0.47:
        print("✅ Spatial bias VALUABLE (α < 0.47)")
        print("   → Clear preference for spatial structure")
    else:
        print("⚖️  Still balanced (α ≈ 0.5)")
        print("   → Need even stronger architecture or more data")

    print()
    print("="*80)
    print("✅ Deep experiment completed!")
    print("="*80)

    return history, alpha_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='src/data/Depth/')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_points', type=int, default=512)
    parser.add_argument('--n_frames', type=int, default=16)
    parser.add_argument('--strong_bias', action='store_true')

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
