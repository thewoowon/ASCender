"""
RBP (Residual Bias Path) Learning Test - STRONG BIAS VERSION

Purpose: Test if α learns away from 0.5 with 2x stronger spatial bias
Key Question: Does stronger bias force model to rely on spatial structure?

Expected Behavior:
- α < 0.4: Spatial bias dominates (SUCCESS!)
- α ≈ 0.5: Still balanced (need real data)
- α > 0.6: Learned attention dominates (bias still not strong enough)

Differences from original:
- w_align: 0.40 → 0.80 (2x)
- w_sep: 0.20 → 0.40 (2x)
- w_coh: 0.30 → 0.60 (2x)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.point_ascender_bias import PointAscenderBias, PointAscenderConfig


class SimplePointClassifier(nn.Module):
    """
    Minimal point cloud classifier with ASCender bias

    Architecture:
    - Embedding: xyz → features
    - Self-attention with optional ASCender bias
    - RBP: α * learned_attn + (1-α) * spatial_bias
    - Global pooling → Classification
    """
    def __init__(self, n_classes=20, d_model=64, n_heads=8, use_ascender=False, strong_bias=False):
        super().__init__()
        self.use_ascender = use_ascender
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Simple embedding
        self.embedding = nn.Linear(3, d_model)
        self.norm1 = nn.LayerNorm(d_model)

        # Self-attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # ASCender bias (if enabled)
        if use_ascender:
            if strong_bias:
                # 2x stronger bias
                cfg = PointAscenderConfig(
                    use_alignment=True,
                    use_separation=True,
                    use_cohesion=True,
                    per_head_scale=True,
                    per_head_gate=True,
                    w_align=0.80,   # 2x from 0.40
                    w_sep=0.40,     # 2x from 0.20
                    w_coh=0.60,     # 2x from 0.30
                    sigma_sep=0.05,
                    sigma_coh=0.50,
                )
            else:
                # Original strength
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
            nn.Linear(d_model * 2, d_model),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, xyz, normals):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            normals: (B, N, 3) surface normals
        Returns:
            logits: (B, n_classes)
        """
        B, N, _ = xyz.shape

        # Embed points
        x = self.embedding(xyz)  # (B, N, d_model)
        x = self.norm1(x)

        # Self-attention
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, N, d_head)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # Learned attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, H, N, N)

        # Apply ASCender bias with RBP
        if self.use_ascender:
            # Get spatial bias
            spatial_bias = self.biaser(q, k, xyz, xyz, normals, normals)  # (B, H, N, N)

            # Residual Bias Path: α * learned + (1-α) * spatial
            alpha = torch.sigmoid(self.alpha_logit).view(1, -1, 1, 1)  # (1, H, 1, 1)
            attn_scores = alpha * attn_scores + (1 - alpha) * spatial_bias

        # Softmax and apply
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)  # (B, H, N, d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.o_proj(out)

        # Residual
        x = x + out

        # Feedforward
        x = x + self.ffn(self.norm2(x))

        # Global pooling and classify
        x = x.mean(dim=1)  # (B, d_model)
        logits = self.classifier(x)

        return logits

    def get_alpha(self):
        """Get current α values"""
        if self.use_ascender:
            return torch.sigmoid(self.alpha_logit).detach().cpu()
        return None


def generate_batch(batch_size=16, n_points=256, n_classes=20, device='cpu'):
    """
    Generate synthetic point cloud batch

    Strategy: Create clusters in 3D space, each cluster = one class
    This gives spatial structure that ASCender can exploit
    """
    xyz_list = []
    normals_list = []
    labels = torch.randint(0, n_classes, (batch_size,), device=device)

    for i in range(batch_size):
        # Create a cluster center based on class
        class_id = labels[i].item()
        center = torch.tensor([
            (class_id % 5) * 0.5,      # x: 0, 0.5, 1.0, 1.5, 2.0
            (class_id // 5) * 0.5,     # y: similar
            (class_id % 3) * 0.3,      # z: variation
        ], device=device)

        # Generate points around cluster center
        points = center.unsqueeze(0) + torch.randn(n_points, 3, device=device) * 0.15
        xyz_list.append(points)

        # Generate normals pointing outward from center
        normals = F.normalize(points - center.unsqueeze(0) + torch.randn(n_points, 3, device=device) * 0.05, dim=-1)
        normals_list.append(normals)

    xyz = torch.stack(xyz_list, dim=0)  # (B, N, 3)
    normals = torch.stack(normals_list, dim=0)  # (B, N, 3)

    return xyz, normals, labels


def train_and_analyze(n_epochs=50, batch_size=16, device='cpu'):
    """
    Train baseline vs ASCender (strong bias) and compare α learning
    """
    print("="*80)
    print("🚀 RBP Learning Test - STRONG BIAS VERSION (2x)")
    print("="*80)
    print()
    print(f"📱 Device: {device}")
    print(f"🔢 Epochs: {n_epochs}")
    print(f"📦 Batch size: {batch_size}")
    print()
    print("💪 Bias Strength:")
    print("   Original → Strong (2x)")
    print("   w_align: 0.40 → 0.80")
    print("   w_sep:   0.20 → 0.40")
    print("   w_coh:   0.30 → 0.60")
    print()

    # Models
    baseline = SimplePointClassifier(n_classes=20, use_ascender=False).to(device)
    ascender_strong = SimplePointClassifier(n_classes=20, use_ascender=True, strong_bias=True).to(device)

    # Count parameters
    n_params_baseline = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    n_params_ascender = sum(p.numel() for p in ascender_strong.parameters() if p.requires_grad)
    print(f"🏗️  Baseline parameters: {n_params_baseline:,}")
    print(f"🏗️  ASCender parameters: {n_params_ascender:,} (+{n_params_ascender - n_params_baseline:,})")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    opt_baseline = torch.optim.AdamW(baseline.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_ascender = torch.optim.AdamW(ascender_strong.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler
    sched_baseline = torch.optim.lr_scheduler.CosineAnnealingLR(opt_baseline, T_max=n_epochs)
    sched_ascender = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ascender, T_max=n_epochs)

    print("🏋️  Training...\n")

    # Track history
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

    for epoch in range(n_epochs):
        # Training
        baseline.train()
        ascender_strong.train()

        # Baseline
        opt_baseline.zero_grad()
        xyz, normals, labels = generate_batch(batch_size, device=device)
        logits_baseline = baseline(xyz, normals)
        loss_baseline = criterion(logits_baseline, labels)
        loss_baseline.backward()
        opt_baseline.step()
        sched_baseline.step()

        acc_baseline = (logits_baseline.argmax(dim=-1) == labels).float().mean().item()

        # ASCender (strong bias)
        opt_ascender.zero_grad()
        xyz, normals, labels = generate_batch(batch_size, device=device)
        logits_ascender = ascender_strong(xyz, normals)
        loss_ascender = criterion(logits_ascender, labels)
        loss_ascender.backward()
        opt_ascender.step()
        sched_ascender.step()

        acc_ascender = (logits_ascender.argmax(dim=-1) == labels).float().mean().item()

        # Get α values
        alpha = ascender_strong.get_alpha()

        # Record history
        history['baseline_loss'].append(loss_baseline.item())
        history['ascender_loss'].append(loss_ascender.item())
        history['baseline_acc'].append(acc_baseline)
        history['ascender_acc'].append(acc_ascender)
        history['alpha_mean'].append(alpha.mean().item())
        history['alpha_std'].append(alpha.std().item())
        history['alpha_min'].append(alpha.min().item())
        history['alpha_max'].append(alpha.max().item())

        # Log
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:03d}/{n_epochs}")
            print(f"  Baseline: Loss={loss_baseline.item():.4f}, Acc={acc_baseline:.3f}")
            print(f"  ASCender: Loss={loss_ascender.item():.4f}, Acc={acc_ascender:.3f}")
            print(f"  α: mean={alpha.mean().item():.4f}, std={alpha.std().item():.4f}, "
                  f"min={alpha.min().item():.4f}, max={alpha.max().item():.4f}")
            print()

    # Final analysis
    print("="*80)
    print("📊 FINAL ANALYSIS")
    print("="*80)
    print()

    # Final α values
    alpha_final = ascender_strong.get_alpha()
    print("1️⃣  Final α Values (per head):")
    print(f"   {alpha_final.numpy()}")
    print()

    # Statistics
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

    # Final performance
    print("3️⃣  Final Performance:")
    print(f"   Baseline: Loss={history['baseline_loss'][-1]:.4f}, Acc={history['baseline_acc'][-1]:.3f}")
    print(f"   ASCender: Loss={history['ascender_loss'][-1]:.4f}, Acc={history['ascender_acc'][-1]:.3f}")
    print()

    # Interpretation
    print("="*80)
    print("🎯 INTERPRETATION")
    print("="*80)
    print()

    if mean_alpha > 0.6:
        print("❌ Learned attention STILL DOMINATES (α > 0.6)")
        print("   → Even 2x bias not strong enough")
        print("   → Synthetic data may be too simple")
        print()
        print("💡 Implications:")
        print("   - Spatial structure in synthetic data is weak")
        print("   - MUST try real data (MSRAction3D)")
        print("   - OR create more complex synthetic data")

    elif mean_alpha < 0.4:
        print("✅ Spatial bias DOMINATES (α < 0.4)")
        print("   → 2x stronger bias worked!")
        print("   → Model relies heavily on spatial structure")
        print()
        print("🎉 SUCCESS! This is what we hoped for!")
        print("   - Spatial structure is learnable and useful")
        print("   - Stronger bias reveals its value")
        print("   - Ready for MSRAction3D experiments")

    else:
        print("⚖️  STILL BALANCED mixing (0.4 ≤ α ≤ 0.6)")
        print("   → 2x bias still not enough to dominate")
        print("   → Synthetic data may be too simple")
        print()
        print("💡 Implications:")
        print("   - Architecture works correctly")
        print("   - But synthetic data lacks real spatial structure")
        print("   - PROCEED to MSRAction3D (real data)")

    print()
    print("="*80)

    # Check if α changed significantly during training
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
        print("   → RBP is working correctly")
        print("   → Stronger bias changed model preference!")
    else:
        print("   ⚠️  α stayed near initialization (drift < 0.1)")
        print("   → Model didn't find strong preference even with 2x bias")
        print("   → Synthetic data likely too simple")

    print()
    print("="*80)

    # Comparison with original test
    print("5️⃣  Comparison with Original Test:")
    print("   Original test (1x bias): α ≈ 0.5039 (balanced)")
    print(f"   Strong test (2x bias):   α ≈ {mean_alpha:.4f}")
    print()

    if mean_alpha < 0.5039:
        delta = 0.5039 - mean_alpha
        print(f"   ✅ α decreased by {delta:.4f}")
        print("   → Stronger bias pushed α toward spatial!")
    else:
        delta = mean_alpha - 0.5039
        print(f"   ⚠️  α increased by {delta:.4f}")
        print("   → Unexpected - bias didn't help")

    print()
    print("="*80)
    print("✅ Test completed!")
    print("="*80)
    print()

    print("🎯 Next Steps:")
    if mean_alpha < 0.4:
        print("   ✅ SUCCESS! Proceed to MSRAction3D")
        print("   1. Extract MSRAction3D depth data")
        print("   2. Implement data loader")
        print("   3. Run full experiments with strong bias")
    else:
        print("   ⚠️  Synthetic data insufficient:")
        print("   1. Extract MSRAction3D depth data (PRIORITY)")
        print("   2. Implement data loader")
        print("   3. Test on REAL spatial structure")
        print("   4. If α < 0.4 on real data → SUCCESS!")
    print()

    return history, alpha_final


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history, alpha = train_and_analyze(n_epochs=50, batch_size=16, device=device)
