"""
Minimal test: ASCender bias on synthetic point clouds

Purpose: Verify that Point Cloud ASCender bias module works correctly
Expected: Should generate spatial bias from 3D coordinates and normals
"""
import sys
sys.path.insert(0, '../src')

import torch
import torch.nn.functional as F
from src.models.point_ascender_bias import PointAscenderBias, PointAscenderConfig


def test_minimal():
    print("="*70)
    print("🧪 Point Cloud ASCender - Minimal Test")
    print("="*70)
    print()

    # Setup
    B, N, d_model = 4, 256, 64
    n_heads = 8
    d_head = d_model // n_heads
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    print(f"📊 Config: B={B}, N={N}, Heads={n_heads}, d_head={d_head}")
    print()

    # Config
    cfg = PointAscenderConfig(
        use_alignment=True,
        use_separation=True,
        use_cohesion=True,
        per_head_scale=True,
        w_align=0.30,
        w_sep=0.15,
        w_coh=0.25,
        sigma_sep=0.05,  # 5cm separation range
        sigma_coh=0.50,  # 50cm cohesion range
    )

    print("⚙️  ASCender Config:")
    print(f"   Alignment:  {'✅' if cfg.use_alignment else '❌'} (w={cfg.w_align})")
    print(f"   Separation: {'✅' if cfg.use_separation else '❌'} (w={cfg.w_sep}, σ={cfg.sigma_sep}m)")
    print(f"   Cohesion:   {'✅' if cfg.use_cohesion else '❌'} (w={cfg.w_coh}, σ={cfg.sigma_coh}m)")
    print()

    # Model
    print("🏗️  Building model...")
    biaser = PointAscenderBias(cfg, n_heads=n_heads).to(device)
    n_params = sum(p.numel() for p in biaser.parameters())
    print(f"✅ Model created: {n_params} parameters")
    print()

    # Synthetic data
    print("🎲 Generating synthetic point cloud...")
    xyz = torch.rand(B, N, 3, device=device) * 2.0  # 0~2 meters
    normals = F.normalize(torch.randn(B, N, 3, device=device), dim=-1)
    qh = torch.randn(B, n_heads, N, d_head, device=device)
    kh = torch.randn(B, n_heads, N, d_head, device=device)
    print(f"   xyz shape: {xyz.shape} (range: [{xyz.min():.2f}, {xyz.max():.2f}])")
    print(f"   normals shape: {normals.shape}")
    print()

    # Forward
    print("▶️  Forward pass...")
    bias = biaser(qh, kh, xyz, xyz, normals, normals)
    print(f"✅ Bias generated: {bias.shape}")
    print()

    # Results
    print("="*70)
    print("📊 RESULTS")
    print("="*70)
    print()

    print("1️⃣  Bias Statistics:")
    print(f"   Mean: {bias.mean().item():+.4f}")
    print(f"   Std:  {bias.std().item():.4f}")
    print(f"   Min:  {bias.min().item():+.4f}")
    print(f"   Max:  {bias.max().item():+.4f}")
    print()

    # Per-head analysis
    gamma = biaser._get_gamma()
    gate = biaser._get_gate()
    sigma_sep, sigma_coh = biaser._get_sigmas()

    print("2️⃣  Per-Head Parameters:")
    print(f"   γ (scale):  {gamma.cpu().detach().numpy()}")
    print(f"   Gate (σ):   {gate.cpu().detach().numpy()}")
    print(f"   σ_sep (m):  {sigma_sep.cpu().detach().numpy()}")
    print(f"   σ_coh (m):  {sigma_coh.cpu().detach().numpy()}")
    print()

    # Component analysis
    print("3️⃣  Component Contributions:")
    with torch.no_grad():
        # Alignment only
        cfg_a = PointAscenderConfig(
            use_alignment=True, use_separation=False, use_cohesion=False,
            w_align=cfg.w_align, per_head_scale=False
        )
        biaser_a = PointAscenderBias(cfg_a, n_heads=n_heads).to(device)
        bias_a = biaser_a(qh, kh, xyz, xyz, normals, normals)

        # Separation only
        cfg_s = PointAscenderConfig(
            use_alignment=False, use_separation=True, use_cohesion=False,
            w_sep=cfg.w_sep, per_head_scale=False
        )
        biaser_s = PointAscenderBias(cfg_s, n_heads=n_heads).to(device)
        bias_s = biaser_s(qh, kh, xyz, xyz, normals, normals)

        # Cohesion only
        cfg_c = PointAscenderConfig(
            use_alignment=False, use_separation=False, use_cohesion=True,
            w_coh=cfg.w_coh, per_head_scale=False
        )
        biaser_c = PointAscenderBias(cfg_c, n_heads=n_heads).to(device)
        bias_c = biaser_c(qh, kh, xyz, xyz, normals, normals)

        print(f"   Alignment (A):  mean={bias_a.mean().item():+.4f}, std={bias_a.std().item():.4f}")
        print(f"   Separation (S): mean={bias_s.mean().item():+.4f}, std={bias_s.std().item():.4f}")
        print(f"   Cohesion (C):   mean={bias_c.mean().item():+.4f}, std={bias_c.std().item():.4f}")
        print()

    print("="*70)
    print("✅ Test completed successfully!")
    print("="*70)
    print()
    print("🎯 Next steps:")
    print("   1. Run: python experiments/test_rbp_learning.py")
    print("   2. Check if α learns away from 0.5")
    print("   3. Analyze component contributions")
    print()


if __name__ == "__main__":
    test_minimal()
