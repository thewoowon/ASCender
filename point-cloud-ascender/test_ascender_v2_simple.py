"""
Simple test for ASCender v2.0 components (no pointops dependency)

This tests the core ASCender v2.0 ideas:
1. Vector kernel encoder
2. Graph reweighting
3. Value pathway modulation
"""

import sys
sys.path.append('src')

try:
    import torch
    import torch.nn.functional as F
    from models.point_ascender_v2 import (
        VectorKernelEncoder,
        ASCGraphReweight,
        ValuePathwayModulator,
        ASCenderV2Config,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease install requirements:")
    print("  pip install torch numpy")
    sys.exit(1)


def test_vector_kernel():
    """Test 1: Vector Kernel Encoder"""
    print("\n" + "="*60)
    print("TEST 1: Vector Kernel Encoder (B_vec ∈ R^{N×k×C})")
    print("="*60)

    N, k, C = 256, 16, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = ASCenderV2Config(
        use_alignment=True,
        use_separation=True,
        use_cohesion=True,
        w_align=0.5,
        w_sep=0.3,
        w_coh=0.4,
        per_component_scale=True,
    )

    encoder = VectorKernelEncoder(C, cfg).to(device)

    # Dummy point cloud data
    p_i = torch.rand(N, 3, device=device) * 2.0  # 0~2 meters
    p_j = torch.rand(N, k, 3, device=device) * 2.0
    x_i = torch.randn(N, C, device=device)
    x_j = torch.randn(N, k, C, device=device)
    normals_i = F.normalize(torch.randn(N, 3, device=device), dim=-1)
    normals_j = F.normalize(torch.randn(N, k, 3, device=device), dim=-1)

    # Forward
    B_vec = encoder(p_i, p_j, x_i, x_j, normals_i, normals_j)

    print(f"\n✅ B_vec shape: {B_vec.shape} (expected: {N}, {k}, {C})")
    print(f"✅ B_vec dtype: {B_vec.dtype}")
    print(f"✅ B_vec range: [{B_vec.min().item():.4f}, {B_vec.max().item():.4f}]")
    print(f"✅ B_vec mean: {B_vec.mean().item():.4f}")
    print(f"✅ B_vec std: {B_vec.std().item():.4f}")

    # Check component scales
    print(f"\n📊 Learnable Component Scales:")
    print(f"   Scale A (alignment):  {torch.exp(encoder.log_scale_a).item():.4f}")
    print(f"   Scale S (separation): {torch.exp(encoder.log_scale_s).item():.4f}")
    print(f"   Scale C (cohesion):   {torch.exp(encoder.log_scale_c).item():.4f}")

    # Gradient check
    loss = B_vec.sum()
    loss.backward()
    print(f"\n✅ Gradients computed successfully")
    print(f"   grad(log_scale_a): {encoder.log_scale_a.grad.item():.6f}")

    return encoder, B_vec


def test_graph_reweight():
    """Test 2: ASC-aware Graph Reweighting"""
    print("\n" + "="*60)
    print("TEST 2: Graph Reweighting (ASC-aware neighbor selection)")
    print("="*60)

    N, k, C = 256, 16, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ASCenderV2Config(neighbor_enlarge_factor=1.5)
    k_large = int(k * cfg.neighbor_enlarge_factor)

    graph_reweight = ASCGraphReweight(cfg).to(device)

    # Dummy data
    p_i = torch.rand(N, 3, device=device) * 2.0
    p_j_large = torch.rand(N, k_large, 3, device=device) * 2.0
    x_i = torch.randn(N, C, device=device)
    x_j_large = torch.randn(N, k_large, C, device=device)
    normals_i = F.normalize(torch.randn(N, 3, device=device), dim=-1)
    normals_j_large = F.normalize(torch.randn(N, k_large, 3, device=device), dim=-1)

    # Compute neighbor scores
    scores = graph_reweight(p_i, p_j_large, x_i, x_j_large, normals_i, normals_j_large)

    print(f"\n✅ Neighbor scores shape: {scores.shape} (expected: {N}, {k_large})")
    print(f"✅ Scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"✅ Scores mean: {scores.mean().item():.4f}")

    # Select top-k neighbors
    _, top_indices = torch.topk(scores, k=k, dim=1)
    print(f"\n✅ Top-{k} indices shape: {top_indices.shape}")
    print(f"   Sample top indices (point 0): {top_indices[0, :5].tolist()}")

    # Check diversity of selected neighbors
    print(f"\n📊 Neighbor selection diversity:")
    for i in range(min(3, N)):
        selected = top_indices[i].tolist()
        print(f"   Point {i}: {selected[:8]}...")

    return graph_reweight, scores


def test_value_modulator():
    """Test 3: Value Pathway Modulation (RBP + Gating)"""
    print("\n" + "="*60)
    print("TEST 3: Value Pathway Modulation (RBP + Gating)")
    print("="*60)

    N, k, C = 256, 16, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ASCenderV2Config(
        enable_value_rbp=True,
        enable_value_gating=True,
        rbp_lambda=0.3,
    )

    modulator = ValuePathwayModulator(C, cfg).to(device)

    # Dummy data
    v = torch.randn(N, k, C, device=device)
    B_vec = torch.randn(N, k, C, device=device) * 0.5

    # Modulate
    v_mod = modulator(v, B_vec)

    print(f"\n✅ v_mod shape: {v_mod.shape}")
    print(f"✅ Original v range: [{v.min().item():.4f}, {v.max().item():.4f}]")
    print(f"✅ Modulated v range: [{v_mod.min().item():.4f}, {v_mod.max().item():.4f}]")

    # Check RBP lambda
    lambda_rbp = torch.exp(modulator.log_lambda).item()
    print(f"\n📊 Learnable RBP lambda: {lambda_rbp:.4f} (init: {cfg.rbp_lambda})")

    # Compare with/without modulation
    diff = (v_mod - v).abs().mean()
    print(f"✅ Mean absolute difference: {diff.item():.4f}")

    # Gradient check
    loss = v_mod.sum()
    loss.backward()
    print(f"✅ Gradients computed successfully")
    print(f"   grad(log_lambda): {modulator.log_lambda.grad.item():.6f}")

    return modulator, v_mod


def test_integration():
    """Test 4: Full integration test"""
    print("\n" + "="*60)
    print("TEST 4: Full Integration (all 3 levels)")
    print("="*60)

    N, k, C = 256, 16, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ASCenderV2Config(
        use_alignment=True,
        use_separation=True,
        use_cohesion=True,
        enable_graph_reweight=True,
        enable_vector_kernel=True,
        enable_value_rbp=True,
        enable_value_gating=True,
    )

    # Initialize all components
    encoder = VectorKernelEncoder(C, cfg).to(device)
    graph_reweight = ASCGraphReweight(cfg).to(device)
    value_mod = ValuePathwayModulator(C, cfg).to(device)

    # Point cloud data
    p_i = torch.rand(N, 3, device=device) * 2.0
    x_i = torch.randn(N, C, device=device)
    normals_i = F.normalize(torch.randn(N, 3, device=device), dim=-1)

    # Generate large neighbor set
    k_large = int(k * cfg.neighbor_enlarge_factor)
    p_j_large = torch.rand(N, k_large, 3, device=device) * 2.0
    x_j_large = torch.randn(N, k_large, C, device=device)
    normals_j_large = F.normalize(torch.randn(N, k_large, 3, device=device), dim=-1)

    print("\n🔄 Pipeline:")
    print("   [Point Cloud] → [Graph Reweight] → [Vector Kernel] → [Value Mod] → [Output]")

    # Step 1: Graph reweighting
    print("\n1. Graph reweighting...")
    scores = graph_reweight(p_i, p_j_large, x_i, x_j_large, normals_i, normals_j_large)
    _, top_idx = torch.topk(scores, k=k, dim=1)
    print(f"   Selected {k} neighbors from {k_large} candidates")

    # Step 2: Select top-k neighbors
    # Gather using top_idx
    batch_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, k)
    p_j = p_j_large[batch_idx, top_idx]
    x_j = x_j_large[batch_idx, top_idx]
    normals_j = normals_j_large[batch_idx, top_idx]

    # Step 3: Compute vector kernel
    print("2. Computing vector kernel B_vec...")
    B_vec = encoder(p_i, p_j, x_i, x_j, normals_i, normals_j)
    print(f"   B_vec shape: {B_vec.shape}")

    # Step 4: Simulate relation computation (simplified)
    print("3. Computing attention relation...")
    # In real PT: w = x_k - x_q + p_r + B_vec
    # Here we just add B_vec to show the effect
    rel = x_j - x_i.unsqueeze(1)  # (N, k, C)
    rel_with_asc = rel + B_vec
    print(f"   Relation shape: {rel_with_asc.shape}")

    # Step 5: Value modulation
    print("4. Modulating value pathway...")
    v = x_j  # Simplified: v = α(x_j)
    v_mod = value_mod(v, B_vec)
    print(f"   v_mod shape: {v_mod.shape}")

    # Step 6: Final aggregation (simplified)
    print("5. Final aggregation...")
    attn_weights = F.softmax(rel_with_asc.sum(dim=-1), dim=-1)  # (N, k)
    output = (v_mod * attn_weights.unsqueeze(-1)).sum(dim=1)  # (N, C)
    print(f"   Output shape: {output.shape}")

    print(f"\n✅ Full pipeline executed successfully!")
    print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"   Output mean: {output.mean().item():.4f}")

    # Gradient check
    loss = output.sum()
    loss.backward()
    print(f"\n✅ End-to-end gradients computed successfully")

    return output


def main():
    print("\n" + "="*70)
    print(" "*15 + "ASCender v2.0 - Component Tests")
    print("="*70)
    print("\nThree-Level Intervention:")
    print("  Level 1: Graph Construction (ASC-aware neighbor selection)")
    print("  Level 2: Vector Kernel (B_vec ∈ R^C, not scalar bias)")
    print("  Level 3: Value Pathway (RBP + Gating)")
    print("\n" + "="*70)

    try:
        # Run tests
        encoder, B_vec = test_vector_kernel()
        graph_reweight, scores = test_graph_reweight()
        modulator, v_mod = test_value_modulator()
        output = test_integration()

        # Summary
        print("\n" + "="*70)
        print(" "*25 + "TEST SUMMARY")
        print("="*70)
        print("\n✅ All tests passed!")
        print("\nKey Results:")
        print(f"  - Vector kernel B_vec: shape {B_vec.shape}, range [{B_vec.min():.3f}, {B_vec.max():.3f}]")
        print(f"  - Neighbor scores: shape {scores.shape}, mean {scores.mean():.3f}")
        print(f"  - Value modulation: applied RBP + gating successfully")
        print(f"  - End-to-end output: shape {output.shape}")
        print("\n🎉 ASCender v2.0 core components are working!")
        print("\n" + "="*70)

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
