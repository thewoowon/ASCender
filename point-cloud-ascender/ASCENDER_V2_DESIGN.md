# ASCender v2.0: Vector Kernel Edition

**Date**: 2025-11-23
**Status**: 🚀 Core Implementation Complete

---

## 🎯 Problem Statement

### ASCender v1.0 Failure Analysis

**Original Design (Language Model)**:
```python
attn = softmax(QK^T / sqrt(d) + B_scalar)  # B is (B, H, T, T) scalar bias
```

**Why it failed in Point Transformers:**
1. **Too weak**: In large models, learned attention `QK^T` dominates, scalar bias gets "buried"
2. **Wrong level**: Only affects attention scores (logits), doesn't shape the kernel or value pathway
3. **Mismatched domain**: Content-driven (text) bias applied to spatial-driven (3D) domain

**Point Transformer Reality (Zhao et al., ICCV 2021)**:
```
y_i = Σ_{j∈N(i)} ρ(γ(φ(x_i) - ψ(x_j) + δ_ij)) ⊙ (α(x_j) + δ_ij)
```

This is **vector attention** - each channel gets its own kernel. Adding a scalar bias is like adding one more `δ_ij` term. Not enough!

---

## 💡 Solution: Three-Level Intervention

Instead of just tweaking attention logits, ASCender v2.0 intervenes at **three critical levels**:

### **Level 1: Graph Construction** 🕸️
**Where**: Neighbor selection (before attention)
**What**: ASC-aware dynamic graph

```
Traditional: k-NN(p, k) → fixed k neighbors
ASCender v2: k-NN(p, k_large) → ASC_score → top-k selection
```

**Impact**:
- Separation: Exclude overcrowded/noisy neighbors
- Cohesion: Prefer neighbors from same object/cluster
- **Who you attend to** is ASC-controlled

**Implementation**:
```python
# Neighbor scoring: [align, separation, cohesion] → scalar score
scores = ASCGraphReweight(p_i, p_j_large, x_i, x_j_large, normals)  # (N, k_large)
_, top_idx = topk(scores, k)  # Select best k neighbors
```

---

### **Level 2: Vector Kernel** 🔬
**Where**: Relation computation (kernel generation)
**What**: B_vec ∈ R^C instead of B_scalar ∈ R

```
Traditional Point Transformer:
  rel_ij = φ(x_i) - ψ(x_j) + δ_ij        # δ_ij ∈ R^C (learned via MLP)

ASCender v2.0:
  rel_ij = φ(x_i) - ψ(x_j) + δ_ij + B_vec_ij   # B_vec_ij ∈ R^C (Boids-driven)
```

**Key Innovation**: Each Boids component generates a **C-dimensional vector**, not a scalar!

```python
A_vec = AlignEncoder(cos(normal_i, normal_j))      # (N, k, C)
S_vec = SepEncoder([distance, density])             # (N, k, C)
C_vec = CohEncoder([distance, feature_similarity])  # (N, k, C)

B_vec = w_a * A_vec + w_s * S_vec + w_c * C_vec    # (N, k, C)
```

**Why this works**:
- **Channel-specific kernels**: Each feature channel sees a different Boids influence
- **Interpretable**: Can visualize which channels are alignment-driven vs cohesion-driven
- **Strong enough**: Not a scalar add-on, but a full kernel modifier

**Contrast with v1.0**:
```python
# v1.0 (weak):
B_scalar = w_a * align_score + w_s * sep_score + w_c * coh_score  # (N, k)
logits = logits + B_scalar  # Broadcast to all channels equally

# v2.0 (strong):
B_vec = {different for each channel}  # (N, k, C)
rel = rel + B_vec  # Each channel gets custom Boids modulation
```

---

### **Level 3: Value Pathway** 🛤️
**Where**: Value modulation (how information flows)
**What**: RBP (Residual Bias Path) + Spatial Gating

```
Traditional Point Transformer:
  v_ij = α(x_j) + δ_ij

ASCender v2.0:
  v_ij' = gate(B_vec) ⊙ v_ij + λ * B_vec
         ^^^^^^^^^^^^^^^^      ^^^^^^^^^
         multiplicative        additive
```

**Two mechanisms**:

1. **Multiplicative Gating**:
```python
gate = sigmoid(GateNet(B_vec))  # (N, k, C)
v_gated = gate ⊙ v              # Channel-wise modulation
```
- Boids decides "how much to trust" each feature channel from each neighbor
- E.g., if separation is high → gate down noisy features

2. **Additive RBP** (Residual Bias Path):
```python
v_final = v_gated + λ * B_vec   # λ is learnable
```
- Direct injection of Boids-driven features
- Similar to residual connections, but spatial-aware

**Impact**:
- Not just "where to look" (attention), but **"what to take"** (value)
- Boids controls both the routing AND the content

---

## 🏗️ Architecture Overview

```
Input: Point Cloud [p, x, normals]
  |
  ├─── Level 1: Graph Construction ───────────────┐
  |    k-NN(k_large) → ASC_score → top-k          |
  |                                                |
  ├─── Level 2: Vector Kernel ────────────────────┤
  |    Compute:                                    |
  |      A_vec: alignment (normal similarity)      |
  |      S_vec: separation (distance + density)    |
  |      C_vec: cohesion (distance + feat_sim)     |
  |    B_vec = w_a*A + w_s*S + w_c*C               |
  |                                                |
  |    rel = φ(x_i) - ψ(x_j) + δ_ij + B_vec        |
  |    att = softmax(γ(rel))                       |
  |                                                |
  ├─── Level 3: Value Pathway ────────────────────┤
  |    v = α(x_j) + δ_ij                           |
  |    gate = sigmoid(GateNet(B_vec))              |
  |    v' = gate ⊙ v + λ * B_vec                   |
  |                                                |
  └─── Output: y_i = Σ_j att_ij ⊙ v'_ij ──────────┘
```

---

## 🔬 Technical Details

### 1. Vector Kernel Encoder

**Alignment**:
```python
# Input: cos(normal_i, normal_j) ∈ [-1, 1]
# Output: A_vec ∈ R^C

align_scalar = (normal_i · normal_j)  # (N, k, 1)
A_vec = MLP(align_scalar)             # (N, k, 1) → (N, k, C)
```

**Separation**:
```python
# Input: [distance, local_density]
# Output: S_vec ∈ R^C (negative for repulsion)

dist = ||p_i - p_j||                  # (N, k, 1)
density = 1 / mean(dist)              # (N, 1, 1)
sep_kernel = exp(-dist²/σ_sep²)       # (N, k, 1)

S_vec = -MLP([dist, density]) * sep_kernel  # Negative = repulsion
```

**Cohesion**:
```python
# Input: [distance, feature_similarity]
# Output: C_vec ∈ R^C (positive for attraction)

dist = ||p_i - p_j||                  # (N, k, 1)
feat_sim = normalize(x_i) · normalize(x_j)  # (N, k, 1)
coh_kernel = exp(-dist²/σ_coh²)       # (N, k, 1)

C_vec = MLP([dist, feat_sim]) * coh_kernel  # Positive = attraction
```

### 2. Learnable Parameters

**Per-Component Scales**:
```python
γ_A = exp(log_scale_a)  # Learnable, initialized at 1.0
γ_S = exp(log_scale_s)
γ_C = exp(log_scale_c)

B_vec = γ_A * w_a * A_vec + γ_S * w_s * S_vec + γ_C * w_c * C_vec
```

**RBP Lambda**:
```python
λ = exp(log_lambda)  # Learnable, initialized at 0.3
v' = gate ⊙ v + λ * B_vec
```

### 3. Gradient Flow

**Key advantage**: All three levels are differentiable!

```
Loss
 ↓
Output y_i
 ↓
Value v' ← (gate, λ)  # Level 3
 ↓
Attention att ← B_vec  # Level 2
 ↓
Neighbor scores ← (w_a, w_s, w_c, σ_sep, σ_coh)  # Level 1
```

The model learns:
- Which neighbors to attend to (Level 1)
- How to shape the kernel (Level 2)
- How to modulate values (Level 3)

All guided by Boids principles + task supervision!

---

## 📊 Expected Benefits

### 1. Stronger Influence
- **v1.0**: Scalar bias, easily buried
- **v2.0**: Vector kernel + value modulation = multi-level intervention

### 2. Interpretability
- **v1.0**: One scalar per pair `(i, j)` - hard to interpret
- **v2.0**:
  - Can visualize A_vec, S_vec, C_vec separately
  - Can analyze which channels are alignment-driven vs cohesion-driven
  - Can see how Boids affects different layers/heads

### 3. Ablation-Friendly
```python
# Test each component
Model(use_A=True,  use_S=False, use_C=False)  # Alignment only
Model(use_A=False, use_S=True,  use_C=False)  # Separation only
Model(use_A=False, use_S=False, use_C=True)   # Cohesion only

# Test each level
Model(level1=False, level2=True,  level3=True)   # No graph reweight
Model(level1=True,  level2=False, level3=True)   # No vector kernel
Model(level1=True,  level2=True,  level3=False)  # No value mod
```

### 4. Boids in True 3D Space
- **v1.0 (text)**: Boids forced into 1D token positions - unnatural
- **v2.0 (point cloud)**: Boids in native 3D Euclidean space - natural fit!

---

## 🎯 Hypothesis to Test

### H1: Stronger α deviation
**v1.0 (text)**: α ≈ 0.5 (bias too weak)
**v2.0 (3D)**: α < 0.4 or α > 0.6 (bias actually matters)

Because:
- Three levels of intervention (not just one)
- True 3D space (Boids' native domain)
- Vector kernels (channel-specific influence)

### H2: Component differentiation
- **Alignment**: Important for shape matching, semantic segmentation
- **Separation**: Important for noisy/dense point clouds, outlier rejection
- **Cohesion**: Important for object tracking, temporal coherence

Ablation will show which component matters for which task!

### H3: Scale-dependent behavior
- Small models: All levels matter
- Large models: Level 2 + Level 3 compensate for weaker Level 1
- Unlike v1.0 where large models buried everything

---

## 🚀 Implementation Status

### ✅ Completed
- [x] VectorKernelEncoder (Level 2)
  - Alignment, Separation, Cohesion → R^C vectors
  - Learnable per-component scales
- [x] ASCGraphReweight (Level 1)
  - Neighbor scoring network
  - Top-k selection based on ASC principles
- [x] ValuePathwayModulator (Level 3)
  - Multiplicative gating
  - Additive RBP with learnable λ
- [x] PointTransformerLayerASC
  - Full integration into Point Transformer
  - Backward compatibility (can disable ASC)

### 📝 Next Steps
1. Test on synthetic data (verify mechanisms)
2. Test on real dataset (MSRAction3D or ModelNet)
3. Ablation studies (A/S/C, Level 1/2/3)
4. Comparison with baseline Point Transformer
5. Analysis of learned parameters (α, λ, scales)

---

## 📚 Key Equations Summary

| Component | Equation | Shape |
|-----------|----------|-------|
| **Alignment** | `A_vec = MLP(cos(n_i, n_j))` | (N, k, C) |
| **Separation** | `S_vec = -MLP([d, ρ]) * exp(-d²/σ_sep²)` | (N, k, C) |
| **Cohesion** | `C_vec = MLP([d, sim]) * exp(-d²/σ_coh²)` | (N, k, C) |
| **Vector Kernel** | `B_vec = Σ_c w_c * γ_c * Component_c` | (N, k, C) |
| **Relation** | `rel = φ(x_i) - ψ(x_j) + δ_ij + B_vec` | (N, k, C) |
| **Attention** | `att = softmax(γ(rel))` | (N, k, C) |
| **Value** | `v' = σ(g(B_vec)) ⊙ v + λ * B_vec` | (N, k, C) |
| **Output** | `y_i = Σ_j att_ij ⊙ v'_ij` | (N, C) |

---

## 🎓 Contribution Summary

1. **Novel Architecture**: First work to apply Boids as **vector kernels** (not scalar bias) in 3D transformers
2. **Three-Level Intervention**: Graph + Kernel + Value (comprehensive, not superficial)
3. **Interpretability**: Can decompose A/S/C contributions channel-wise
4. **Domain Match**: Boids in native 3D space (vs forced into 1D text)
5. **Strong Baselines**: Built on SOTA Point Transformer, not toy models

**Differentiation from Point Transformer**:
- PT: Black-box δ_ij = MLP(p_i - p_j)
- ASC v2: White-box B_vec = explicit A/S/C decomposition

**Differentiation from ASCender v1.0**:
- v1.0: Scalar bias, single-level, text domain
- v2.0: Vector kernel, three-level, 3D domain

---

**Status**: 🟢 Ready for experimental validation
**Next Milestone**: Run on real point cloud dataset and measure α, component contributions
