# ASCender for Dynamic Point Clouds

**Boids-Inspired Spatial Bias for Dynamic Point Cloud Transformers**

## 🎯 Project Goal

Extend ASCender (Alignment-Separation-Cohesion bias) from 1D text sequences to **dynamic 3D point clouds**, leveraging Boids' original temporal dynamics.

## 🆚 Differentiation from Existing Work

### Existing Point Transformer (2021)
- **Position Encoding**: MLP(Δxyz) - learned, black box
- **Application**: Mainly static point clouds
- **Interpretability**: None - cannot explain why it works

### ASCender Point Cloud (This Work)
- **Position Encoding**: Explicit Boids 3-component decomposition
  - **Alignment**: Points with similar motion/direction
  - **Separation**: Suppresses noisy/outlier points
  - **Cohesion**: Maintains local spatial structure over time
- **Application**: **Dynamic** point clouds (temporal coherence)
- **Interpretability**: Each component's contribution is measurable via ablation
- **Architecture**: Residual Bias Path (RBP) - automatic mixing of learned + structural bias

## 🔬 Key Innovations

1. **Temporal Boids Dynamics**: Extend static spatial bias to temporal domain
   - Alignment tracks motion coherence across frames
   - Cohesion maintains object persistence over time

2. **3D Euclidean Distance**: Replace token position difference with true spatial distance
   ```python
   # Text (1D): |i - j|
   # Point Cloud (3D): ||xyz_i - xyz_j||_2
   ```

3. **Normal-based Alignment**: Use surface normals for semantic coherence
   ```python
   align_bias = (normals_i · normals_j) / temperature
   ```

4. **Per-head Adaptive Scales**: Each attention head learns different spatial scales
   - Head 0: Local details (σ=0.1m)
   - Head 4: Global structure (σ=1.0m)

## 📊 Target Datasets

### Primary: MSRAction3D
- **Size**: 567 sequences, 20 action classes
- **Format**: Depth sequences → point clouds
- **Task**: Action recognition from dynamic point clouds

### Secondary: NTU RGB+D 120
- **Size**: 114,480 samples, 120 action classes
- **Format**: RGB-D + skeleton → point clouds
- **Task**: Large-scale action recognition

### Tertiary: KITTI Tracking
- **Format**: LiDAR sequences
- **Task**: Object tracking in autonomous driving

## 🏗️ Architecture

```
Dynamic Point Cloud Transformer with ASCender

Input: Sequence of point clouds [P_1, P_2, ..., P_T]
       P_t = {(x,y,z)_i, normal_i}

Encoder (per frame):
  ├─ Point embedding
  ├─ Multi-head Self-Attention
  │   ├─ Learned Attention (Q·K^T)
  │   └─ ASCender Bias:
  │       ├─ Alignment: sim(normal_i, normal_j)
  │       ├─ Separation: -exp(-||xyz_i-xyz_j||²/σ_sep²)
  │       └─ Cohesion: exp(-||xyz_i-xyz_j||²/σ_coh²)
  └─ Residual Bias Path: α·learned + (1-α)·bias

Temporal Aggregation:
  └─ Sequence pooling / Temporal attention

Output: Action class probabilities
```

## 🚀 Quick Start Plan

### Phase 1: Research & Analysis (Day 1-2) ✅
- [x] Survey Point Transformer implementations
- [x] Identify dynamic point cloud datasets
- [ ] Analyze Point Transformer's position encoding mechanism
- [ ] Document exact differentiation points

### Phase 2: Minimal Adaptation (Day 3-5)
- [ ] Create point_cloud_ascender_bias.py
  - Replace token distance with Euclidean distance
  - Add normal-based alignment option
- [ ] Adapt MultiHeadAttention for 3D coordinates
- [ ] Write data loader for MSRAction3D

### Phase 3: Quick PoC (Day 6-10)
- [ ] Train baseline Point Transformer on MSRAction3D
- [ ] Train ASCender version with fixed bias (α=0.5)
- [ ] Train RBP version with learnable α
- [ ] Compare accuracy + analyze α values

### Phase 4: Analysis & Decision (Day 11-12)
- [ ] If α ≈ 0.5 again → bias too weak, increase strength
- [ ] If α shows preference → ablate A/S/C components
- [ ] If results promising → full research project
- [ ] If results underwhelming → pivot or stop

## 📁 Project Structure

```
point-cloud-ascender/
├── src/
│   ├── models/
│   │   ├── point_transformer.py       # Baseline
│   │   ├── point_ascender_bias.py     # 3D Boids bias
│   │   └── point_multihead_attn.py    # MHA with spatial bias
│   ├── data/
│   │   ├── msraction3d_loader.py
│   │   └── point_cloud_utils.py
│   └── utils/
│       ├── spatial_kernels.py         # 3D Gaussian kernels
│       └── normal_estimation.py       # Surface normal computation
├── configs/
│   ├── baseline_msraction3d.yaml
│   └── ascender_msraction3d.yaml
├── experiments/
│   └── quick_poc.py
├── tests/
│   └── test_spatial_bias.py
└── README.md
```

## 🎓 Expected Contributions

1. **Novel Application**: First application of Boids-inspired bias to dynamic point clouds
2. **Interpretability**: Explicit component analysis (what does each Boids rule contribute?)
3. **Temporal Extension**: Extend spatial bias to temporal coherence
4. **Empirical Validation**: Does structural bias help in 3D domain more than 1D text?

## 📌 Key Questions to Answer

1. **Does α move away from 0.5 in 3D?**
   - Hypothesis: Yes, because 3D space has true physical meaning

2. **Which component matters most?**
   - Alignment for motion tracking?
   - Cohesion for object persistence?
   - Separation for noise robustness?

3. **Does it outperform learned position encoding?**
   - Not guaranteed, but interpretability has value even if performance ties

## 📚 References

- **Point Transformer**: Zhao et al., ICCV 2021
- **MSRAction3D**: Li et al., CVPR 2010
- **NTU RGB+D**: Shahroudy et al., CVPR 2016
- **Boids**: Reynolds, SIGGRAPH 1987
- **ASCender (original)**: This project's NLP baseline

---

**Status**: Phase 1 in progress
**Next**: Analyze Point Transformer position encoding mechanism
