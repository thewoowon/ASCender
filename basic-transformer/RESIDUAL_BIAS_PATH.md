# Residual Bias Path Architecture

**Date**: 2025-11-01
**Status**: ✅ Fully Implemented
**Config**: `configs/ascender256_residual.yaml`

---

## 🎯 The Fundamental Problem with Additive Bias

### The Softmax Trap

When you add bias directly before softmax, something subtle and problematic happens:

```python
# Standard additive bias (OLD approach)
attn = softmax(Q·K/√d + BIAS)
     = exp(Q·K/√d + BIAS) / Σ exp(Q·K/√d + BIAS)
     = exp(Q·K/√d) * exp(BIAS) / Σ [exp(Q·K/√d) * exp(BIAS)]
```

**Problem**: Adding in log-space becomes multiplication in probability space!

- Bias = +2 → multiply probability by exp(2) = **7.4x**
- Bias = -2 → multiply probability by exp(-2) = **0.14x**

This creates **three critical issues**:

### Issue 1: Non-Linear, Context-Dependent Effects

You can't predict what a given bias value will do without knowing the entire score distribution:

```python
# Example: Bias = +1.0 applied to different contexts

# Context A: Uniform scores [0, 0, 0, 0]
# → Bias makes one position 2.7x more likely (noticeable)

# Context B: One dominant score [10, 0, 0, 0]
# → Bias has almost no effect (softmax already saturated)

# Context C: Two competing scores [5, 5, 0, 0]
# → Bias determines the winner (huge impact)
```

**Same bias value, wildly different effects!**

### Issue 2: Gradient Suppression

Softmax normalization **suppresses gradients** for bias when attention is already confident:

```python
# When attention is confident (peaked distribution)
attn = [0.9, 0.05, 0.03, 0.02]

# Gradient of loss w.r.t. bias is tiny!
# Model can't learn what bias should be
```

The more the model relies on learned QK attention, the less it can learn to use bias.

### Issue 3: Interference, Not Cooperation

Bias and learned attention compete rather than cooperate:

```python
# Learned attention says: "attend to position 2" (Q·K gives high score)
# Bias says: "attend to position 3" (structural prior)

# After softmax: fight for probability mass
# Result: degraded performance on BOTH signals
```

---

## ✨ The Residual Bias Path Solution

### Architecture

Instead of forcing bias and learned attention to fight in the same softmax:

```python
# Residual Bias Path (NEW approach)

# Path 1: Pure learned attention
attn_normal = softmax(Q·K / √d) @ V

# Path 2: Bias-influenced attention
attn_biased = softmax(Q·K / √d + BIAS) @ V

# Learnable per-head mixing
α = sigmoid(α_logit)  # Per-head learned weight ∈ [0,1]
output = α * attn_normal + (1-α) * attn_biased
```

### Why This Works

1. **No Competition**: Each path computes independently
2. **Natural Gradient Flow**: Both paths have clean gradients
3. **Adaptive Mixing**: Model learns when to trust bias vs. learned attention
4. **Per-Head Specialization**: Each head can learn different α

---

## 🔬 Mathematical Analysis

### Standard Additive Bias

```
Output = softmax(S + B) @ V
where S = Q·K/√d, B = bias

∂L/∂B depends on:
- Current attention distribution (non-linear)
- Value vectors (indirect)
- Softmax saturation (can be near-zero)
```

**Problem**: Gradient magnitude varies wildly with attention state.

### Residual Path

```
Output = α · softmax(S) @ V + (1-α) · softmax(S + B) @ V

∂L/∂B = (1-α) · ∂[softmax(S+B) @ V]/∂B
```

**Benefits**:
- Gradient scales with (1-α), not with attention saturation
- α is learned, providing meta-gradient signal
- Bias always has a path to influence output (even when α→1)

---

## 🧪 Implementation Details

### Code Location

**File**: `src/models/transformer.py`

**Key Components**:

1. **Parameter** (line 177):
   ```python
   self.alpha_logit = nn.Parameter(torch.zeros(n_heads))  # Per-head mixing
   self.enable_residual_path: bool = False  # Feature flag
   ```

2. **Forward Pass** (lines 407-444):
   ```python
   if self.enable_residual_path:
       # Path 1: Normal attention
       attn_normal = softmax(scores) @ V

       # Path 2: Biased attention
       attn_biased = softmax(scores + bias) @ V

       # Mix
       α = sigmoid(alpha_logit)
       output = α * attn_normal + (1-α) * attn_biased
   else:
       # Standard additive bias
       output = softmax(scores + bias) @ V
   ```

3. **Config** (`TransformerConfig`, line 675):
   ```python
   enable_residual_path: bool = False  # Set True to activate
   ```

### Training Considerations

1. **α Initialization**: `alpha_logit = 0` → `α = sigmoid(0) = 0.5`
   - Starts with equal mixing
   - Can learn to emphasize either path

2. **Computational Cost**:
   - ~2x the attention computation (two softmax operations)
   - No change to memory complexity
   - Worth it for the architectural benefits!

3. **Monitoring**:
   ```python
   # Access learned α values per head
   model.decoder.layers[0].self_attn._alpha_effective  # (H,) tensor
   ```

---

## 📊 Expected Behavior

### Healthy Learning Patterns

**Early Training** (epochs 1-2):
- α should vary per head: `[0.3, 0.6, 0.4, 0.7, ...]`
- Some heads favor bias (α<0.5), others favor learned (α>0.5)

**Mid Training** (epoch 3+):
- α should stabilize but remain diverse
- Low-layer heads (L0) might use more bias (α~0.3-0.4)
- High-layer heads (L2) might use less bias (α~0.6-0.7)

### Troubleshooting

| Observation | Likely Cause | Fix |
|-------------|--------------|-----|
| All α → 1.0 | Bias not helpful | Check bias weights, kernel widths |
| All α → 0.0 | Bias overwhelming | Reduce target_ratio, check clamps |
| α not changing | Learning rate too low | Increase `lr_asc` |
| NaN in α | Numerical instability | Check bias clamps, reduce weights |

---

## 🚀 How to Use

### Option 1: Full Residual + Emergent Config (Recommended)

```bash
python src/train.py --config configs/ascender256_residual.yaml
```

This config includes:
- ✅ Residual Bias Path enabled
- ✅ All emergent structure fixes
- ✅ Optimized hyperparameters

### Option 2: Add to Existing Config

```yaml
model:
  # ... your existing config ...

  # Add this line:
  enable_residual_path: true  # ← ENABLE RESIDUAL PATH
```

### Option 3: Programmatic

```python
from src.models.transformer import Transformer, TransformerConfig

cfg = TransformerConfig(
    src_vocab_size=30000,
    tgt_vocab_size=30000,
    # ... other params ...
    use_ascender=True,
    enable_residual_path=True,  # ← Enable here
)

model = Transformer(cfg)
```

---

## 📈 Comparison: Standard vs. Residual

### Standard Additive Bias

**Pros**:
- Simple
- No extra computation
- Direct control

**Cons**:
- ❌ Softmax saturation kills gradients
- ❌ Non-linear effects hard to predict
- ❌ Bias and learned attention fight
- ❌ Context-dependent behavior

### Residual Bias Path

**Pros**:
- ✅ Clean gradient flow for both paths
- ✅ Model learns when to use bias
- ✅ Predictable linear mixing
- ✅ Per-head specialization
- ✅ No gradient suppression

**Cons**:
- ~2x attention computation
- One extra parameter per head (α)

**Verdict**: The architectural benefits far outweigh the computational cost!

---

## 🔍 Theoretical Justification

### Why Linear Mixing Works Better

In residual path, output is a **convex combination**:

```
output = α · O_normal + (1-α) · O_biased
where α ∈ [0,1]
```

This has nice properties:
1. **Boundedness**: Output is bounded by the two paths
2. **Interpolation**: Smoothly blends between extremes
3. **Reversibility**: α can be adjusted without retraining everything
4. **Interpretability**: α directly shows bias influence

In standard bias, output is:

```
output = softmax(S + B) @ V
```

This is **non-convex**, **non-linear**, and **context-dependent**.

### Connection to Ensemble Methods

Residual path is similar to ensemble learning:
- Path 1: Expert on learned patterns
- Path 2: Expert on structural patterns
- α: Learned ensemble weight

This is why it works better than forcing both signals through one softmax!

---

## 🧬 Integration with ASCender

### Compatibility

Residual Bias Path is **fully compatible** with all ASCender components:

- ✅ Alignment bias
- ✅ Cohesion bias
- ✅ Separation bias
- ✅ Auto-calibration
- ✅ Gate mechanism
- ✅ Per-head scaling
- ✅ All config options

### Recommended Stack

For maximum effectiveness, combine:

1. **Residual Path** → Solves softmax problem
2. **All three Boids components** → Creates structure
3. **Wide cohesion kernel** (σ=30) → Linguistic neighborhoods
4. **No centering** → Preserves global patterns
5. **No ALiBi** → Clean signal
6. **Gentle auto-calibration** → Natural learning

This is exactly what `ascender256_residual.yaml` provides!

---

## 📚 Related Architectures

### Comparison to Other Approaches

| Approach | Mechanism | Pros | Cons |
|----------|-----------|------|------|
| **Standard Bias** | S + B → softmax | Simple | Softmax interference |
| **Residual Path** | α·softmax(S) + (1-α)·softmax(S+B) | Clean gradients | 2x compute |
| **Multiplicative** | softmax(S) ⊙ B | No normalization issue | Hard to control |
| **Pre-softmax gate** | softmax(S + σ·B) | One softmax | Still has interference |
| **Post-softmax** | softmax(S) + B → renorm | Additive | Extra normalization |

**Residual Path is the cleanest solution to the softmax problem.**

---

## 🎓 When to Use

### Use Residual Path When:

- ✅ You have structural bias (like ASCender)
- ✅ Bias competes with learned attention
- ✅ Gradients for bias are weak
- ✅ You want interpretable α weights
- ✅ Per-head specialization is desired

### Stick with Standard Bias When:

- ⚠️ Bias is very weak (might not be worth 2x compute)
- ⚠️ You only have positional encoding (not learned bias)
- ⚠️ Computational budget is extremely tight
- ⚠️ Bias should always dominate (no need for mixing)

**For ASCender emergent structures: Residual Path is strongly recommended!**

---

## 🔮 Future Directions

### Potential Enhancements

1. **Hierarchical α**: Learn different α per layer
2. **Dynamic α**: Condition α on input (attention-dependent mixing)
3. **Multi-path**: Three paths (normal, cohesion, alignment)
4. **Soft gating**: Replace sigmoid with learned gate network
5. **Curriculum**: Start with high α (learned), gradually introduce bias

### Research Questions

- What α patterns emerge for different linguistic phenomena?
- Can α be used as an interpretability tool?
- Does α transfer across domains?
- What's the optimal number of paths?

---

## ✅ Summary

**Problem**: Additive bias before softmax creates non-linear, context-dependent effects that suppress gradients.

**Solution**: Residual Bias Path computes two separate attention paths and mixes them linearly with learned per-head weights.

**Benefits**:
- Clean gradient flow
- Predictable behavior
- Per-head specialization
- No gradient suppression
- Model learns when to use bias

**Cost**: ~2x attention computation (worth it!)

**Status**: ✅ Fully implemented and ready to use

**Config**: `configs/ascender256_residual.yaml`

---

**This is the architectural fix that makes ASCender emergent structures actually work!** 🚀
