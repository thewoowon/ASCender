# ASCender Emergent Structure Fixes

**Date**: 2025-11-01
**Principle**: "Local interactions → emergent global structures"

---

## 🎯 Core Philosophy

ASCender is NOT attempting to simulate Boids physics. Instead, it borrows the **principle** that simple local interaction rules can create complex emergent global patterns. The three components are:

- **Alignment**: Semantic similarity attraction (content-based clustering)
- **Cohesion**: Spatial neighborhood attraction (local grouping)
- **Separation**: Repulsion (boundary formation, prevent over-clustering)

---

## 🔴 Critical Issues Fixed

### 1. **Scale Destruction Pipeline** ✅ FIXED

**Problem**: Built bias in range [-10, 20], then clamped to [-2, 2] → 80% signal discarded!

```python
# BEFORE (ascender_bias.py:38-39)
clamp_min: float = -2.0
clamp_max: float = 2.0

# AFTER
clamp_min: float = -10.0  # ✅ Preserves full weight range
clamp_max: float = 10.0
```

**Impact**: Weights now matter! w_align=5, w_coh=4 actually produces [-9, 9] range instead of being crushed.

---

### 2. **Centering Destroyed Global Structure** ✅ FIXED

**Problem**: `bias = bias - bias.mean()` removed any global preferences, preventing emergent patterns!

```python
# BEFORE (ascender_bias.py:405)
bias = bias - bias.mean(dim=-1, keepdim=True)  # Always centered

# AFTER (ascender_bias.py:407-410)
if getattr(self.cfg, "enable_centering", False):
    bias = bias - bias.mean(dim=-1, keepdim=True)
# Default: enable_centering = False ✅
```

**Impact**: Global trends can now emerge (e.g., "attend more to recent tokens overall").

---

### 3. **Cohesion Kernel Too Narrow** ✅ FIXED

**Problem**: `sigma_coh: 3.0` only affected 6 tokens, but language dependencies span 50-200 tokens!

```python
# BEFORE (configs/ascender256.yaml)
sigma_coh: 3.0  # Effective range: ~6 tokens (2.3% of 256-token context)

# AFTER (configs/ascender256_emergent.yaml)
sigma_coh: 30.0  # Effective range: ~60 tokens (23% of context)
```

**Gaussian decay**:
- Old: influence drops to 1% at 10 tokens
- New: influence drops to 1% at 100 tokens

**Impact**: Cohesion now captures actual linguistic neighborhoods!

---

### 4. **Separation Disabled** ✅ FIXED

**Problem**: Without repulsion, you only had attraction → no structure, just clustering!

```python
# BEFORE
use_separation: false
w_sep: 0.00

# AFTER
use_separation: true   # ✅ ESSENTIAL for emergent structure
w_sep: 1.5            # Half of cohesion strength (maintains balance)
```

**Impact**: Creates boundaries between clusters, prevents over-aggregation.

---

### 5. **ALiBi Mixing Interference** ✅ FIXED

**Problem**: ALiBi's linear recency bias conflicts with Gaussian spatial structure!

```python
# BEFORE
use_alibi_mix: true
alpha_start: 0.20  # 80% ALiBi initially!

# AFTER
use_alibi_mix: false  # ✅ Let ASCender emergent patterns develop naturally
```

**Conflict**:
- ALiBi gradient: `∂bias/∂dist = -m` (constant negative)
- Cohesion gradient: `∂bias/∂dist = positive @ dist < σ, negative @ dist > σ`

They fight each other!

---

### 6. **Auto-Calibration Suppressed Learning** ✅ SOFTENED

**Problem**: `target_ratio: 0.30` kept bias at 30% of attention scores → bias is a "correction", not a "driver"!

```python
# BEFORE
target_ratio: 0.30
calibrate_step_clamp_lo: 0.90  # Aggressive adjustments
calibrate_step_clamp_hi: 1.12

# AFTER
target_ratio: 0.50             # ✅ Allow 50% influence
calibrate_step_clamp_lo: 0.97  # ✅ Gentler (was changing γ by ±10%)
calibrate_step_clamp_hi: 1.03  # ✅ Now only ±3%

# Code change (ascender_bias.py:472)
gentle_adj = (adj_h - 1.0) * 0.1 + 1.0  # ✅ 10x slower adjustment
```

**Impact**: Auto-calibration guides rather than controls. Gradients can actually learn!

---

### 7. **Gate Range Too Narrow** ✅ FIXED

**Problem**: Gate started at floor with nowhere to go down, ceiling too low!

```python
# BEFORE
gate_init: -0.6    # sigmoid(-0.6) ≈ 0.35 (at floor!)
gate_floor: 0.35
gate_ceiling: 0.65

# AFTER
gate_init: 0.0     # sigmoid(0) = 0.5 (centered, room to learn both ways)
gate_floor: 0.15   # ✅ Can suppress bias if needed
gate_ceiling: 0.85 # ✅ Can amplify bias if needed
```

**Impact**: Gate can now learn the full range of control.

---

### 8. **Band-Pass Too Restrictive** ✅ FIXED

```python
# BEFORE
band_max: 48  # Only up to 48 tokens

# AFTER
band_max: 128  # ✅ Allow full-context interactions
```

**Impact**: Long-range dependencies not artificially cut off.

---

## 📊 Comparison: Before vs After

| Component | Before | After | Impact |
|-----------|--------|-------|--------|
| **Clamp Range** | [-2, 2] | [-10, 10] | ✅ 80% signal preserved |
| **Centering** | Always ON | OFF by default | ✅ Global structure emerges |
| **Cohesion Range** | ~6 tokens | ~60 tokens | ✅ 10x wider neighborhoods |
| **Separation** | Disabled | Enabled (w=1.5) | ✅ Structure formation |
| **ALiBi Mix** | 60% initially | Disabled | ✅ No interference |
| **Auto-Calib Target** | 0.30 | 0.50 | ✅ Stronger influence |
| **Auto-Calib Step** | ±10% | ±3% | ✅ 3x gentler |
| **Gate Range** | [0.35, 0.65] | [0.15, 0.85] | ✅ 2.3x wider control |
| **Band Max** | 48 tokens | 128 tokens | ✅ 2.7x longer range |

---

## 🚀 How to Use

### Option 1: Moderate Config (Balanced)
```bash
python src/train.py --config configs/ascender256_moderate.yaml
```

**Profile**: All three components, moderate weights, good starting point.

### Option 2: Emergent Config (Optimized)
```bash
python src/train.py --config configs/ascender256_emergent.yaml
```

**Profile**: Fully optimized for emergent structure formation. All fixes applied.

### Option 3: Original (Now Fixed)
```bash
python src/train.py --config configs/ascender256.yaml
```

**Note**: Original file may need manual updates. Use `emergent` config for best results.

---

## 🔬 Expected Behavior Changes

### Before Fixes:
- ❌ Bias had minimal effect (suppressed by clamping, centering, auto-calib)
- ❌ Only 2/3 of Boids components active (no separation)
- ❌ Too local (6-token neighborhoods)
- ❌ Competing with ALiBi
- ❌ Gradient starvation (hard limits everywhere)

### After Fixes:
- ✅ Bias can form emergent global patterns
- ✅ All three components create structure
- ✅ Linguistically meaningful neighborhoods (60+ tokens)
- ✅ Clean signal without ALiBi interference
- ✅ Smooth gradient flow for natural learning

---

## 📈 Monitoring Emergent Structure

Watch these metrics during training:

```python
# In train.py logs:
delta_p_mean   # Should be 0.05-0.15 (stronger influence now)
bias_ratio     # Should approach 0.50 (not 0.30!)
gamma (γ)      # Should vary per-head (0.5 - 8.0 range)
gate (σ)       # Should learn per-layer (0.15 - 0.85 range)
```

### Healthy Emergent Patterns:
- **Alignment**: Similar tokens cluster (attention to synonyms, related concepts)
- **Cohesion**: Local smoothing (nearby tokens support each other)
- **Separation**: Boundaries (topic shifts, clause breaks get spacing)

### Visualize:
Check `logs/heatmaps/bias_epoch_*.png` for structure formation over time.

---

## 🧪 Ablation Study Recommendations

To validate that emergent structure is working:

1. **Baseline**: No ASCender (`use_ascender: false`)
2. **Alignment Only**: `use_separation: false, use_cohesion: false`
3. **Alignment + Cohesion**: `use_separation: false` (old config)
4. **Full Emergent**: All three components (new config)

Expected: (4) > (3) > (2) > (1) if emergent structure is beneficial.

---

## 📚 Key Code Changes

### ascender_bias.py
- Line 38-39: Clamp limits widened
- Line 42: `enable_centering` flag added
- Line 47: `band_max` default increased
- Line 57-59: Gate defaults changed
- Line 409-410: Centering now optional
- Line 472: Auto-calibration gentler (0.1x step)
- Line 481: Gate adjustment gentler (k=0.02)

### configs/ascender256_moderate.yaml
- All three components enabled
- Weights: align=3.0, sep=1.5, coh=4.0
- sigma_coh: 20.0 (was 3.0)
- clamp: [-10, 10] (was [-2, 2])
- ALiBi disabled
- Auto-calib softened

### configs/ascender256_emergent.yaml
- NEW FILE: Fully optimized configuration
- Extensive inline documentation
- All recommendations applied

---

## 🎓 Theoretical Foundation

### Why These Fixes Enable Emergence:

1. **No Centering** → Global biases can exist → Patterns span entire context
2. **Wide Cohesion** → Multi-token neighborhoods → Linguistic units (phrases, clauses)
3. **Active Separation** → Boundaries form → Structure rather than homogeneity
4. **High Clamp** → Full signal preserved → Interactions have real strength
5. **No ALiBi** → Clean signal → Emergent patterns not masked
6. **Gentle Calibration** → Natural learning → Gradients find optimal structure

### Emergence vs Control:

**Old approach**: "Control bias to 30% of attention" → Bias is corrective
**New approach**: "Let bias learn its role naturally" → Bias discovers emergent patterns

The model now **learns** what structures are useful, rather than being forced into a predefined 30% contribution.

---

## ✅ Validation Checklist

Before running experiments:

- [x] Code changes applied to `src/models/ascender_bias.py`
- [x] Config updated to `ascender256_emergent.yaml`
- [x] All three components enabled (`use_separation: true`)
- [x] Centering disabled (`enable_centering: false`)
- [x] Clamp limits match weight magnitudes ([-10, 10])
- [x] Cohesion kernel widened (`sigma_coh >= 20`)
- [x] ALiBi mixing disabled (`use_alibi_mix: false`)
- [x] Auto-calibration softened (`target_ratio >= 0.40`)
- [x] Gate range expanded (`gate_ceiling >= 0.85`)

---

## 🔮 Expected Results

### If Emergent Structure is Beneficial:
- Lower perplexity than baseline
- Visible structure in bias heatmaps
- Per-head γ diversity (heads specialize)
- delta_p stabilizes around 0.08-0.12

### If Still No Improvement:
Possible causes:
1. Language modeling may not benefit from spatial structure (worth testing!)
2. Causal masking breaks symmetry assumption (architectural limitation)
3. Need architectural modifications (Residual Bias Path from `architectural_mods.py`)

---

## 📞 Next Steps

1. **Run experiments** with `ascender256_emergent.yaml`
2. **Compare** to baseline and old config
3. **Visualize** bias heatmaps for emergent patterns
4. **Monitor** per-head diversity (γ, σ should vary)
5. **Consider** architectural modifications if needed

---

**The code is now ready to discover emergent structures!** 🚀
