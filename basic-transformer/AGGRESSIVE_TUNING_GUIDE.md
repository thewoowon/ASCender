# ASCender Aggressive Tuning Guide
## Making Structural Bias Have Measurable Impact

---

## 🎯 Problem: "Bias Has No Effect"

If your A/B tests show **identical loss** with bias ON vs OFF, the bias is too weak. This guide provides progressively aggressive strategies to ensure a **clear, measurable effect**.

---

## 📊 Three Levels of Aggression

### Level 1: Moderate Aggressive (Recommended Starting Point)
**File**: `configs/ascender_moderate_aggressive.yaml`

**Key Parameters:**
- `std_match_ratio_override: 0.50` - Bias is 50% of attention scores std
- `w_align: 0.15, w_coh: 0.20` - Stronger component weights
- All three components enabled (Alignment + Cohesion + Separation)
- Per-head gamma and gate
- Auto-calibration DISABLED (let it run free)
- Higher ASC learning rate (0.0008)

**Expected Effect:**
- Top-1 prediction changes: 5-15%
- NLL difference: 0.05-0.20
- **Clear signal** without destroying learning

**Run:**
```bash
python -m src.train --config configs/ascender_moderate_aggressive.yaml
```

---

### Level 2: Very Aggressive (Maximum Signal)
**File**: `configs/ascender_very_aggressive.yaml`

**Key Parameters:**
- `std_match_ratio_override: 0.90` - Bias is 90% of scores std (very strong!)
- `w_align: 0.35, w_coh: 0.40` - Nearly doubled weights
- Gate very open (`gate_init: -0.5, ceiling: 0.80`)
- High gamma cap (8.0)
- Sharp alignment (`temperature: 0.5`)
- 3x ASC learning rate

**Expected Effect:**
- Top-1 prediction changes: 15-30%
- NLL difference: 0.20-0.50
- **Obvious signal** - will likely hurt performance but proves bias works

**Run:**
```bash
python -m src.train --config configs/ascender_very_aggressive.yaml
```

⚠️ **Warning**: This WILL likely increase loss. Use for diagnostic purposes only.

---

### Level 3: Baseline (For Comparison)
**File**: `configs/baseline.yaml`

Pure transformer with NO bias. Use this to establish ground truth.

```bash
python -m src.train --config configs/baseline.yaml
```

---

## 🔬 Measuring Bias Effect

Use the diagnostic script to quantify exactly how much bias affects the model:

```bash
python scripts/measure_bias_effect.py --config configs/ascender_moderate_aggressive.yaml
```

**Output Metrics:**
- **Logit Δ**: How much output logits change
- **Top-1 disagreement**: % of predictions that flip
- **KL divergence**: Distribution shift caused by bias
- **NLL improvement**: Whether bias helps (+) or hurts (-)
- **Attention Δ**: How much attention patterns change

**Interpretation:**
```
Top-1 disagreement < 1%:   ❌ NO EFFECT - use very aggressive config
Top-1 disagreement 5-15%:  ✅ GOOD - measurable effect
Top-1 disagreement > 30%:  ⚠️  TOO STRONG - might hurt learning
```

---

## 🏗️ Architectural Modifications (Beyond Hyperparameters)

If even very aggressive configs don't work, try **fundamental architectural changes**.

**File**: `src/models/architectural_mods.py`

### Option 1: Residual Bias Path
Instead of adding bias to scores, create **two parallel attention paths** and mix outputs:
- Path 1: Normal attention
- Path 2: Biased attention
- Final: `α * normal + (1-α) * biased` where α is learned

**Advantage**: Bias can never completely overwhelm learned patterns.

### Option 2: Gated Bias Integration
Learn a **per-position gate** from query features:
- Gate: `g(q) = σ(MLP(q))`
- Bias: `g(q) * structural_bias`

**Advantage**: Model decides when to trust bias vs ignore it.

### Option 3: Multi-Scale Bias
Apply different bias components at different scales:
- Local bias (σ=2): Nearby tokens
- Mid-range bias (σ=8): Moderate distance
- Global bias: All positions

Each scale has independent learned strength.

### Option 4: Bias-Conditioned Value
Let bias affect not just WHERE to attend (scores), but also WHAT to retrieve (values):
- `V' = V * (1 + ε * tanh(aggregated_bias))`

**Advantage**: Two-way influence creates stronger effect.

### Option 5: Hierarchical Bias
Apply bias in stages:
1. Coarse bias (positional) shapes rough pattern
2. Fine bias (content-based) refines based on coarse

---

## 📈 Tuning Strategy: Start Aggressive, Then Dial Back

### Phase 1: Prove Bias Can Work
1. Run **very aggressive** config
2. Confirm top-1 disagreement > 15%
3. Observe attention heatmaps (should show clear structure)

**Goal**: Establish that bias mechanism works, even if performance suffers.

### Phase 2: Find Sweet Spot
1. Start with **moderate aggressive** config
2. Monitor A/B diagnostics during training
3. Gradually reduce strength until:
   - NLL(bias ON) ≤ NLL(bias OFF)
   - Top-1 disagreement still > 5%

**Goal**: Maximum bias effect while maintaining performance.

### Phase 3: Optimize for Performance
1. Enable auto-calibration with low target (0.15-0.25)
2. Use dynamic schedules (alpha_schedule, tau_schedule if helpful)
3. Fine-tune per-layer strengths independently

**Goal**: Best final performance with bias providing genuine improvement.

---

## 🔍 Debugging Checklist

If bias STILL has no effect even with very aggressive config:

### Check 1: Is Bias Being Generated?
```python
# During training, add this print:
if biaser is not None:
    print(f"Bias stats: mean={bias.mean():.3f}, std={bias.std():.3f}, max={bias.abs().max():.3f}")
```
Expected: std > 0.5, max > 2.0 for aggressive configs

### Check 2: Is std_match_ratio Applied?
```python
# In transformer.py, verify:
print(f"Layer 0 std_match_ratio: {model.decoder.layers[0].self_attn.std_match_ratio}")
```
Should print: 0.50 (moderate) or 0.90 (very aggressive)

### Check 3: Is Bias Actually Added to Scores?
```python
# In MultiHeadAttention.forward(), add:
print(f"Scores std: {scores.std():.3f}, Bias std: {runtime_bias.std():.3f}")
```
Ratio of bias std / scores std should match your target.

### Check 4: Is Auto-Calibration Killing It?
```python
# In ascender_bias.py forward(), check:
if self.cfg.use_auto_calibrate:
    print(f"[CALIB] Dampening bias: {ratio_now:.3f} -> target {self.cfg.target_ratio:.3f}")
```
If auto-calibrate is ON, it might be suppressing aggressive bias.

### Check 5: Is Hard Limiter Clamping?
```python
# In ascender_bias.py, check:
if ratio_now > float(self.cfg.hard_max_ratio):
    print(f"⚠️  HARD LIMIT HIT: {ratio_now:.3f} > {self.cfg.hard_max_ratio}")
```
If you see this, increase `hard_max_ratio` in config.

---

## 🎛️ Quick Parameter Reference

### Component Weights (pre-gamma)
- **Conservative**: w_align=0.03, w_coh=0.08
- **Moderate**: w_align=0.15, w_coh=0.20
- **Aggressive**: w_align=0.35, w_coh=0.40

### std_match_ratio (final magnitude multiplier)
- **Conservative**: 0.10-0.20
- **Moderate**: 0.30-0.60
- **Aggressive**: 0.70-0.95
- **Very Aggressive**: 1.0-1.5

### Gate Settings
- **Conservative**: init=-3.0, ceiling=0.35
- **Moderate**: init=-1.5, ceiling=0.60
- **Aggressive**: init=-0.5, ceiling=0.80

### Gamma Cap
- **Conservative**: 2.5
- **Moderate**: 4.0
- **Aggressive**: 8.0

---

## 🧪 Experiment Template

```bash
# Baseline (no bias)
python -m src.train --config configs/baseline.yaml
# Note the final loss: e.g., 3.50

# Moderate aggressive (first try)
python -m src.train --config configs/ascender_moderate_aggressive.yaml
# Expected: 3.40-3.60 (within 3% of baseline)

# Measure effect
python scripts/measure_bias_effect.py --config configs/ascender_moderate_aggressive.yaml
# Look for Top-1 disagreement > 5%

# If no effect: very aggressive
python -m src.train --config configs/ascender_very_aggressive.yaml
# Expected: 3.80-4.20 (worse than baseline, but proves bias works)

# If still no effect: check for bugs (see debugging checklist)
```

---

## 💡 Key Insights

### Why Bias Might Have No Effect

1. **Auto-calibration is too aggressive**: Target ratio is too low (< 0.10)
2. **Hard limiter is clamping**: `hard_max_ratio` is too conservative
3. **std_match_ratio not applied**: Check that override actually reaches MHA
4. **Component weights too small**: Need to be > 0.10 for meaningful signal
5. **Gate is closed**: `gate_init` too negative or `gate_ceiling` too low
6. **Bias center-normalized to zero**: Mean is removed, reducing impact

### The Multiplicative Cascade

Effective bias magnitude is:
```
effective_bias = component_weight × gamma × gate × std_match_ratio × scores_std
```

Example (moderate aggressive):
```
0.20 × 1.0 × 0.50 × 0.50 × 2.0 = 0.10 (10% of scores magnitude)
```

If ANY factor is too small, effective bias → 0.

---

## 🚀 Next Steps

1. **Run baseline** to establish ground truth
2. **Run moderate aggressive** and check A/B diagnostics
3. **Use diagnostic script** to quantify effect
4. **If no effect**: Run very aggressive to prove mechanism works
5. **If still no effect**: Check debugging checklist
6. **Once effect confirmed**: Dial back to find optimal strength
7. **If hyperparameters maxed out**: Try architectural modifications

---

## 📚 Files Summary

| File | Purpose |
|------|---------|
| `configs/ascender_safe.yaml` | Conservative (might have no effect) |
| `configs/ascender_moderate_aggressive.yaml` | **Start here** - clear signal |
| `configs/ascender_very_aggressive.yaml` | Maximum diagnostic signal |
| `configs/baseline.yaml` | No bias comparison |
| `scripts/measure_bias_effect.py` | Quantify bias contribution |
| `src/models/architectural_mods.py` | Fundamental architecture changes |
| `FIXES_EXPLAINED.md` | Why previous config failed |
| `AGGRESSIVE_TUNING_GUIDE.md` | This document |

---

Good luck! The moderate aggressive config should give you a clear, measurable effect. If not, escalate to very aggressive or architectural mods. 🔥
