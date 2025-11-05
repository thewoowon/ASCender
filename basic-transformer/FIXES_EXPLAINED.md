# ASCender Bias Fixes - Why Performance Was Poor

## 🔴 The Problem

Your structural bias was **2x worse than baseline** (loss ~9.89 vs ~4.04). Here's why:

### Root Cause: Bias Magnitude Too Large

The bias was **overwhelming the learned attention patterns** instead of gently guiding them.

```python
# ❌ BEFORE (transformer.py:650)
self.decoder.layers[0].self_attn.std_match_ratio = 1.6  # Bias is 1.6x scores std!
```

This made the bias **larger than the attention scores themselves**, so the model couldn't learn - it was just following your hand-crafted rules.

## 🔧 What I Fixed

### 1. **Reduced std_match_ratio** (transformer.py:652-662)
```python
# ✅ AFTER
self.decoder.layers[0].self_attn.std_match_ratio = 0.15  # 10x gentler!
self.decoder.layers[1].self_attn.std_match_ratio = 0.10
```

**Why this matters**: `std_match_ratio` controls how strong the bias is relative to attention scores:
- `0.1-0.2`: Gentle guidance (recommended for learning)
- `0.3-0.5`: Moderate influence
- `0.8+`: Dominates attention (breaks learning!)
- `1.6`: Completely overwhelms (what you had)

### 2. **Created Safe Config** (configs/ascender_safe.yaml)

Key changes:
- **Disabled alignment component** initially (avoids double-counting q·k)
- **Lower gate ceiling**: 0.35 instead of 0.65
- **Enabled auto-calibration**: Self-regulates bias strength
- **Lower ASC learning rate**: 0.0002 vs 0.0005
- **Tighter gamma cap**: 2.5 instead of 6.0
- **Lower component weights**: w_coh=0.08 (was 0.10)

### 3. **Disabled Aggressive Tau Schedule** (train.py:250-260)

The cosine temperature schedule (1.0 → 1.10) was softening attention while bias was hardening it. Conflicting signals = poor learning.

### 4. **Created Baseline Config** (configs/baseline.yaml)

For proper A/B testing without ASCender.

## 📊 Expected Results

### Before (Your Current Results)
```
Epoch 1: Loss = 9.89
Epoch 2: Loss = 9.51
Epoch 3: Loss = 9.28
```
❌ Barely learning, 2x worse than baseline

### After (Expected with Safe Config)
```
Epoch 1: Loss = 4.0-4.2
Epoch 2: Loss = 3.7-3.9
Epoch 3: Loss = 3.5-3.7
```
✅ Should match or beat baseline

If ASCender helps, you might see:
- Faster convergence (better loss early on)
- Lower final loss (by 0.1-0.3)
- Better attention patterns in heatmaps

## 🚀 How to Test

### Step 1: Run Baseline
```bash
cd /Users/aepeul/ASCender/basic-transformer
python -m src.train --config configs/baseline.yaml
```

### Step 2: Run Safe ASCender
```bash
python -m src.train --config configs/ascender_safe.yaml
```

### Step 3: Compare Results
```bash
cat logs/results_summary.csv | tail -20
```

Look for:
- **Baseline**: `use_ascender=False`
- **ASCender**: `use_ascender=True, bias_combo=C` (cohesion only)

### Step 4: Check A/B Diagnostics

During training, look for lines like:
```
[AB] NLL on=3.845 | off=3.912
```

- If `on < off`: ✅ Bias is helping!
- If `on > off`: ❌ Bias is hurting

## 🎛️ Tuning Guide (After Safe Config Works)

Once you confirm the safe config works, you can gradually increase bias strength:

### Phase 1: Verify Cohesion Works
```yaml
use_cohesion: true
w_coh: 0.08
std_match_ratio: 0.15  # (in transformer.py)
```

### Phase 2: Add Alignment (Carefully)
```yaml
use_alignment: true
w_align: 0.03         # Start very small!
align_source: "preproj"  # Use embeddings, not qk
```

### Phase 3: Increase Strength (Gradually)
```yaml
# In transformer.py, try increasing slowly:
std_match_ratio: 0.15 → 0.20 → 0.25
```

**Rule**: Increase by 0.05 at a time, verify it helps before going further.

### Phase 4: Open Gate (Optional)
```yaml
gate_ceiling: 0.35 → 0.40 → 0.45
```

## 🐛 Debugging Checklist

If loss is still high (>6.0):

1. ✅ Verify std_match_ratio in transformer.py is ≤ 0.20
2. ✅ Check config: `use_alignment: false` initially
3. ✅ Confirm auto_calibrate is enabled with low target (0.12-0.15)
4. ✅ Look at A/B diagnostics in training logs
5. ✅ Check gamma values don't explode: should stay < 2.0
6. ✅ Verify gate stays closed: should be < 0.5

## 📈 Understanding the Components

### Cohesion (C)
- **Effect**: Encourages attending to nearby tokens
- **Weight**: `w_coh * gaussian(distance, sigma=3.0)`
- **Safe range**: w_coh = 0.05 - 0.15

### Alignment (A)
- **Effect**: Amplifies attention to semantically similar tokens
- **Weight**: `w_align * cosine_similarity(q, k)`
- **⚠️ Danger**: Already in attention scores! Easy to double-count
- **Safe range**: w_align = 0.01 - 0.05

### Separation (S)
- **Effect**: Discourages attending to very close tokens
- **Weight**: `-w_sep * gaussian(distance, sigma=1.0)`
- **⚠️ Rarely helps**: Usually disabled
- **Safe range**: w_sep = 0.0 (keep disabled)

## 💡 Key Insights

### Why Structural Bias is Hard

1. **Attention scores are already well-scaled** by softmax
2. **Small biases have big effects** due to softmax sensitivity
3. **Too much bias prevents learning** - model can't discover patterns
4. **Components interact multiplicatively** (bias × gate × gamma × std_match)

### The "Goldilocks Zone"

Your bias should be:
- **Strong enough** to guide attention initially
- **Weak enough** that learned patterns can override it
- **Adaptive** so it fades when not helpful

Target: `std(bias) ≈ 0.10-0.20 × std(scores)`

## 📚 References

- Original std_match_ratio: 1.6 (transformer.py:650)
- Safe std_match_ratio: 0.15 (10x reduction)
- Gate ceiling: 0.65 → 0.35 (46% reduction)
- Component weights: 0.22 → 0.03 (7x reduction for alignment)

Good luck! The safe config should get you back to baseline performance, then you can tune up from there. 🚀
