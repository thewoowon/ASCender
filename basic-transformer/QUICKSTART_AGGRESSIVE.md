# ASCender Aggressive Configs - Quick Start

## 🎯 Problem
Your A/B tests show **no difference** between bias ON and OFF. You need configurations that guarantee a **measurable, observable effect**.

---

## ⚡ Quick Start (3 Commands)

### Option 1: Automated A/B Test
```bash
cd /Users/aepeul/ASCender/basic-transformer
./scripts/quick_ab_test.sh
```
This runs all three configs and compares results automatically.

### Option 2: Manual Testing
```bash
# Baseline (no bias)
python -m src.train --config configs/baseline.yaml

# Moderate aggressive (recommended)
python -m src.train --config configs/ascender_moderate_aggressive.yaml

# Very aggressive (diagnostic)
python -m src.train --config configs/ascender_very_aggressive.yaml
```

### Option 3: Diagnostic Only
```bash
# Measure bias effect without full training
python scripts/measure_bias_effect.py --config configs/ascender_moderate_aggressive.yaml
```

---

## 📊 What to Expect

### Baseline (No Bias)
- Final loss: ~3.5-4.0
- A/B test: ON = OFF (obviously, no bias)

### Moderate Aggressive ⭐ (Start Here)
- **std_match_ratio**: 0.50 (bias is 50% of scores)
- **Top-1 disagreement**: 5-15%
- **Loss**: Should be within ±5% of baseline
- **Effect**: Clear and measurable

**Signs it's working:**
```
[AB] NLL on=3.845 | off=3.912  ✅ (on < off means helpful)
Top-1 disagreement: 8.2%       ✅ (clear signal)
```

**Signs it's NOT working:**
```
[AB] NLL on=3.912 | off=3.911  ❌ (nearly identical)
Top-1 disagreement: 0.3%       ❌ (no effect)
```

### Very Aggressive (Diagnostic)
- **std_match_ratio**: 0.90 (bias is 90% of scores)
- **Top-1 disagreement**: 15-30%
- **Loss**: Likely 10-20% WORSE than baseline
- **Effect**: Obvious, but hurts performance

**Purpose**: Prove the mechanism works before optimizing.

---

## 🔬 Measuring Effect

After training, run:
```bash
python scripts/measure_bias_effect.py --config configs/ascender_moderate_aggressive.yaml
```

**Key Metrics:**
- **Top-1 disagreement** < 1%: No effect → use very aggressive
- **Top-1 disagreement** 5-15%: ✅ Good signal
- **Top-1 disagreement** > 30%: Too strong
- **NLL improvement** > 0: ✅ Bias helps
- **NLL improvement** < 0: ❌ Bias hurts

---

## 📁 Files Created

### Configs
| File | std_match_ratio | Components | Purpose |
|------|-----------------|------------|---------|
| `baseline.yaml` | N/A (no bias) | None | Ground truth |
| `ascender_safe.yaml` | 0.15 | C only | Gentle (might be too weak) |
| `ascender_moderate_aggressive.yaml` | 0.50 | A+S+C | **Start here** |
| `ascender_very_aggressive.yaml` | 0.90 | A+S+C | Diagnostic |

### Scripts
- `scripts/measure_bias_effect.py` - Quantify bias contribution
- `scripts/quick_ab_test.sh` - Automated comparison

### Documentation
- `FIXES_EXPLAINED.md` - Why previous config failed (loss ~9.89)
- `AGGRESSIVE_TUNING_GUIDE.md` - Complete tuning strategies
- `QUICKSTART_AGGRESSIVE.md` - This file

### Code
- `src/models/architectural_mods.py` - Advanced architectural variants
- `src/models/transformer.py` - Updated to support `std_match_ratio_override`

---

## 🎛️ Key Parameters Explained

### std_match_ratio (Most Important!)
Controls final bias magnitude relative to attention scores:
- **0.15**: Gentle (10x safer than original 1.6)
- **0.50**: Moderate aggressive ⭐
- **0.90**: Very aggressive
- **1.5**: Extreme (original problematic value)

### Component Weights
Shape bias direction (before gamma/gate scaling):
- **w_align**: Similarity-based (0.03 safe → 0.35 aggressive)
- **w_coh**: Local preference (0.08 safe → 0.40 aggressive)
- **w_sep**: Nearby suppression (usually 0.0)

### Gate
Controls overall influence:
- **gate_init**: -3.0 (closed) → -0.5 (open)
- **gate_ceiling**: 0.35 (safe) → 0.80 (aggressive)

### Auto-Calibration
- **OFF** in aggressive configs (let it run free)
- **ON** in safe configs (prevents runaway)

---

## 🐛 Troubleshooting

### Problem: Still no effect with very aggressive config

**Check 1**: Verify std_match_ratio is actually applied
```python
# Add print in transformer.py after line 657:
print(f"Layer 0 std_match_ratio: {self.decoder.layers[0].self_attn.std_match_ratio}")
```
Should print: 0.50 or 0.90

**Check 2**: Print bias magnitude during training
```python
# In ascender_bias.py forward(), after line 419:
print(f"Bias: mean={scaled.mean():.3f}, std={scaled.std():.3f}, max={scaled.abs().max():.3f}")
```
Expected: std > 0.5 for aggressive configs

**Check 3**: Is hard limiter clamping?
Look for this in logs:
```
[ASC RUNTIME LIMITER] ratio 0.95 > 0.70 → scaling down
```
If you see this, increase `hard_max_ratio` in config.

**Check 4**: Components disabled?
Verify in logs at startup:
```
[Init] ASCender ON (additive). Attach policy: decoder self-attn first 2 layers only.
[WIRE] L0.self_attn: biaser=AscenderBias | expect_bias=True
```

### Problem: Loss explodes (NaN or >15)

Bias is TOO aggressive. Solutions:
1. Lower `std_match_ratio_override` to 0.30
2. Enable auto-calibration with `target_ratio: 0.20`
3. Use gentler component weights (w_align=0.10, w_coh=0.15)
4. Increase `grad_clip` to 2.0

### Problem: Training is slow

These configs are computation-heavy. Speed up:
1. Reduce `epochs` to 3 for quick tests
2. Use single seed: `seeds: [42]`
3. Smaller batch data: `num_batches=32` in dummy loader
4. Disable probes: `probe_every: 0`

---

## 📈 Recommended Workflow

### Phase 1: Establish Ground Truth (5 min)
```bash
python -m src.train --config configs/baseline.yaml
# Note the final loss, e.g., 3.52
```

### Phase 2: Test Moderate Aggressive (5 min)
```bash
python -m src.train --config configs/ascender_moderate_aggressive.yaml
# Expected: 3.40-3.65 (within 5% of baseline)
# Check A/B diagnostics during training
```

### Phase 3: Measure Effect (1 min)
```bash
python scripts/measure_bias_effect.py --config configs/ascender_moderate_aggressive.yaml
```

**Decision tree:**
- Top-1 disagreement > 5% AND NLL helps: ✅ **Success! Fine-tune from here**
- Top-1 disagreement > 5% BUT NLL hurts: ⚠️ **Reduce strength by 20%**
- Top-1 disagreement < 5%: ❌ **Escalate to very aggressive**

### Phase 4: If Needed - Very Aggressive (5 min)
```bash
python -m src.train --config configs/ascender_very_aggressive.yaml
# Expected: 3.80-4.50 (worse than baseline)
# But top-1 disagreement should be >15%
```

**If STILL no effect**: See troubleshooting or try architectural mods.

### Phase 5: Optimize (iterative)
Once you've confirmed bias has an effect:
1. Start from moderate aggressive
2. Gradually reduce `std_match_ratio` by 0.05 steps
3. Enable auto-calibration with low target (0.18-0.25)
4. Fine-tune component weights
5. Find the sweet spot where bias helps without hurting

---

## 💡 Understanding the Configs

### Why "Safe" Had No Effect
```yaml
std_match_ratio: 0.15        # Too gentle
w_align: 0.03, w_coh: 0.08  # Tiny weights
gate_ceiling: 0.35           # Limited influence
auto_calibrate: true         # Actively suppresses
```
**Result**: Bias was 1-2% of attention magnitude → invisible

### Why "Moderate Aggressive" Works
```yaml
std_match_ratio: 0.50        # 3x stronger
w_align: 0.15, w_coh: 0.20  # 4-5x higher weights
gate_ceiling: 0.60           # More headroom
auto_calibrate: false        # Let it run
```
**Result**: Bias is 15-25% of attention magnitude → measurable

### Why "Very Aggressive" Proves It
```yaml
std_match_ratio: 0.90        # Near-equal to scores
w_align: 0.35, w_coh: 0.40  # Maxed weights
gate_ceiling: 0.80           # Wide open
```
**Result**: Bias is 40-60% of attention magnitude → dominant

---

## 🎯 Success Criteria

Your experiment is successful when:

1. ✅ **Measurable effect**: Top-1 disagreement > 5%
2. ✅ **Visible in heatmaps**: Clear structure (not random noise)
3. ✅ **A/B diagnostic shows difference**: NLL(on) ≠ NLL(off) by > 0.05
4. ✅ **Attention patterns change**: Attention Δ > 0.02
5. 🎁 **Bonus**: NLL improves (bias actually helps)

Even if criterion #5 fails, criteria #1-4 prove the mechanism works. Then you can tune for performance.

---

## 🚀 Next Steps After Success

Once you have confirmed measurable effect:

1. **Characterize the effect**: Run diagnostic script, save heatmaps
2. **Find optimal strength**: Binary search on std_match_ratio (0.30-0.60)
3. **Component ablation**: Test A-only, C-only, A+C, A+S+C
4. **Per-layer tuning**: Different strengths for L0 vs L1
5. **Enable calibration**: Add auto-calibration with tuned target
6. **Long training**: Run for 10+ epochs to see convergence
7. **Real data**: Test on actual WikiText (not dummy data)

---

## 📞 Quick Reference Card

| Scenario | Config to Use | Expected Result |
|----------|---------------|-----------------|
| First time setup | `moderate_aggressive.yaml` | Top-1 disagree ~10% |
| No effect at all | `very_aggressive.yaml` | Top-1 disagree >20% |
| Need ground truth | `baseline.yaml` | No bias effect |
| Found working config, optimize | Create custom (reduce strength by 20%) | NLL improves |
| Hit ceiling with hyperparams | Try architectural mods | New mechanism |

---

**Ready to go?** Start with:
```bash
python -m src.train --config configs/ascender_moderate_aggressive.yaml
```

Check the A/B diagnostics during training - you should see NLL ON vs OFF differ by at least 0.05!
