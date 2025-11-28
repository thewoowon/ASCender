# ModelNet40 Scaling Experiments

## Goal
Validate the **Capacity-Bias Trade-off Hypothesis** on real-world data (ModelNet40).

**Hypothesis**: ASCender helps ultra-lightweight models but hurts larger models due to optimization interference.

## Experiment Plan

### Model Sizes to Test

| Model | Hidden Dim | Params | Expected Result |
|-------|-----------|--------|-----------------|
| **Ultra-lightweight** | 32 | ~7K | ✅ **DONE** (Baseline: 74.84%, ASCender: 76.05%, Δ=+1.21%) |
| **Small** | 48 | ~14K | ❓ Crossover point? |
| **Medium-small** | 64 | ~24K | ❓ ASCender may start hurting |
| **Medium** | 80 | ~37K | ❓ Expect negative Δ |

### Commands

#### 1. Hidden Dim = 48 (~14K params)

**Baseline:**
```bash
nohup python experiments/modelnet40_scaling_experiment.py \
  --hidden-dim 48 \
  --epochs 50 \
  --batch-size 32 \
  --output results/modelnet40_scaling/h48_baseline.json \
  > logs/h48_baseline.log 2>&1 &
```

**ASCender:**
```bash
nohup python experiments/modelnet40_scaling_experiment.py \
  --hidden-dim 48 \
  --epochs 50 \
  --batch-size 32 \
  --use-ascender \
  --output results/modelnet40_scaling/h48_ascender.json \
  > logs/h48_ascender.log 2>&1 &
```

#### 2. Hidden Dim = 64 (~24K params)

**Baseline:**
```bash
nohup python experiments/modelnet40_scaling_experiment.py \
  --hidden-dim 64 \
  --epochs 50 \
  --batch-size 32 \
  --output results/modelnet40_scaling/h64_baseline.json \
  > logs/h64_baseline.log 2>&1 &
```

**ASCender:**
```bash
nohup python experiments/modelnet40_scaling_experiment.py \
  --hidden-dim 64 \
  --epochs 50 \
  --batch-size 32 \
  --use-ascender \
  --output results/modelnet40_scaling/h64_ascender.json \
  > logs/h64_ascender.log 2>&1 &
```

#### 3. Hidden Dim = 80 (~37K params)

**Baseline:**
```bash
nohup python experiments/modelnet40_scaling_experiment.py \
  --hidden-dim 80 \
  --epochs 50 \
  --batch-size 32 \
  --output results/modelnet40_scaling/h80_baseline.json \
  > logs/h80_baseline.log 2>&1 &
```

**ASCender:**
```bash
nohup python experiments/modelnet40_scaling_experiment.py \
  --hidden-dim 80 \
  --epochs 50 \
  --batch-size 32 \
  --use-ascender \
  --output results/modelnet40_scaling/h80_ascender.json \
  > logs/h80_ascender.log 2>&1 &
```

## Execution Strategy

### Option 1: Sequential (Safe, slower)
Run experiments one by one to avoid CPU overload:
1. h48 baseline → h48 ascender
2. h64 baseline → h64 ascender
3. h80 baseline → h80 ascender

**Total time**: ~6 × 33 min = **~3.3 hours**

### Option 2: Parallel Pairs (Faster, higher CPU load)
Run baseline + ascender for same size in parallel:
1. h48 baseline + h48 ascender (parallel)
2. h64 baseline + h64 ascender (parallel)
3. h80 baseline + h80 ascender (parallel)

**Total time**: ~3 × 33 min = **~1.7 hours**

### Option 3: All Parallel (Fastest, max CPU load)
Run all 6 experiments at once on M3 Mac (8+ cores).

**Total time**: ~33 min (if CPU handles it)

## Monitoring

Check progress:
```bash
# List running experiments
ps aux | grep modelnet40_scaling

# Check logs
tail -f logs/h48_baseline.log
tail -f logs/h48_ascender.log

# Check output files
ls -lh results/modelnet40_scaling/
```

## Expected Results

Based on synthetic data experiments:

| Model Size | Baseline Acc | ASCender Acc | Δ | Interpretation |
|-----------|--------------|--------------|---|----------------|
| 7K (done) | 74.84% | 76.05% | **+1.21%** | ✅ ASCender helps |
| 14K | ~78%? | ~78-79%? | +0-1% | Crossover point |
| 24K | ~80%? | ~78-79%? | -1-2%? | ASCender starts hurting |
| 37K | ~81%? | ~78%? | -3%? | ASCender hurts more |

**Key Question**: Does the crossover point (~10-15K on synthetic) hold on real-world data?

## Analysis Plan

After all experiments complete:

1. **Plot scaling curve** (similar to Figure 5 in paper)
2. **Compare with synthetic results** (Section 6.1)
3. **Update paper Section 6.5** with complete ModelNet40 validation
4. **Discuss why real-world crossover may differ from synthetic**

## Notes

- Each experiment: ~33 minutes on M3 Mac CPU
- Batch size 32 uses ~1-2GB RAM per process
- M3 Mac (8-core) can handle 2-4 parallel experiments comfortably
- Results saved to `results/modelnet40_scaling/*.json`
- Logs saved to `logs/h*_{baseline,ascender}.log`

---

**Status**: 📝 Ready to run
**Next**: Choose execution strategy and start experiments
