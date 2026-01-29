# Training Optimization Cheat Sheet

## 🚀 Quick Commands

```bash
# Apply all fixes
bash scripts/utils/apply_critical_fixes.sh

# Test training
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# Monitor health
python monitor_training.py train_logs/train_*.log

# Validate fixes
python validate_optimizations.py

# Benchmark
bash scripts/utils/benchmark_training.sh
```

## 🔧 Critical Fixes

| Fix | File | Line | Change |
|-----|------|------|--------|
| NCCL Timeout | train_qwen_english_l2t_stable.sh | 141 | 12000000 → 1800 |
| Data Workers | train_qwen_english_l2t_stable.sh | 116 | 16 → 8 |
| Prefetch | data_simple.py | 138 | Add prefetch_factor=4 |
| DDP | train_mmdit_stable.py | 276 | Add gradient_as_bucket_view=True |
| Optimizer | optimizer.py | - | Add fused=True |

## 📊 Performance Impact

```
Baseline:    100% speed, 100% memory
Optimized:   130% speed,  65% memory
Improvement: +30% speed, -35% memory
```

## ⚙️ Environment Variables

```bash
# Conservative (if NaN occurs)
LEARNING_RATE=3e-5
GRAD_CLIP_NORM=0.3
WARMUP_STEPS=3000
LATENT_LOSS_WEIGHT=0.05

# Balanced (default)
LEARNING_RATE=5e-5
GRAD_CLIP_NORM=0.5
WARMUP_STEPS=2000
LATENT_LOSS_WEIGHT=0.1

# Aggressive (if stable)
LEARNING_RATE=1e-4
GRAD_CLIP_NORM=1.0
WARMUP_STEPS=1000
LATENT_LOSS_WEIGHT=0.5
```

## 🎯 Batch Size Guide

| GPU Memory | Batch Size | Grad Accum |
|------------|------------|------------|
| 24GB (3090) | 2 | 4 |
| 40GB (A100) | 4 | 2 |
| 80GB (H100) | 8 | 1 |

## 🔍 Health Indicators

### ✅ Good Training
- Loss: 3.0 → 2.0 in first 500 steps
- Grad norm: < 1.0
- GPU util: > 85%
- No errors

### ❌ Problems
- Loss: Not decreasing → Increase LR
- Grad norm: > 5.0 → Reduce LR
- GPU util: < 70% → Optimize data loading
- NaN: → Reduce LR, increase warmup

## 🐛 Quick Fixes

```bash
# OOM
L2T_TRAIN_BS=2 GRAD_ACCUM_STEPS=4

# NaN
LEARNING_RATE=3e-5 GRAD_CLIP_NORM=0.3

# Slow
DATA_WORKERS=8 L2T_TRAIN_BS=8

# Connection failed
MASTER_ADDR=<ip> MASTER_PORT=29500
```

## 📈 Monitoring

```bash
# Watch loss
tail -f train_logs/train_*.log | grep "Loss:"

# Check errors
grep -E "ERROR|NaN" train_logs/train_*.log

# GPU memory
nvidia-smi dmon -s mu

# GPU utilization
nvidia-smi dmon -s u
```

## 🔄 Rollback

```bash
# Restore backup
cp -r ../backup_YYYYMMDD_HHMMSS/* .

# Or restore specific file
cp ../backup_*/train_qwen_english_l2t_stable.sh .
```

## 📚 Documentation

| Doc | Purpose |
|-----|---------|
| MASTER_PLAN.md | Complete overview |
| QUICK_IMPLEMENTATION_GUIDE.md | Code snippets |
| MIGRATION_GUIDE.md | Step-by-step |
| TROUBLESHOOTING_GUIDE.md | Common issues |

## ⏱️ Implementation Time

- Critical fixes: 30 min
- Data loading: 30 min
- Memory optimization: 1 hour
- DDP optimization: 30 min
- Testing: 2 hours
- **Total: ~5 hours**

## 🎯 Success Metrics

- [ ] Speed: +30%
- [ ] Memory: -35%
- [ ] Batch size: 4 → 8
- [ ] No NaN in 1000 steps
- [ ] Multi-node works

---

**Quick Start**: `bash scripts/utils/apply_critical_fixes.sh && NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh`
