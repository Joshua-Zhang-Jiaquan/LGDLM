# NaN Gradient Fix - Complete Solution Summary

## Problem Identified

Your training encountered persistent NaN gradients starting at **step 92261** and continuing through **step 190163**. The training was running on 2 nodes × 8 GPUs (16 GPUs total) with:
- Model: MMDiT with 892M parameters
- Latent dimension: 32
- Training mode: l2t (latent-to-text)
- Learning rate: 1e-4
- Gradient clip: 1.0
- Dtype: bf16

## Root Causes

1. **Insufficient gradient clipping** - Current value of 1.0 is too high for this model size
2. **High learning rate** - 1e-4 is aggressive for 892M parameters
3. **Unstable loss computation** - Division without epsilon, no normalization of latents
4. **Poor NaN handling** - Current code zeros gradients but continues, doesn't skip bad batches
5. **No loss validation** - Backward pass happens even with invalid loss values

## Solution Overview

I've created a complete fix package with 5 files:

### 1. **NAN_GRADIENT_FIXES.md**
Complete analysis document with:
- Root cause analysis
- Detailed solutions
- Prevention checklist
- Monitoring commands

### 2. **TRAIN_MMDIT_PATCH.py**
Patches for `latentDLM_mmdit/train_mmdit.py`:
- Loss validation BEFORE backward pass
- Improved gradient clipping (clip BEFORE NaN check)
- Skip bad batches instead of zeroing gradients
- Better error reporting with diagnostics

### 3. **IMPROVED_TRAINER_PATCH.py**
Patches for `latentDLM_mmdit/improved_trainer.py`:
- Add epsilon (1e-8) to all divisions
- Normalize latents before MSE loss
- Clamp loss values to prevent overflow
- Validate loss at each computation step
- Reduce latent loss weight to 0.1

### 4. **mmdit_stable_config.yaml**
Stable training configuration:
- Learning rate: 5e-5 (reduced from 1e-4)
- Gradient clip: 0.5 (reduced from 1.0)
- Warmup steps: 2000 (increased from 1000)
- Latent loss weight: 0.1 (reduced from 1.0)
- Gradient accumulation: 2 steps
- More frequent checkpointing

### 5. **apply_nan_fixes.sh**
Helper script to:
- Create backups of original files
- Install stable configuration
- Guide you through manual changes

## Quick Start - Apply Fixes Now

### Step 1: Run the helper script
```bash
cd /inspire/hdd/project/project-public/zhangjiaquan-253108540222/jiaquan/latent/MM-LDLM
bash scripts/utils/apply_nan_fixes.sh
```

This will create backups and install the stable config.

### Step 2: Apply code patches

**Option A: Manual editing (recommended)**
1. Open `latentDLM_mmdit/train_mmdit.py`
2. Find lines 420-437 (the backward pass section)
3. Replace with code from `results/archive/TRAIN_MMDIT_PATCH.py`

4. Open `latentDLM_mmdit/improved_trainer.py`
5. Find lines 336-374 (the loss computation section)
6. Replace with code from `results/archive/IMPROVED_TRAINER_PATCH.py`

**Option B: Use sed (automated but risky)**
```bash
# Backup first!
cp latentDLM_mmdit/train_mmdit.py latentDLM_mmdit/train_mmdit.py.backup
cp latentDLM_mmdit/improved_trainer.py latentDLM_mmdit/improved_trainer.py.backup

# Then manually apply patches using the patch files as reference
```

### Step 3: Update your training script

Modify `train_qwen_english_l2t_scaled.sh` or use the improved version:
```bash
# Use stable config
L2T_CONFIG="${L2T_CONFIG:-mmdit_stable}"

# Or override specific parameters
export OPTIMIZER_LR=5e-5
export GRAD_CLIP_NORM=0.5
```

### Step 4: Test with single node first

```bash
# Test with 2 GPUs on single node
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_scaled_improved.sh
```

Monitor the output for:
- No "ERROR: Invalid loss" messages
- No "ERROR: Invalid gradient norm" messages
- Stable loss values (should decrease smoothly)
- Gradient norms < 1.0

### Step 5: Scale up gradually

If single-node works:
```bash
# Try 2 nodes
NNODES=2 bash scripts/training/train_qwen_english_l2t_scaled_improved.sh

# Then full scale
NNODES=8 bash scripts/training/train_qwen_english_l2t_scaled_improved.sh
```

## Key Changes Summary

| Component | Before | After | Why |
|-----------|--------|-------|-----|
| Learning rate | 1e-4 | 5e-5 | Prevent gradient explosion |
| Grad clip | 1.0 | 0.5 | Better gradient control |
| Warmup steps | 1000 | 2000 | Smoother learning rate ramp |
| Latent loss weight | 1.0 | 0.1 | Prevent latent loss dominance |
| Loss validation | None | Before backward | Catch bad batches early |
| Gradient clipping | After NaN check | Before NaN check | Clip first, then check |
| Bad batch handling | Zero grads, continue | Skip batch entirely | Don't train on bad data |
| Latent normalization | None | L2 normalize | Numerical stability |
| Division epsilon | None | 1e-8 | Prevent divide by zero |

## Monitoring During Training

### Real-time monitoring
```bash
# Watch for errors
tail -f train_logs/latest.log | grep -E "ERROR|NaN|loss:"

# Monitor gradient norms
tail -f train_logs/latest.log | grep "grad_norm"

# Check loss values
tail -f saved/mmdit-qwen-32d-l2t-stable/training_log.jsonl | jq '.loss'
```

### Post-training analysis
```bash
# Plot loss curve
python -c "
import json
import matplotlib.pyplot as plt

losses = []
with open('saved/mmdit-qwen-32d-l2t-stable/training_log.jsonl') as f:
    for line in f:
        data = json.loads(line)
        losses.append(data['loss'])

plt.figure(figsize=(12, 6))
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.yscale('log')
plt.grid(True)
plt.savefig('loss_curve.png', dpi=150)
print(f'Plotted {len(losses)} steps')
"
```

## Expected Results

With these fixes, you should see:
- ✅ No NaN gradients
- ✅ Smooth loss decrease
- ✅ Gradient norms consistently < 1.0
- ✅ Stable training for 100k+ steps
- ✅ Better convergence

## If Issues Persist

1. **Reduce learning rate further**: Try 3e-5 or 1e-5
2. **Increase gradient accumulation**: Set to 4 or 8 steps
3. **Check data quality**: Verify no NaN/Inf in preprocessed data
4. **Use fp32**: Change `training.dtype: fp32` (slower but more stable)
5. **Resume from earlier checkpoint**: Before NaN occurred

## Files Created

```
MM-LDLM/
├── NAN_GRADIENT_FIXES.md              # Complete analysis
├── TRAIN_MMDIT_PATCH.py               # Training script patches
├── IMPROVED_TRAINER_PATCH.py          # Trainer patches
├── mmdit_stable_config.yaml           # Stable configuration
├── apply_nan_fixes.sh                 # Helper script
└── NAN_FIX_SUMMARY.md                 # This file
```

## Next Steps

1. ✅ Read this summary
2. ⬜ Run `bash scripts/utils/apply_nan_fixes.sh`
3. ⬜ Apply code patches manually
4. ⬜ Test with single node (2 GPUs)
5. ⬜ Monitor for 1000 steps
6. ⬜ Scale up if stable
7. ⬜ Report results

## Support

If you encounter issues:
1. Check the detailed analysis in `NAN_GRADIENT_FIXES.md`
2. Review the patch files for exact code changes
3. Verify all changes were applied correctly
4. Test with even lower learning rate (3e-5)
5. Share the new training logs for further analysis

Good luck with your training! 🚀
