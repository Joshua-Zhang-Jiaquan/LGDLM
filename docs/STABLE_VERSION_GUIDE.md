# Stable Training Files - Usage Guide

## What Are These Files?

I've created **stable versions** of your training code with all NaN gradient fixes applied. These are **separate files** that don't modify your original code.

### Files Created

1. **`latentDLM_mmdit/improved_trainer_stable.py`** ✅
   - Copy of `improved_trainer.py` with NaN-safe loss computation
   - Adds epsilon to divisions
   - Normalizes latents before MSE loss
   - Validates loss values
   - Clamps loss to prevent overflow

2. **`latentDLM_mmdit/train_mmdit_stable.py`** ✅
   - Copy of `train_mmdit.py` with NaN-safe gradient handling
   - Validates loss BEFORE backward pass
   - Clips gradients BEFORE NaN check
   - Skips bad batches instead of continuing
   - Better error reporting

3. **`launch_stable_training.py`** ✅
   - Simple launcher script for stable version

## How to Use Stable Versions

### Option 1: Use the Bash Script (Easiest)

The bash script `train_qwen_english_l2t_stable.sh` can be updated to use the stable version:

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Edit the bash script to use train_mmdit_stable.py instead of train_mmdit.py
# Change line with: latentDLM_mmdit/train_mmdit.py
# To: latentDLM_mmdit/train_mmdit_stable.py
```

### Option 2: Use torchrun Directly

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  latentDLM_mmdit/train_mmdit_stable.py \
  --config-name mmdit_stable \
  logging.run_name=mmdit-qwen-32d-l2t-stable \
  training.train_batch_size=4 \
  optimizer.lr=5e-5 \
  optimizer.grad_clip_norm=0.5
```

### Option 3: Use the Launcher Script

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  launch_stable_training.py \
  --config-name mmdit_stable
```

## Comparison: Original vs Stable

| Feature | Original | Stable |
|---------|----------|--------|
| **Loss validation** | None | ✅ Before backward |
| **Gradient clipping order** | After NaN check | ✅ Before NaN check |
| **Bad batch handling** | Zero grads, continue | ✅ Skip batch entirely |
| **Division safety** | No epsilon | ✅ Epsilon (1e-8) |
| **Latent normalization** | None | ✅ L2 normalize |
| **Loss clamping** | None | ✅ Clamp to prevent overflow |
| **Error reporting** | Basic warning | ✅ Detailed diagnostics |

## What's Different in Stable Versions?

### improved_trainer_stable.py Changes

**Loss Computation (lines 336-374):**
```python
# Added:
eps = 1e-8  # Numerical stability

# Text loss with epsilon
mask_sum = text_mask.sum().clamp(min=1) + eps
text_loss = (text_loss_unmasked * text_mask.view(-1)).sum() / mask_sum
text_loss = torch.clamp(text_loss, min=0.0, max=100.0)

# Latent loss with normalization
latent_pred_norm = F.normalize(latent_pred, p=2, dim=-1, eps=eps)
latent_target_norm = F.normalize(latent_target, p=2, dim=-1, eps=eps)
latent_loss = F.mse_loss(latent_pred_norm, latent_target_norm)
latent_loss = torch.clamp(latent_loss, min=0.0, max=10.0)

# Apply reduced weight
latent_loss_weight = 0.1  # Reduced from 1.0
total_loss = total_loss + latent_loss * latent_loss_weight
```

### train_mmdit_stable.py Changes

**Gradient Handling (lines 420-437):**
```python
# Validate loss BEFORE backward
if torch.isnan(loss) or torch.isinf(loss):
    print("ERROR: Invalid loss detected")
    optimizer.zero_grad(set_to_none=True)
    continue  # Skip this batch

# Backward pass
(loss * config.loss.loss_scale).backward()

# Clip BEFORE checking
norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), grad_clip_value)

# Check AFTER clipping
if torch.isnan(norm) or torch.isinf(norm):
    print("ERROR: Invalid gradient norm")
    optimizer.zero_grad(set_to_none=True)
    continue  # Skip optimizer step

# Only step if everything is valid
optimizer.step()
```

## Switching Between Versions

### Use Original Version
```bash
# In bash script, use:
latentDLM_mmdit/train_mmdit.py
```

### Use Stable Version (Recommended)
```bash
# In bash script, use:
latentDLM_mmdit/train_mmdit_stable.py
```

## Testing the Stable Version

```bash
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Quick test with 2 GPUs
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  latentDLM_mmdit/train_mmdit_stable.py \
  --config-name mmdit_stable \
  training.num_train_steps=100 \
  logging.log_freq=10
```

Watch for:
- ✅ No "ERROR: Invalid loss" messages
- ✅ No "ERROR: Invalid gradient norm" messages
- ✅ Smooth loss decrease
- ✅ Gradient norms < 1.0

## Advantages of Stable Version

1. **No code modification needed** - Original files untouched
2. **Easy to switch** - Just change the script name
3. **Side-by-side comparison** - Can test both versions
4. **Safe to experiment** - Original code preserved
5. **All fixes included** - No manual patching needed

## Files Summary

```
latentDLM_mmdit/
├── improved_trainer.py          # Original (unchanged)
├── improved_trainer_stable.py   # ✅ Stable with NaN fixes
├── train_mmdit.py               # Original (unchanged)
└── train_mmdit_stable.py        # ✅ Stable with NaN fixes

Root directory/
├── launch_stable_training.py    # ✅ Launcher for stable version
└── train_qwen_english_l2t_stable.sh  # Bash script (needs update)
```

## Next Steps

1. **Update bash script** to use `train_mmdit_stable.py`
2. **Test with 2 GPUs** to verify it works
3. **Monitor for 1000 steps** to ensure stability
4. **Scale up** if everything looks good

## Updating the Bash Script

Edit `train_qwen_english_l2t_stable.sh` and change line ~280:

```bash
# Find this line:
latentDLM_mmdit/train_mmdit.py \

# Change to:
latentDLM_mmdit/train_mmdit_stable.py \
```

That's it! The stable version will be used automatically.
