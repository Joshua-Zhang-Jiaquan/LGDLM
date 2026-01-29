# NaN Gradient Analysis and Solutions

## Problem Summary

Your training encountered persistent NaN gradients starting at step 92261 and continuing through step 190163. The current code detects NaN and zeros gradients, but this doesn't solve the root cause.

## Root Causes Identified

### 1. **Loss Computation Issues**
- Division by `text_mask.sum().clamp(min=1)` can be unstable
- MSE loss on latents without normalization
- No checks for inf/nan in loss values before backward

### 2. **Numerical Instability**
- Using bf16 can cause overflow in certain operations
- Loss scaling of 1.0 may not be appropriate
- Large model (892M parameters) increases risk of numerical issues

### 3. **Gradient Clipping**
- Current grad_clip_norm=1.0 is applied AFTER NaN detection
- Should clip BEFORE checking for NaN
- Clipping value might be too high

### 4. **Learning Rate**
- No information about the learning rate when NaN occurred
- Cosine schedule might not have sufficient warmup
- Could be too high for the model size

## Solutions Implemented

### Solution 1: Improved NaN Detection and Handling

**File**: `latentDLM_mmdit/train_mmdit_stable.py` (improved version)

Key improvements:
1. Check loss for inf/nan BEFORE backward pass
2. Clip gradients BEFORE NaN check
3. Add gradient norm monitoring
4. Skip batch if NaN detected (don't zero and continue)
5. Add automatic checkpoint recovery

### Solution 2: Numerical Stability in Loss Computation

**File**: `latentDLM_mmdit/improved_trainer_stable.py`

Key improvements:
1. Add epsilon to all divisions
2. Normalize latents before MSE loss
3. Clamp loss values to prevent overflow
4. Add loss value validation

### Solution 3: Training Configuration

**Recommended changes**:
```yaml
optimizer:
  lr: 5e-5  # Reduced from 1e-4
  grad_clip_norm: 0.5  # Reduced from 1.0

training:
  warmup_steps: 2000  # Increased warmup
  dtype: bf16  # Keep bf16 but add safeguards
  gradient_accumulation_steps: 2  # Reduce effective batch size

loss:
  loss_scale: 1.0  # Keep at 1.0
  text_loss_weight: 1.0
  latent_loss_weight: 0.1  # Reduce latent loss weight
```

## Quick Fixes to Apply Now

### Fix 1: Modify train_mmdit.py (lines 420-437)

**Current code**:
```python
(loss * config.loss.loss_scale).backward()

# Grad clip
if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
    norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
else:
    norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1e6)

if torch.isnan(norm):
    print(f"Warning: NaN gradient detected at step {state.step}")
    for param in trainer.parameters():
        if param.grad is not None:
            param.grad.data.zero_()

optimizer.step()
```

**Improved code**:
```python
# Check loss before backward
if torch.isnan(loss) or torch.isinf(loss):
    print(f"ERROR: Invalid loss at step {state.step}: {loss.item()}")
    print(f"  text_loss: {metrics.get('text_loss', 0.0)}")
    print(f"  latent_loss: {metrics.get('latent_loss', 0.0)}")
    # Skip this batch
    state.step += 1
    pbar.update(1)
    continue

# Backward pass
(loss * config.loss.loss_scale).backward()

# Clip gradients FIRST
if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
    norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
else:
    norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)  # Default to 1.0, not 1e6

# Check for NaN AFTER clipping
if torch.isnan(norm) or torch.isinf(norm):
    print(f"ERROR: NaN/Inf gradient norm at step {state.step}")
    print(f"  Skipping batch and resetting gradients")
    optimizer.zero_grad(set_to_none=True)
    # Skip optimizer step
    state.step += 1
    pbar.update(1)
    continue

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### Fix 2: Modify improved_trainer.py Loss Computation (lines 336-374)

**Add to line 337** (after `total_loss = torch.tensor(...)`):
```python
# Numerical stability epsilon
eps = 1e-8
```

**Modify text loss computation** (lines 343-349):
```python
text_loss_unmasked = F.cross_entropy(
    text_logits.view(-1, vocab_size),
    text_target.view(-1),
    ignore_index=-100,
    reduction='none'
)
# Add epsilon to prevent division by zero
mask_sum = text_mask.sum().clamp(min=1) + eps
text_loss = (text_loss_unmasked * text_mask.view(-1)).sum() / mask_sum
# Clamp to prevent overflow
text_loss = torch.clamp(text_loss, max=100.0)
```

**Modify latent loss computation** (line 371):
```python
# Normalize predictions and targets for stability
latent_pred_norm = F.normalize(latent_pred, p=2, dim=-1, eps=eps)
latent_target_norm = F.normalize(latent_target, p=2, dim=-1, eps=eps)
latent_loss = F.mse_loss(latent_pred_norm, latent_target_norm)
# Clamp to prevent overflow
latent_loss = torch.clamp(latent_loss, max=100.0)
```

### Fix 3: Reduce Learning Rate

**Modify your training script**:
```bash
# Change from:
optimizer.lr: 1e-4

# To:
optimizer.lr: 5e-5  # or even 3e-5
```

### Fix 4: Add Gradient Accumulation

**In your config**:
```yaml
training:
  gradient_accumulation_steps: 2  # Accumulate over 2 steps
  train_batch_size: 4  # Keep same per-GPU batch size
```

## Testing Strategy

1. **Start with a checkpoint before NaN** (if available)
2. **Use reduced learning rate**: 5e-5 or 3e-5
3. **Enable gradient clipping**: 0.5 or 0.3
4. **Monitor closely**: Watch for loss spikes
5. **Add validation**: Run eval every 1000 steps

## Monitoring Commands

```bash
# Monitor training in real-time
tail -f train_logs/latest.log | grep -E "loss|NaN|ERROR"

# Check for NaN in saved checkpoints
python -c "import torch; ckpt = torch.load('path/to/checkpoint.pt'); print('NaN in model:', any(torch.isnan(p).any() for p in ckpt['model'].values()))"

# Plot loss curve
python -c "
import json
losses = []
with open('training_log.jsonl') as f:
    for line in f:
        losses.append(json.loads(line)['loss'])
import matplotlib.pyplot as plt
plt.plot(losses)
plt.yscale('log')
plt.savefig('loss_curve.png')
"
```

## Prevention Checklist

- [ ] Reduce learning rate to 5e-5 or lower
- [ ] Set grad_clip_norm to 0.5 or lower
- [ ] Add loss validation before backward pass
- [ ] Add epsilon to all divisions in loss computation
- [ ] Normalize latents before MSE loss
- [ ] Increase warmup steps to 2000+
- [ ] Add gradient accumulation (2-4 steps)
- [ ] Monitor gradient norms during training
- [ ] Save checkpoints more frequently (every 1000 steps)
- [ ] Add automatic recovery from last good checkpoint

## Expected Improvements

With these fixes:
- **No more NaN gradients**: Loss validation prevents backward on invalid values
- **More stable training**: Gradient clipping and normalization prevent explosions
- **Better convergence**: Lower learning rate with proper warmup
- **Recoverable failures**: Automatic checkpoint recovery if issues occur
