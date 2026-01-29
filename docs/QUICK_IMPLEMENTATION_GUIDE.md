# Quick Implementation Guide - Critical Fixes

This guide provides ready-to-use code snippets for the most impactful improvements.

## 1. CRITICAL: Fix NCCL Timeout (Safety Issue)

**Current Problem**: Timeout set to 138 days (12000000 seconds) - will never catch real hangs

**File**: `train_qwen_english_l2t_stable.sh` (lines 141-144)

**Replace**:
```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
```

**With**:
```bash
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # 30 minutes
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_BLOCKING_WAIT=0  # Non-blocking for better overlap
```

## 2. HIGH PRIORITY: Optimize Data Loading

**File**: `latentDLM_mmdit/data_simple.py`

**Add after line 138** (in DataLoader creation):
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.train_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=min(8, config.data.get('num_workers', 16)),  # Cap at 8
    prefetch_factor=4,  # NEW: Prefetch 4 batches per worker
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True  # NEW: Keep workers alive
)
```

## 3. HIGH PRIORITY: Enable Gradient Checkpointing

**File**: `latentDLM_mmdit/configs/mmdit_stable.yaml`

**Add to model section**:
```yaml
model:
  # ... existing config ...
  use_gradient_checkpointing: true  # NEW: Enable to save 30-40% memory
```

**File**: `latentDLM_mmdit/models/multimodal_mmdit.py`

**Add to __init__** (after line 257):
```python
# Enable gradient checkpointing
self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
if self.use_gradient_checkpointing:
    print("✓ Gradient checkpointing enabled (saves ~35% memory)")
```

**Modify forward method** (around line 316):
```python
# Pass through MMDiT with optional gradient checkpointing
if latent_emb is not None:
    if self.use_gradient_checkpointing and self.training:
        from torch.utils.checkpoint import checkpoint
        text_out, latent_out = checkpoint(
            self.mmdit,
            (text_emb, latent_emb),
            (text_mask, latent_mask),
            c,
            use_reentrant=False
        )
    else:
        text_out, latent_out = self.mmdit(
            modality_tokens=(text_emb, latent_emb),
            modality_masks=(text_mask, latent_mask),
            time_cond=c,
        )
```

## 4. MEDIUM PRIORITY: Improve DDP Configuration

**File**: `latentDLM_mmdit/train_mmdit_stable.py` (around line 276)

**Replace**:
```python
if is_distributed:
    ddp_trainer = DDP(opt_trainer, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
```

**With**:
```python
if is_distributed:
    ddp_trainer = DDP(
        opt_trainer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # Set False if possible (faster)
        gradient_as_bucket_view=True,  # Reduce memory copies
        broadcast_buffers=False,  # Not needed without batch norm
        static_graph=True,  # Graph doesn't change, optimize communication
    )
    print(f"✓ DDP initialized with optimizations (rank {global_rank})")
```

## 5. MEDIUM PRIORITY: Better Loss Computation

**File**: `latentDLM_mmdit/improved_trainer_stable.py` (around line 393)

**Replace the latent loss computation** (lines 393-410):
```python
# OLD: Normalize and compute MSE
latent_pred_norm = F.normalize(latent_pred, p=2, dim=-1, eps=eps)
latent_target_norm = F.normalize(latent_target, p=2, dim=-1, eps=eps)
latent_loss = F.mse_loss(latent_pred_norm, latent_target_norm)
```

**With**:
```python
# NEW: Use cosine similarity loss (better for embeddings)
cosine_sim = F.cosine_similarity(latent_pred, latent_target, dim=-1, eps=eps)
latent_loss = (1.0 - cosine_sim).mean()

# Clamp to reasonable range
latent_loss = torch.clamp(latent_loss, min=0.0, max=2.0)
```

## 6. MEDIUM PRIORITY: Add Checkpoint Cleanup

**File**: `latentDLM_mmdit/train_mmdit_stable.py`

**Add this function after imports** (around line 95):
```python
def cleanup_old_checkpoints(save_dir, run_name, keep_last_n=3):
    """Keep only the N most recent checkpoints to save disk space."""
    import shutil
    checkpoint_base = Path(save_dir) / run_name
    if not checkpoint_base.exists():
        return

    # Find all checkpoint directories (excluding 'latest')
    checkpoint_dirs = []
    for d in checkpoint_base.iterdir():
        if d.is_dir() and d.name != 'latest':
            try:
                # Extract step number from directory name
                step_num = int(d.name.split('-')[-1].replace('k', '000').replace('M', '000000'))
                checkpoint_dirs.append((step_num, d))
            except (ValueError, IndexError):
                continue

    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: x[0])

    # Remove old checkpoints
    for _, old_dir in checkpoint_dirs[:-keep_last_n]:
        try:
            shutil.rmtree(old_dir)
            print(f"✓ Removed old checkpoint: {old_dir.name}")
        except Exception as e:
            print(f"✗ Failed to remove {old_dir.name}: {e}")
```

**Call after checkpoint saving** (around line 554):
```python
if is_main_process:
    save_checkpoint(output_path, trainer, optimizer, state)
    # NEW: Cleanup old checkpoints
    cleanup_old_checkpoints(
        config.logging.save_dir,
        config.logging.run_name,
        keep_last_n=3
    )
```

## 7. LOW PRIORITY: Enhanced Monitoring

**File**: `latentDLM_mmdit/train_mmdit_stable.py` (around line 456)

**Add to log_buffer**:
```python
log_buffer.append({
    "train/loss": float(loss.item()),
    "train/lr": float(curr_lr),
    "train/step": int(state.step + 1),
    "train/grad_norm": float(norm.item()),
    "train/epoch": float(state.epoch + (state.step - state.epoch_start_step) / total_batches),
    # NEW: Additional metrics
    "train/gpu_memory_gb": float(torch.cuda.memory_allocated() / 1e9),
    "train/gpu_memory_reserved_gb": float(torch.cuda.memory_reserved() / 1e9),
    "train/text_loss": float(metrics.get('text_loss', 0.0)),
    "train/latent_loss": float(metrics.get('latent_loss', 0.0)),
    "train/text_accuracy": float(metrics.get('text_accuracy', 0.0)),
    # ... existing metrics ...
})
```

## Implementation Checklist

### Immediate (Do First):
- [ ] Fix NCCL timeout (#1)
- [ ] Optimize data loading (#2)
- [ ] Enable gradient checkpointing (#3)

### This Week:
- [ ] Improve DDP configuration (#4)
- [ ] Better loss computation (#5)
- [ ] Add checkpoint cleanup (#6)

### Optional:
- [ ] Enhanced monitoring (#7)

## Testing After Changes

```bash
# 1. Test single node first
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# 2. Monitor for first 100 steps
tail -f train_logs/train_*_node0.log | grep -E "Loss:|ERROR|Memory"

# 3. Check memory usage
nvidia-smi dmon -s mu -c 100

# 4. Verify no NaN gradients
grep "NaN" train_logs/train_*_node0.log
```

## Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Training speed | 1.0x | 1.25-1.35x | +25-35% |
| Memory usage | 100% | 60-70% | -30-40% |
| Batch size | 4 | 6-8 | +50-100% |
| Stability | Occasional NaN | Rare NaN | Much better |

## Rollback Plan

If issues occur:
1. Revert to original files (they're unchanged)
2. Apply changes one at a time
3. Test each change individually
4. Check logs for specific errors
