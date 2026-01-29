# Step-by-Step Migration Guide

This guide walks you through implementing all improvements safely and systematically.

## Pre-Migration Checklist

- [ ] Backup current working directory
- [ ] Document current baseline performance
- [ ] Ensure you have at least 50GB free disk space
- [ ] Verify git status is clean (or commit current work)
- [ ] Test current training runs successfully

## Phase 1: Safety and Critical Fixes (Day 1 - 2 hours)

### Step 1.1: Create Backup

```bash
# Create timestamped backup
BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "../$BACKUP_NAME"
cp -r . "../$BACKUP_NAME/"
echo "Backup created: ../$BACKUP_NAME"
```

### Step 1.2: Baseline Performance Test

```bash
# Run 100 steps to establish baseline
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=100 \
L2T_BASE_SAVE_FREQ=999999 \
bash scripts/training/train_qwen_english_l2t_stable.sh 2>&1 | tee baseline_test.log

# Record key metrics
echo "Baseline Metrics:" > baseline_metrics.txt
grep "Loss:" baseline_test.log | tail -20 >> baseline_metrics.txt
grep "samples_per_sec" baseline_test.log | tail -10 >> baseline_metrics.txt
nvidia-smi --query-gpu=memory.used --format=csv >> baseline_metrics.txt
```

### Step 1.3: Fix NCCL Timeout (CRITICAL)

**File**: `train_qwen_english_l2t_stable.sh`

**Line 141-144**, change:
```bash
# OLD (DANGEROUS - 138 days timeout!)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
```

To:
```bash
# NEW (Safe - 30 minutes timeout)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_TIMEOUT=1800
export NCCL_BLOCKING_WAIT=0  # Non-blocking for better overlap
```

**Verification**:
```bash
grep "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" train_qwen_english_l2t_stable.sh
# Should show: 1800 (not 12000000)
```

### Step 1.4: Test After Critical Fix

```bash
# Quick test (10 steps)
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=10 \
bash scripts/training/train_qwen_english_l2t_stable.sh

# Verify no errors
echo "✓ Phase 1 complete if no errors above"
```

## Phase 2: Data Loading Optimization (Day 1 - 1 hour)

### Step 2.1: Optimize DataLoader

**File**: `latentDLM_mmdit/data_simple.py`

Find the DataLoader creation (around line 138) and modify:

```python
# OLD
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.train_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=min(4, data_config.get('num_workers', 2)),
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
)

# NEW
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.train_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    num_workers=min(8, data_config.get('num_workers', 16)),  # Increased cap
    prefetch_factor=4,  # NEW: Prefetch 4 batches per worker
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,  # NEW: Keep workers alive between epochs
)
```

### Step 2.2: Update Config

**File**: `train_qwen_english_l2t_stable.sh`

**Line 116**, change:
```bash
# OLD
DATA_WORKERS="${DATA_WORKERS:-16}"

# NEW
DATA_WORKERS="${DATA_WORKERS:-8}"  # Reduced from 16 to 8
```

### Step 2.3: Test Data Loading

```bash
# Test with new data loading
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=50 \
bash scripts/training/train_qwen_english_l2t_stable.sh 2>&1 | tee phase2_test.log

# Compare throughput
echo "Phase 1 throughput:"
grep "samples_per_sec" baseline_test.log | tail -5
echo "Phase 2 throughput:"
grep "samples_per_sec" phase2_test.log | tail -5
```

**Expected**: 10-15% improvement in samples/sec

## Phase 3: Memory Optimization (Day 2 - 2 hours)

### Step 3.1: Enable Gradient Checkpointing

**File**: `latentDLM_mmdit/configs/mmdit_stable.yaml`

Add to model section:
```yaml
model:
  # ... existing config ...
  use_gradient_checkpointing: true  # NEW
```

**File**: `latentDLM_mmdit/models/multimodal_mmdit.py`

**After line 257** (in `__init__`), add:
```python
# Enable gradient checkpointing
self.use_gradient_checkpointing = config.get("use_gradient_checkpointing", False)
if self.use_gradient_checkpointing:
    print("✓ Gradient checkpointing enabled (saves ~35% memory)")
```

**Around line 316** (in `forward`), replace:
```python
# OLD
if latent_emb is not None:
    text_out, latent_out = self.mmdit(
        modality_tokens=(text_emb, latent_emb),
        modality_masks=(text_mask, latent_mask),
        time_cond=c,
    )

# NEW
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

### Step 3.2: Test Memory Usage

```bash
# Monitor memory during training
nvidia-smi dmon -s mu -c 100 > memory_phase3.log &
MONITOR_PID=$!

# Run training
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=50 \
bash scripts/training/train_qwen_english_l2t_stable.sh

# Stop monitoring
kill $MONITOR_PID

# Check memory reduction
echo "Memory usage comparison:"
echo "Before (Phase 2):"
tail -20 memory_phase2.log | awk '{sum+=$2; count++} END {print "Average:", sum/count, "MB"}'
echo "After (Phase 3):"
tail -20 memory_phase3.log | awk '{sum+=$2; count++} END {print "Average:", sum/count, "MB"}'
```

**Expected**: 30-40% memory reduction

### Step 3.3: Increase Batch Size

If memory reduced significantly, try larger batch size:

```bash
# Test with larger batch size
NNODES=1 NPROC_PER_NODE=2 \
L2T_TRAIN_BS=6 \
L2T_BASE_STEPS=50 \
bash scripts/training/train_qwen_english_l2t_stable.sh

# If successful, try batch size 8
NNODES=1 NPROC_PER_NODE=2 \
L2T_TRAIN_BS=8 \
L2T_BASE_STEPS=50 \
bash scripts/training/train_qwen_english_l2t_stable.sh
```

## Phase 4: DDP Optimization (Day 2 - 1 hour)

### Step 4.1: Optimize DDP Configuration

**File**: `latentDLM_mmdit/train_mmdit_stable.py`

**Around line 276**, replace:
```python
# OLD
if is_distributed:
    ddp_trainer = DDP(opt_trainer, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

# NEW
if is_distributed:
    ddp_trainer = DDP(
        opt_trainer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # Faster if no unused params
        gradient_as_bucket_view=True,  # Reduce memory copies
        broadcast_buffers=False,  # Not needed without batch norm
    )
    if is_main_process:
        print("✓ DDP initialized with performance optimizations")
```

### Step 4.2: Test Multi-GPU

```bash
# Test on 2 GPUs
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=50 \
bash scripts/training/train_qwen_english_l2t_stable.sh 2>&1 | tee phase4_test.log

# Check for DDP errors
grep -i "error\|warning" phase4_test.log | grep -i "ddp\|unused"
```

**If you see "unused parameters" warning**: Set `find_unused_parameters=True`

## Phase 5: Optimizer Optimization (Day 3 - 1 hour)

### Step 5.1: Enable Fused Optimizer

**File**: `latentDLM_mmdit/optimizer.py`

Find the AdamW creation and add `fused=True`:

```python
# OLD
optimizer = torch.optim.AdamW(
    param_groups,
    lr=config.optimizer.lr,
    betas=(config.optimizer.beta1, config.optimizer.beta2),
    eps=config.optimizer.eps,
)

# NEW
try:
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
        fused=True,  # Fused kernels (20-30% faster)
    )
    print("✓ Using fused AdamW optimizer")
except Exception as e:
    print(f"⚠ Fused AdamW not available: {e}")
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
    )
```

### Step 5.2: Test Optimizer

```bash
# Test fused optimizer
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=50 \
bash scripts/training/train_qwen_english_l2t_stable.sh 2>&1 | tee phase5_test.log

# Verify fused optimizer is used
grep "fused" phase5_test.log
```

## Phase 6: Loss Function Improvement (Day 3 - 1 hour)

### Step 6.1: Improve Latent Loss

**File**: `latentDLM_mmdit/improved_trainer_stable.py`

**Around line 393-410**, replace:
```python
# OLD
latent_pred_norm = F.normalize(latent_pred, p=2, dim=-1, eps=eps)
latent_target_norm = F.normalize(latent_target, p=2, dim=-1, eps=eps)
latent_loss = F.mse_loss(latent_pred_norm, latent_target_norm)
latent_loss = torch.clamp(latent_loss, min=0.0, max=10.0)

# NEW
# Use cosine similarity loss (better for embeddings)
cosine_sim = F.cosine_similarity(latent_pred, latent_target, dim=-1, eps=eps)
latent_loss = (1.0 - cosine_sim).mean()
latent_loss = torch.clamp(latent_loss, min=0.0, max=2.0)
```

### Step 6.2: Test Loss Convergence

```bash
# Run longer test to check convergence
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=500 \
bash scripts/training/train_qwen_english_l2t_stable.sh 2>&1 | tee phase6_test.log

# Plot loss curve
python3 << 'PYTHON'
import re
import matplotlib.pyplot as plt

losses = []
with open('phase6_test.log') as f:
    for line in f:
        match = re.search(r'Loss:\s*(\d+\.\d+)', line)
        if match:
            losses.append(float(match.group(1)))

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Convergence (Phase 6)')
plt.savefig('phase6_loss_curve.png')
print(f"✓ Loss curve saved to phase6_loss_curve.png")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
PYTHON
```

## Phase 7: Checkpoint Management (Day 3 - 30 min)

### Step 7.1: Add Checkpoint Cleanup

**File**: `latentDLM_mmdit/train_mmdit_stable.py`

**After imports** (around line 95), add:
```python
def cleanup_old_checkpoints(save_dir, run_name, keep_last_n=3):
    """Keep only the N most recent checkpoints to save disk space."""
    import shutil
    checkpoint_base = Path(save_dir) / run_name
    if not checkpoint_base.exists():
        return

    checkpoint_dirs = []
    for d in checkpoint_base.iterdir():
        if d.is_dir() and d.name != 'latest':
            try:
                step_num = int(d.name.split('-')[-1].replace('k', '000').replace('M', '000000'))
                checkpoint_dirs.append((step_num, d))
            except (ValueError, IndexError):
                continue

    checkpoint_dirs.sort(key=lambda x: x[0])
    for _, old_dir in checkpoint_dirs[:-keep_last_n]:
        try:
            shutil.rmtree(old_dir)
            print(f"✓ Removed old checkpoint: {old_dir.name}")
        except Exception as e:
            print(f"✗ Failed to remove {old_dir.name}: {e}")
```

**Around line 554** (after checkpoint saving), add:
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

## Phase 8: Final Validation (Day 4)

### Step 8.1: Full Performance Test

```bash
# Run comprehensive test (1000 steps)
NNODES=1 NPROC_PER_NODE=2 \
L2T_BASE_STEPS=1000 \
bash scripts/training/train_qwen_english_l2t_stable.sh 2>&1 | tee final_test.log

# Extract final metrics
python3 << 'PYTHON'
import re
import json

metrics = {
    'losses': [],
    'throughput': [],
    'memory': [],
}

with open('final_test.log') as f:
    for line in f:
        if 'Loss:' in line:
            match = re.search(r'Loss:\s*(\d+\.\d+)', line)
            if match:
                metrics['losses'].append(float(match.group(1)))

        if 'samples_per_sec' in line:
            match = re.search(r'samples_per_sec["\']:\s*(\d+\.\d+)', line)
            if match:
                metrics['throughput'].append(float(match.group(1)))

# Calculate statistics
stats = {}
for key, values in metrics.items():
    if values:
        stats[key] = {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
        }

with open('final_metrics.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("✓ Final metrics saved to final_metrics.json")
print(json.dumps(stats, indent=2))
PYTHON
```

### Step 8.2: Compare with Baseline

```bash
# Compare all phases
python3 << 'PYTHON'
import json

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60 + "\n")

# Load baseline
with open('baseline_metrics.txt') as f:
    baseline = f.read()

# Load final
with open('final_metrics.json') as f:
    final = json.load(f)

print("Baseline throughput: [extract from baseline_metrics.txt]")
print(f"Final throughput: {final['throughput']['mean']:.2f} samples/sec")
print(f"\nImprovement: [calculate percentage]")
print("\n" + "="*60)
PYTHON
```

## Phase 9: Multi-Node Testing (Day 5)

### Step 9.1: Test 2-Node Setup

```bash
# On master node (NODE_RANK=0)
NNODES=2 NODE_RANK=0 \
MASTER_ADDR=$(hostname -I | awk '{print $1}') \
bash scripts/training/train_qwen_english_l2t_stable.sh

# On worker node (NODE_RANK=1)
NNODES=2 NODE_RANK=1 \
MASTER_ADDR=<master-ip> \
bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Step 9.2: Verify Scaling

```bash
# Compare single-node vs multi-node throughput
# Expected: ~1.8-1.9x speedup with 2 nodes (not perfect 2x due to communication)
```

## Rollback Procedure

If any phase fails:

```bash
# Restore from backup
cd ..
rm -rf MM-LDLM
cp -r $BACKUP_NAME MM-LDLM
cd MM-LDLM

# Or restore specific files
cp ../$BACKUP_NAME/train_qwen_english_l2t_stable.sh .
cp ../$BACKUP_NAME/latentDLM_mmdit/train_mmdit_stable.py latentDLM_mmdit/
# etc.
```

## Success Criteria

After all phases:
- [ ] Training speed improved by 25-35%
- [ ] Memory usage reduced by 30-40%
- [ ] No NaN gradients in 1000 steps
- [ ] Loss converging smoothly
- [ ] Multi-node training works
- [ ] Checkpoints saving correctly

## Troubleshooting

See `TROUBLESHOOTING_GUIDE.md` for common issues and solutions.
