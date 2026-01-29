# Training Script Usage Guide

## New Script: train_qwen_english_l2t_stable.sh

This is an improved version of `train_qwen_english_l2t_scaled.sh` that fixes both:
1. **Distributed training connectivity issues**
2. **NaN gradient problems**

## What's New

### Distributed Training Fixes
- ✅ Better MASTER_ADDR detection (uses IP instead of hostname)
- ✅ Pre-flight connectivity checks
- ✅ Network reachability validation
- ✅ Improved error messages
- ✅ Automatic log file creation with timestamps

### NaN Gradient Prevention
- ✅ Reduced learning rate: **5e-5** (was 1e-4)
- ✅ Reduced gradient clip: **0.5** (was 1.0)
- ✅ Increased warmup: **2000 steps** (was 1000)
- ✅ Reduced latent loss weight: **0.1** (was 1.0)
- ✅ Added gradient accumulation: **2 steps**
- ✅ Uses stable config by default: `mmdit_stable`

### Additional Improvements
- ✅ Comprehensive pre-flight checks
- ✅ Better error handling and reporting
- ✅ Automatic log file management
- ✅ Post-training summary with troubleshooting tips
- ✅ All hyperparameters can be overridden via environment variables

## Quick Start

### Test on Single Node (Recommended First Step)

```bash
# Test with 2 GPUs
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# Test with 4 GPUs
NNODES=1 NPROC_PER_NODE=4 bash scripts/training/train_qwen_english_l2t_stable.sh

# Test with all GPUs on one node
NNODES=1 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Scale to Multiple Nodes

```bash
# 2 nodes × 8 GPUs = 16 GPUs
NNODES=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# 4 nodes × 8 GPUs = 32 GPUs
NNODES=4 bash scripts/training/train_qwen_english_l2t_stable.sh

# Full scale: 8 nodes × 8 GPUs = 64 GPUs
NNODES=8 bash scripts/training/train_qwen_english_l2t_stable.sh
```

## Environment Variables

All hyperparameters can be customized via environment variables:

### Training Hyperparameters
```bash
# Learning rate (default: 5e-5)
LEARNING_RATE=3e-5 bash scripts/training/train_qwen_english_l2t_stable.sh

# Gradient clipping (default: 0.5)
GRAD_CLIP_NORM=0.3 bash scripts/training/train_qwen_english_l2t_stable.sh

# Warmup steps (default: 2000)
WARMUP_STEPS=3000 bash scripts/training/train_qwen_english_l2t_stable.sh

# Latent loss weight (default: 0.1)
LATENT_LOSS_WEIGHT=0.05 bash scripts/training/train_qwen_english_l2t_stable.sh

# Gradient accumulation (default: 2)
GRAD_ACCUM_STEPS=4 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Batch Size and Workers
```bash
# Batch size per GPU (default: 4)
L2T_TRAIN_BS=8 bash scripts/training/train_qwen_english_l2t_stable.sh

# Data workers (default: 16)
DATA_WORKERS=8 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Configuration
```bash
# Use different config (default: mmdit_stable)
L2T_CONFIG=mmdit_preprocessed bash scripts/training/train_qwen_english_l2t_stable.sh

# Custom run name (default: mmdit-qwen-32d-l2t-stable)
L2T_RUN_NAME=my-experiment bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Distributed Setup
```bash
# Custom master address
MASTER_ADDR=192.168.1.100 bash scripts/training/train_qwen_english_l2t_stable.sh

# Custom master port (default: 29500)
MASTER_PORT=12345 bash scripts/training/train_qwen_english_l2t_stable.sh

# Specific node rank (for multi-node)
NODE_RANK=0 NNODES=4 bash scripts/training/train_qwen_english_l2t_stable.sh  # Master node
NODE_RANK=1 NNODES=4 bash scripts/training/train_qwen_english_l2t_stable.sh  # Worker node 1
```

## Combining Multiple Overrides

```bash
# Conservative training (very stable)
LEARNING_RATE=3e-5 \
GRAD_CLIP_NORM=0.3 \
WARMUP_STEPS=3000 \
GRAD_ACCUM_STEPS=4 \
bash scripts/training/train_qwen_english_l2t_stable.sh

# Faster training (less stable)
LEARNING_RATE=1e-4 \
GRAD_CLIP_NORM=1.0 \
WARMUP_STEPS=1000 \
bash scripts/training/train_qwen_english_l2t_stable.sh

# Large batch training
L2T_TRAIN_BS=8 \
GRAD_ACCUM_STEPS=4 \
bash scripts/training/train_qwen_english_l2t_stable.sh
```

## Monitoring Training

### Real-time Monitoring

```bash
# Watch the latest log file
tail -f train_logs/train_*_node0.log

# Monitor for errors
tail -f train_logs/train_*_node0.log | grep -E "ERROR|NaN|✗"

# Monitor loss values
tail -f train_logs/train_*_node0.log | grep -E "Loss:|loss:"

# Monitor gradient norms
tail -f train_logs/train_*_node0.log | grep "grad_norm"
```

### Check for Issues

```bash
# Find all error messages
grep -E "ERROR|FAILED|✗" train_logs/train_*_node0.log

# Check for NaN gradients
grep "NaN" train_logs/train_*_node0.log

# Check distributed setup
grep -E "DistNetworkError|Connection" train_logs/train_*_node0.log

# View NCCL debug log
cat /tmp/nccl_debug_0.log
```

### Post-Training Analysis

```bash
# Count total steps completed
grep "step:" train_logs/train_*_node0.log | wc -l

# Extract loss values
grep "loss:" train_logs/train_*_node0.log | awk '{print $NF}' > losses.txt

# Plot loss curve
python -c "
import matplotlib.pyplot as plt
losses = [float(x.strip()) for x in open('losses.txt')]
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.yscale('log')
plt.savefig('loss_curve.png')
print(f'Plotted {len(losses)} steps')
"
```

## Pre-flight Checks

The script automatically runs these checks before training:

1. ✓ GPU availability and count
2. ✓ Data directory existence (tokens and latents)
3. ✓ Config file existence
4. ✓ Master node reachability (for worker nodes)
5. ✓ PyTorch and NCCL versions

If any check fails, the script will exit with an error message.

## Expected Output

### Successful Start
```
==========================================
Training Configuration
==========================================
Distributed Setup:
  Node Rank: 0
  NNODES: 1
  NPROC_PER_NODE: 2
  GLOBAL_WORLD_SIZE: 2
  ...

Running pre-flight checks...
✓ Found 8 GPUs
✓ Token directory exists: ...
✓ Latent directory exists: ...
✓ Config file exists: ...
✓ PyTorch version: 2.x.x
✓ CUDA available: True

Pre-flight checks completed!
==========================================

Launching training...
```

### During Training
```
Training: 100%|████████| 1000/1000 [10:00<00:00, 1.67it/s]
Epoch: 0/10, Progress: 10.0%, Loss: 2.3456, Text: 2.1234, Latent: 0.2222, Acc: 0.45
```

### Successful Completion
```
==========================================
✓ Training completed successfully!
==========================================

Summary:
  Exit code: 0
  Log file: train_logs/train_20260129_120000_node0.log
  NCCL debug: /tmp/nccl_debug_0.log
  Checkpoints: /path/to/saved/mmdit-qwen-32d-l2t-stable/
```

## Troubleshooting

### Issue: "ERROR: Could not determine MASTER_ADDR"
**Solution**: Set it explicitly
```bash
MASTER_ADDR=192.168.1.100 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Issue: "Cannot connect to master port"
**Solution**: Check firewall or use different port
```bash
MASTER_PORT=12345 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Issue: Still getting NaN gradients
**Solution**: Reduce learning rate further
```bash
LEARNING_RATE=3e-5 GRAD_CLIP_NORM=0.3 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Issue: Out of memory
**Solution**: Reduce batch size or increase gradient accumulation
```bash
L2T_TRAIN_BS=2 GRAD_ACCUM_STEPS=4 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Issue: Training too slow
**Solution**: Increase batch size or reduce gradient accumulation
```bash
L2T_TRAIN_BS=8 GRAD_ACCUM_STEPS=1 bash scripts/training/train_qwen_english_l2t_stable.sh
```

## Comparison with Original Script

| Feature | Original | New (Stable) |
|---------|----------|--------------|
| Learning rate | 1e-4 | 5e-5 (50% lower) |
| Gradient clip | 1.0 | 0.5 (50% lower) |
| Warmup steps | 1000 | 2000 (2x more) |
| Latent loss weight | 1.0 | 0.1 (10x lower) |
| Gradient accumulation | None | 2 steps |
| Pre-flight checks | None | Comprehensive |
| Error handling | Basic | Advanced |
| Log management | Manual | Automatic |
| Config | mmdit_preprocessed | mmdit_stable |
| MASTER_ADDR detection | Hostname | IP address |
| Connectivity checks | None | Yes |

## Best Practices

1. **Always test on single node first**
   ```bash
   NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh
   ```

2. **Monitor for first 1000 steps**
   - Watch for NaN gradients
   - Check loss is decreasing
   - Verify gradient norms < 1.0

3. **Save checkpoints frequently**
   - Default: every 6250 steps (scaled)
   - Can override with `L2T_BASE_SAVE_FREQ`

4. **Scale up gradually**
   - 1 node → 2 nodes → 4 nodes → 8 nodes
   - Verify stability at each scale

5. **Keep logs organized**
   - Logs are automatically timestamped
   - One log file per node per run

## Next Steps

1. **Apply code patches** (if not done yet)
   - See `results/archive/TRAIN_MMDIT_PATCH.py`
   - See `results/archive/IMPROVED_TRAINER_PATCH.py`

2. **Test the script**
   ```bash
   NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh
   ```

3. **Monitor closely for 1000 steps**
   ```bash
   tail -f train_logs/train_*_node0.log | grep -E "Loss:|ERROR|NaN"
   ```

4. **Scale up if stable**
   ```bash
   NNODES=2 bash scripts/training/train_qwen_english_l2t_stable.sh
   ```

5. **Report results**
   - Share log files if issues occur
   - Note any error messages
   - Check NCCL debug logs

## Support

If you encounter issues:
1. Check the log file for detailed error messages
2. Review pre-flight check output
3. Verify all code patches were applied
4. Try more conservative hyperparameters
5. Test on single node first

Good luck with your training! 🚀
