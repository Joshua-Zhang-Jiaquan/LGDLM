# Troubleshooting Guide

This guide covers common issues you may encounter during training optimization and migration.

## Issue 1: "find_unused_parameters" Error

### Symptoms:
```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
```

### Cause:
DDP with `find_unused_parameters=False` but model has unused parameters in some forward passes.

### Solution:

**Option A**: Set `find_unused_parameters=True` (slower but safer)
```python
ddp_trainer = DDP(
    opt_trainer,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True,  # Set to True
    gradient_as_bucket_view=True,
    broadcast_buffers=False,
)
```

**Option B**: Fix the model to use all parameters (better)
```python
# In improved_trainer_stable.py - ensure all parameters are used
# Check which parameters are unused:
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"Unused parameter: {name}")
```

## Issue 2: Out of Memory (OOM)

### Symptoms:
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

### Diagnosis:
```bash
# Check current memory usage
nvidia-smi

# Monitor memory during training
nvidia-smi dmon -s mu -c 100
```

### Solutions (in order of preference):

**1. Enable gradient checkpointing** (if not already):
```yaml
# In mmdit_stable.yaml
model:
  use_gradient_checkpointing: true
```

**2. Reduce batch size**:
```bash
L2T_TRAIN_BS=2 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**3. Increase gradient accumulation**:
```bash
GRAD_ACCUM_STEPS=4 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**4. Reduce sequence length**:
```yaml
# In config
data:
  max_length: 256  # Reduced from 512
```

**5. Use mixed precision** (if not already):
```bash
DTYPE=fp16 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**6. Reduce model size**:
```yaml
model:
  hidden_size: 768  # Reduced from 1024
  n_blocks: 12      # Reduced from 24
```

## Issue 3: NaN Gradients

### Symptoms:
```
WARNING: NaN gradient detected at step X
Loss: nan
```

### Diagnosis:
```bash
# Check where NaN first appears
grep -n "NaN\|nan" train_logs/train_*.log | head -20

# Check loss values before NaN
grep "Loss:" train_logs/train_*.log | tail -50
```

### Solutions:

**1. Reduce learning rate**:
```bash
LEARNING_RATE=3e-5 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**2. Reduce gradient clip norm**:
```bash
GRAD_CLIP_NORM=0.3 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**3. Increase warmup steps**:
```bash
WARMUP_STEPS=3000 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**4. Check for bad data**:
```python
# Add to trainer forward pass
if torch.isnan(batch['input_ids']).any():
    print("WARNING: NaN in input_ids")
    continue

if torch.isnan(batch['latent']).any():
    print("WARNING: NaN in latents")
    continue
```

**5. Use more conservative loss weights**:
```bash
LATENT_LOSS_WEIGHT=0.05 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**6. Enable loss scaling** (for fp16):
```python
# In train_mmdit_stable.py
from torch.cuda.amp import GradScaler
scaler = GradScaler(enabled=(dtype == torch.float16))
```

## Issue 4: Slow Data Loading

### Symptoms:
- GPU utilization < 70%
- Long pauses between steps
- High CPU usage

### Diagnosis:
```bash
# Monitor GPU utilization
nvidia-smi dmon -s u -c 100

# Check data loading time
# Add timing in train_mmdit_stable.py:
import time
data_start = time.time()
batch = next(batch_iterator)
data_time = time.time() - data_start
print(f"Data loading time: {data_time:.3f}s")
```

### Solutions:

**1. Optimize num_workers**:
```bash
# Try different values
DATA_WORKERS=4 bash scripts/training/train_qwen_english_l2t_stable.sh
DATA_WORKERS=8 bash scripts/training/train_qwen_english_l2t_stable.sh
DATA_WORKERS=12 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**2. Enable prefetching**:
```python
# In data_simple.py
train_loader = DataLoader(
    ...,
    prefetch_factor=4,  # Prefetch 4 batches per worker
    persistent_workers=True,
)
```

**3. Use faster storage**:
```bash
# Move data to SSD if on HDD
rsync -av --progress /hdd/data/ /ssd/data/
```

**4. Reduce data preprocessing**:
```python
# Cache preprocessed data
# Use memory-mapped files for large datasets
```

## Issue 5: Distributed Training Hangs

### Symptoms:
- Training starts but hangs at initialization
- "Waiting for all processes" message
- No progress for > 5 minutes

### Diagnosis:
```bash
# Check NCCL debug logs
cat /tmp/nccl_debug_0.log

# Check network connectivity
ping <master_node_ip>
nc -zv <master_node_ip> 29500

# Check if processes are running
ps aux | grep python
```

### Solutions:

**1. Verify environment variables**:
```bash
# On all nodes, check:
echo $MASTER_ADDR
echo $MASTER_PORT
echo $RANK
echo $WORLD_SIZE
echo $LOCAL_RANK
```

**2. Check firewall**:
```bash
# On master node, allow port
sudo ufw allow 29500/tcp

# Or disable firewall temporarily
sudo ufw disable
```

**3. Set network interface explicitly**:
```bash
# In train_qwen_english_l2t_stable.sh
export NCCL_SOCKET_IFNAME=eth0  # or your network interface
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

**4. Increase timeout**:
```bash
# In train_qwen_english_l2t_stable.sh
export NCCL_TIMEOUT=3600  # 1 hour for slow networks
```

**5. Use TCP instead of IB**:
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

**6. Test with single node first**:
```bash
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh
```

## Issue 6: Checkpoint Loading Fails

### Symptoms:
```
RuntimeError: Error(s) in loading state_dict
KeyError: 'model.text_encoder.weight'
```

### Diagnosis:
```bash
# Check checkpoint contents
python3 << 'PYTHON'
import torch
ckpt = torch.load('path/to/checkpoint/model.pt', map_location='cpu')
print("Keys in checkpoint:", ckpt.keys())
print("Model keys:", [k for k in ckpt.get('model', {}).keys()][:10])
PYTHON
```

### Solutions:

**1. Check model architecture matches**:
```python
# Verify config matches checkpoint
print(f"Config latent_dim: {config.model.latent_dim}")
print(f"Checkpoint latent_dim: {checkpoint['config'].model.latent_dim}")
```

**2. Use strict=False for partial loading**:
```python
model.load_state_dict(checkpoint['model'], strict=False)
```

**3. Remove DDP wrapper prefix**:
```python
# If checkpoint saved with DDP
state_dict = checkpoint['model']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict)
```

## Issue 7: Loss Not Decreasing

### Symptoms:
- Loss stays constant or increases
- Text accuracy remains low
- No improvement after 1000+ steps

### Diagnosis:
```bash
# Plot loss curve
python3 << 'PYTHON'
import re
import matplotlib.pyplot as plt

losses = []
with open('train_logs/train_*.log') as f:
    for line in f:
        match = re.search(r'Loss:\s*(\d+\.\d+)', line)
        if match:
            losses.append(float(match.group(1)))

plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
PYTHON
```

### Solutions:

**1. Check learning rate**:
```bash
# Increase if too low
LEARNING_RATE=1e-4 bash scripts/training/train_qwen_english_l2t_stable.sh

# Decrease if too high (causing instability)
LEARNING_RATE=3e-5 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**2. Verify data is correct**:
```python
# Check a few samples
for batch in train_loader:
    print("Input IDs shape:", batch['input_ids'].shape)
    print("Latent shape:", batch['latent'].shape)
    print("Sample text:", tokenizer.decode(batch['input_ids'][0]))
    break
```

**3. Check parameter freezing**:
```python
# Verify correct parameters are trainable
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
```

**4. Verify loss computation**:
```python
# Add debug prints in trainer
print(f"Text loss: {text_loss.item():.4f}")
print(f"Latent loss: {latent_loss.item():.4f}")
print(f"Combined loss: {loss.item():.4f}")
```

**5. Check for mode mismatch**:
```bash
# Ensure training mode matches config
# For L2T: latent should be clean, text should be noisy
# For T2L: text should be clean, latent should be noisy
```

## Issue 8: Compilation Errors (torch.compile)

### Symptoms:
```
torch._dynamo.exc.BackendCompilerFailed
```

### Solutions:

**1. Disable compilation**:
```bash
L2T_COMPILE=false bash scripts/training/train_qwen_english_l2t_stable.sh
```

**2. Use different backend**:
```python
opt_trainer = torch.compile(trainer, backend="inductor")
# or
opt_trainer = torch.compile(trainer, backend="aot_eager")
```

**3. Allow graph breaks**:
```python
opt_trainer = torch.compile(trainer, fullgraph=False)
```

**4. Update PyTorch**:
```bash
pip install --upgrade torch
```

## Issue 9: Fused Optimizer Not Available

### Symptoms:
```
TypeError: AdamW.__init__() got an unexpected keyword argument 'fused'
```

### Solutions:

**1. Update PyTorch**:
```bash
pip install --upgrade torch>=2.0
```

**2. Use fallback**:
```python
try:
    optimizer = torch.optim.AdamW(..., fused=True)
except TypeError:
    optimizer = torch.optim.AdamW(...)  # Without fused
```

## Issue 10: Gradient Checkpointing Slower

### Symptoms:
- Training slower after enabling gradient checkpointing
- Expected memory savings but no speed improvement

### Explanation:
Gradient checkpointing trades compute for memory. It's slower but uses less memory.

### Solutions:

**1. Only use if memory-constrained**:
```yaml
# Disable if you have enough memory
model:
  use_gradient_checkpointing: false
```

**2. Increase batch size to compensate**:
```bash
# Use saved memory for larger batches
L2T_TRAIN_BS=8 bash scripts/training/train_qwen_english_l2t_stable.sh
```

**3. Selective checkpointing**:
```python
# Only checkpoint some layers
if layer_idx % 2 == 0:  # Every other layer
    output = checkpoint(layer, input)
```

## Quick Diagnostic Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check training logs for errors
grep -i "error\|warning\|nan" train_logs/train_*.log

# Check NCCL logs
cat /tmp/nccl_debug_0.log | grep -i "error\|warn"

# Monitor training progress
tail -f train_logs/train_*_node0.log | grep "Loss:"

# Check disk space
df -h

# Check memory usage
free -h

# Check process status
ps aux | grep python | grep train_mmdit

# Check network connectivity (multi-node)
ping <master_node_ip>
nc -zv <master_node_ip> 29500
```

## Getting Help

If issues persist:

1. **Collect information**:
   ```bash
   # Save all relevant logs
   tar -czf debug_info.tar.gz \
       train_logs/ \
       /tmp/nccl_debug_*.log \
       train_qwen_english_l2t_stable.sh \
       latentDLM_mmdit/configs/mmdit_stable.yaml
   ```

2. **Check documentation**:
   - TRAINING_IMPROVEMENTS.md
   - MIGRATION_GUIDE.md
   - ADVANCED_OPTIMIZATIONS.md

3. **Minimal reproduction**:
   ```bash
   # Try simplest possible setup
   NNODES=1 NPROC_PER_NODE=1 \
   L2T_TRAIN_BS=1 \
   L2T_BASE_STEPS=10 \
   bash scripts/training/train_qwen_english_l2t_stable.sh
   ```

4. **Rollback to working version**:
   ```bash
   # Restore from backup
   cp -r ../backup_YYYYMMDD_HHMMSS/* .
   ```
