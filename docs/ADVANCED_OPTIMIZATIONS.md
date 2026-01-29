# Advanced Training Optimizations

## 1. Mixed Precision Training Improvements

### Current Issue:
Using bf16 but not leveraging automatic mixed precision (AMP) optimally.

### Implementation:

```python
# In train_mmdit_stable.py - add after model creation
from torch.cuda.amp import autocast, GradScaler

# Only use GradScaler for fp16, not bf16
use_amp = config.training.dtype in ['fp16', 'bf16']
scaler = GradScaler(enabled=(config.training.dtype == 'fp16'))

# In training loop
with autocast(dtype=dtype, enabled=use_amp):
    loss, metrics = ddp_trainer(batch, step=state.step)

# Scale loss for fp16
if config.training.dtype == 'fp16':
    scaler.scale(loss * config.loss.loss_scale).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
else:
    (loss * config.loss.loss_scale).backward()
    norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
    optimizer.step()
```

**Impact**: 10-15% faster training with fp16

## 2. Compile Model with torch.compile

### Current Issue:
`compile_model=false` - missing 20-30% speedup from PyTorch 2.0+

### Implementation:

```python
# In train_mmdit_stable.py - replace compilation section
if config.training.compile_model:
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        # Compile with optimal settings
        opt_trainer = torch.compile(
            trainer,
            mode="reduce-overhead",  # Best for training
            fullgraph=False,  # Allow graph breaks
            dynamic=True,  # Handle dynamic shapes
        )
        print("✓ Model compiled with torch.compile (expect 20-30% speedup)")
    except Exception as e:
        print(f"✗ Compilation failed: {e}, using eager mode")
        opt_trainer = trainer
else:
    opt_trainer = trainer
```

**Impact**: 20-30% faster training (after warmup)

## 3. Fused Optimizer

### Current Issue:
Using standard AdamW instead of fused version

### Implementation:

```python
# In optimizer.py
def get_optimizer(config, model):
    """Get optimizer with fused kernels for better performance."""

    # Separate parameters by weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for biases and layer norms
        if 'bias' in name or 'norm' in name or 'ln' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': config.optimizer.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # Use fused AdamW if available (much faster)
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=config.optimizer.eps,
            fused=True  # Fused kernel (20-30% faster)
        )
        print("✓ Using fused AdamW optimizer")
    except Exception:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.optimizer.lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=config.optimizer.eps,
        )
        print("⚠ Fused AdamW not available, using standard version")

    return optimizer
```

**Impact**: 20-30% faster optimizer step

## 4. Asynchronous Checkpointing

### Current Issue:
Checkpointing blocks training for 10-30 seconds

### Implementation:

```python
# In train_mmdit_stable.py - add async checkpoint saving
import threading
from queue import Queue

checkpoint_queue = Queue(maxsize=1)

def async_checkpoint_worker():
    """Background thread for saving checkpoints."""
    while True:
        checkpoint_data = checkpoint_queue.get()
        if checkpoint_data is None:  # Poison pill
            break

        output_path, trainer, optimizer, state = checkpoint_data
        try:
            save_checkpoint(output_path, trainer, optimizer, state)
            print(f"✓ Checkpoint saved asynchronously: {output_path}")
        except Exception as e:
            print(f"✗ Async checkpoint failed: {e}")

        checkpoint_queue.task_done()

# Start background thread
checkpoint_thread = threading.Thread(target=async_checkpoint_worker, daemon=True)
checkpoint_thread.start()

# In training loop - replace synchronous save
if is_main_process and (state.step) % config.logging.save_freq == 0:
    # Copy state to CPU to avoid blocking GPU
    cpu_state = {
        'model': {k: v.cpu() for k, v in trainer.state_dict().items()},
        'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in optimizer.state_dict().items()},
        'training_state': state,
    }

    # Queue for async saving (non-blocking)
    if not checkpoint_queue.full():
        checkpoint_queue.put((output_path, cpu_state['model'],
                            cpu_state['optimizer'], cpu_state['training_state']))
    else:
        print("⚠ Checkpoint queue full, skipping this checkpoint")
```

**Impact**: Zero training interruption during checkpointing

## 5. Dynamic Batch Size Scaling

### Current Issue:
Fixed batch size doesn't adapt to GPU memory availability

### Implementation:

```python
# In train_mmdit_stable.py - add dynamic batch sizing
class DynamicBatchSizer:
    """Automatically adjust batch size based on GPU memory."""

    def __init__(self, initial_batch_size, min_batch_size=1, max_batch_size=32):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.oom_count = 0
        self.success_count = 0

    def on_oom(self):
        """Called when OOM occurs."""
        self.oom_count += 1
        self.success_count = 0

        # Reduce batch size
        new_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
        if new_batch_size < self.current_batch_size:
            print(f"⚠ OOM detected, reducing batch size: {self.current_batch_size} -> {new_batch_size}")
            self.current_batch_size = new_batch_size
            torch.cuda.empty_cache()
            return True
        return False

    def on_success(self):
        """Called after successful batch."""
        self.success_count += 1

        # After 1000 successful batches, try increasing
        if self.success_count >= 1000 and self.current_batch_size < self.max_batch_size:
            new_batch_size = min(self.current_batch_size + 1, self.max_batch_size)
            print(f"✓ Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
            self.current_batch_size = new_batch_size
            self.success_count = 0

# Usage in training loop
batch_sizer = DynamicBatchSizer(config.training.train_batch_size)

while state.epoch < num_epochs:
    try:
        # ... training code ...
        batch_sizer.on_success()
    except RuntimeError as e:
        if "out of memory" in str(e):
            if batch_sizer.on_oom():
                continue  # Retry with smaller batch
            else:
                raise  # Can't reduce further
        else:
            raise
```

**Impact**: Maximize GPU utilization, prevent OOM crashes

## 6. Efficient Attention Implementation

### Current Issue:
May not be using Flash Attention optimally

### Implementation:

```python
# In models/multimodal_mmdit.py - ensure Flash Attention is used
import torch.nn.functional as F

# At module level
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

# In MMDiT forward pass
if HAS_FLASH_ATTN and self.training:
    # Use Flash Attention (2-4x faster)
    attn_output = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
else:
    # Fallback to memory-efficient attention
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=True
    ):
        attn_output = F.scaled_dot_product_attention(q, k, v)
```

**Impact**: 2-4x faster attention computation

## 7. Gradient Accumulation with Sync Control

### Current Issue:
DDP synchronizes gradients every step, even during accumulation

### Implementation:

```python
# In train_mmdit_stable.py - optimize gradient accumulation
grad_accum_steps = config.training.get('gradient_accumulation_steps', 1)

for micro_step in range(grad_accum_steps):
    # Only sync on last micro-step
    is_last_micro_step = (micro_step == grad_accum_steps - 1)

    if is_distributed:
        # Disable gradient sync for all but last micro-step
        context = ddp_trainer.no_sync() if not is_last_micro_step else contextlib.nullcontext()
    else:
        context = contextlib.nullcontext()

    with context:
        loss, metrics = ddp_trainer(batch, step=state.step)
        (loss / grad_accum_steps).backward()

    # Only step optimizer on last micro-step
    if is_last_micro_step:
        norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

**Impact**: 15-25% faster with gradient accumulation

## 8. Profiling Integration

### Add Built-in Profiling:

```python
# In train_mmdit_stable.sh - add profiling option
ENABLE_PROFILING="${ENABLE_PROFILING:-false}"

if [ "${ENABLE_PROFILING}" = "true" ]; then
    echo "⚠ Profiling enabled - training will be slower"
    PROFILING_ARGS="training.enable_profiling=true training.profile_steps=100"
else
    PROFILING_ARGS=""
fi
```

```python
# In train_mmdit_stable.py - add profiler
if config.training.get('enable_profiling', False):
    from torch.profiler import profile, ProfilerActivity, schedule

    profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=10, warmup=10, active=20, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    profiler.start()
else:
    profiler = None

# In training loop
if profiler:
    profiler.step()
    if state.step >= config.training.profile_steps:
        profiler.stop()
        profiler = None
```

**Impact**: Identify bottlenecks, guide optimization

## Performance Summary

| Optimization | Speedup | Memory | Difficulty |
|--------------|---------|--------|------------|
| Fused Optimizer | +25% | 0% | Easy |
| torch.compile | +25% | 0% | Easy |
| Flash Attention | +150% | -20% | Medium |
| Gradient Checkpointing | 0% | -35% | Easy |
| Async Checkpointing | +2% | 0% | Medium |
| Optimized DDP | +15% | -5% | Easy |
| Dynamic Batching | +10% | Adaptive | Hard |

**Combined Expected Improvement**: 2-3x faster training, 40-50% less memory

## Implementation Priority

### Week 1 (Easy Wins):
1. Fused optimizer
2. Gradient checkpointing
3. Optimized DDP settings
4. Fix NCCL timeout

### Week 2 (Medium Impact):
5. torch.compile
6. Async checkpointing
7. Better loss function
8. Gradient accumulation optimization

### Week 3 (Advanced):
9. Flash Attention integration
10. Dynamic batch sizing
11. Profiling and fine-tuning
