# Distributed Training Debug Summary

## What Happened

Your training job (`train_qwen_english_l2t_scaled.sh`) failed during distributed initialization across 8 nodes (64 GPUs total). The error was:

```
torch.distributed.DistNetworkError: Failed to recv, got 0 bytes.
Connection was likely closed. Did the remote server shutdown or crash?
```

**Key finding**: Training never started - the failure occurred during the process group initialization phase before any actual training steps.

## Root Cause Analysis

The most likely causes are:

1. **Network connectivity issues** - Worker nodes cannot reach the master node
2. **Firewall blocking** - Port 23456 is blocked between nodes
3. **Hostname resolution** - Worker nodes cannot resolve the master hostname
4. **NCCL configuration** - Network interface or InfiniBand issues
5. **Transient failures** - Temporary network glitches during startup

## What I've Created for You

### 1. Diagnostic Tools

**`test_distributed.py`** - Comprehensive diagnostic script that tests:
- Environment variables and system info
- Network connectivity to master node
- CUDA device setup
- Distributed process group initialization
- Collective operations (barrier, all_reduce, broadcast)
- Data path accessibility

**`scripts/utils/test_single_node.sh`** - Test distributed training on a single node
- Simpler test to isolate multi-node vs code issues
- Tests with 2 GPUs by default (configurable)
- Good starting point for debugging

**`scripts/utils/test_distributed_setup.sh`** - Test full multi-node setup
- Uses same configuration as your training script
- Must be run on each node with appropriate NODE_RANK

### 2. Improved Training Script

**`train_qwen_english_l2t_scaled_improved.sh`** - Enhanced version with:
- Better MASTER_ADDR detection (uses IP instead of hostname)
- Pre-flight checks (GPU availability, data paths, network connectivity)
- Clearer error messages
- Optional retry logic (set MAX_RETRIES=3 for automatic retries)
- Reduced NCCL debug verbosity (WARN instead of INFO)
- Network connectivity tests before training

### 3. Documentation

**`DISTRIBUTED_DEBUG_GUIDE.md`** - Complete debugging guide with:
- Step-by-step diagnostic procedures
- Common issues and solutions
- Configuration recommendations
- Quick reference commands

## Recommended Action Plan

### Step 1: Test Single Node First (5 minutes)

This will tell you if the issue is with multi-node communication or the training code itself:

```bash
cd /inspire/hdd/project/project-public/zhangjiaquan-253108540222/jiaquan/latent/MM-LDLM

# Test with 2 GPUs
bash scripts/utils/test_single_node.sh

# Or test with more GPUs
NPROC_PER_NODE=4 bash scripts/utils/test_single_node.sh
```

**Expected outcome**:
- ✓ If this passes → Issue is with multi-node communication (proceed to Step 2)
- ✗ If this fails → Issue is with CUDA/training code (check NCCL logs)

### Step 2: Check NCCL Debug Logs

```bash
# View the NCCL debug log
cat /tmp/nccl_debug_single_node.log

# Look for errors related to:
# - Network interfaces
# - Socket creation
# - Connection timeouts
```

### Step 3: Test Multi-Node (if single-node passed)

Run the diagnostic on 2 nodes first:

```bash
# On master node (node 0):
NODE_RANK=0 NNODES=2 bash scripts/utils/test_distributed_setup.sh

# On worker node (node 1):
NODE_RANK=1 NNODES=2 MASTER_ADDR=<master_ip> bash scripts/utils/test_distributed_setup.sh
```

### Step 4: Use Improved Training Script

Once diagnostics pass, use the improved training script:

```bash
# Single node test
NNODES=1 bash scripts/training/train_qwen_english_l2t_scaled_improved.sh

# Two nodes
NNODES=2 bash scripts/training/train_qwen_english_l2t_scaled_improved.sh

# Full 8 nodes (with retry logic)
NNODES=8 MAX_RETRIES=3 bash scripts/training/train_qwen_english_l2t_scaled_improved.sh
```

## Quick Fixes to Try

### Fix 1: Use IP Address for Master

If hostname resolution is the issue:

```bash
# Find master node IP
hostname -I

# Use it explicitly
MASTER_ADDR="192.168.x.x" bash scripts/training/train_qwen_english_l2t_scaled_improved.sh
```

### Fix 2: Change Master Port

If port 23456 is blocked:

```bash
MASTER_PORT=29500 bash scripts/training/train_qwen_english_l2t_scaled_improved.sh
```

### Fix 3: Set Network Interface

If NCCL can't find the right interface:

```bash
# Find your network interface
ip addr show

# Set it explicitly (edit the improved script and uncomment these lines):
export NCCL_SOCKET_IFNAME=eth0  # or your interface name
export NCCL_IB_DISABLE=1  # if InfiniBand is not available
```

### Fix 4: Increase Timeout

Edit `latentDLM_mmdit/train_mmdit.py` line 123:

```python
# Change from:
timeout=datetime.timedelta(minutes=30),

# To:
timeout=datetime.timedelta(minutes=60),
```

## Files Created

```
MM-LDLM/
├── test_distributed.py                      # Diagnostic script
├── test_single_node.sh                      # Single-node test
├── test_distributed_setup.sh                # Multi-node test
├── train_qwen_english_l2t_scaled_improved.sh # Improved training script
├── DISTRIBUTED_DEBUG_GUIDE.md               # Complete guide
└── DISTRIBUTED_DEBUG_SUMMARY.md             # This file
```

## What to Report Back

After running the diagnostics, please share:

1. **Single-node test results**: Did it pass or fail?
2. **NCCL debug log**: Contents of `/tmp/nccl_debug_single_node.log`
3. **Error messages**: Any specific errors from the diagnostic output
4. **Network setup**: How are your nodes connected? (Ethernet, InfiniBand, etc.)
5. **Cluster type**: Are you using SLURM, Kubernetes, or custom job scheduler?

## Additional Resources

- **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- **PyTorch Distributed**: https://pytorch.org/tutorials/intermediate/dist_tuto.html
- **Debugging Guide**: See `DISTRIBUTED_DEBUG_GUIDE.md` for detailed troubleshooting

## Contact Points

If the issue persists after trying these steps:

1. Check with your cluster administrator about:
   - Network configuration between nodes
   - Firewall rules for inter-node communication
   - Available network interfaces (InfiniBand, Ethernet)

2. Verify all nodes have:
   - Same PyTorch version
   - Same CUDA version
   - Same NCCL version
   - Access to the same data paths

## Next Steps

**Start here**: Run `bash scripts/utils/test_single_node.sh` and report the results.

This will immediately tell us whether the issue is with multi-node communication or something else.
