# Distributed Training Debugging Guide

## Problem Summary

Your training job failed during distributed initialization with `torch.distributed.DistNetworkError: Failed to recv, got 0 bytes. Connection was likely closed.`

The job configuration:
- **8 nodes × 8 GPUs = 64 GPUs total**
- **Master node**: job-13ee63a8-92fd-4de6-901f-cce230f05688-worker-0-0
- **Master port**: 23456
- **Training never started** - failed during process group initialization

## Root Causes

Based on the log analysis, the likely issues are:

1. **Network connectivity problems** between nodes
2. **Firewall blocking** port 23456 between nodes
3. **Master node unreachable** from worker nodes
4. **NCCL timeout** during initialization (despite aggressive timeout settings)
5. **Possible node failures** during startup

## Diagnostic Steps

### Step 1: Test Single-Node Setup First

Start with a simpler single-node test to isolate whether the issue is with multi-node communication or the training code itself:

```bash
# Test with 2 GPUs on a single node
bash scripts/utils/test_single_node.sh

# Or specify number of GPUs
NPROC_PER_NODE=4 bash scripts/utils/test_single_node.sh
```

**Expected outcome**: If this works, the issue is specifically with multi-node communication.

### Step 2: Test Multi-Node Communication

If single-node works, test the full multi-node setup:

```bash
# Run this on each node with appropriate NODE_RANK
# Node 0 (master):
NODE_RANK=0 NNODES=8 bash scripts/utils/test_distributed_setup.sh

# Node 1:
NODE_RANK=1 NNODES=8 bash scripts/utils/test_distributed_setup.sh

# ... and so on for all 8 nodes
```

**What to check**:
- Can each node resolve the master hostname?
- Is port 23456 accessible from all nodes?
- Do all nodes have the same PyTorch/NCCL versions?
- Are NCCL debug logs showing connection attempts?

### Step 3: Check NCCL Debug Logs

After running tests, check the NCCL debug logs:

```bash
# Single node test
cat /tmp/nccl_debug_single_node.log

# Multi-node test (check on each node)
cat /tmp/nccl_debug_0.log  # Node 0
cat /tmp/nccl_debug_1.log  # Node 1
# etc.
```

Look for:
- Connection timeout errors
- Socket creation failures
- Network interface issues
- Version mismatches

## Common Issues and Solutions

### Issue 1: Master Node Hostname Not Resolvable

**Symptom**: `socket.gaierror` or hostname resolution failures

**Solution**:
```bash
# Use IP address instead of hostname
MASTER_ADDR="192.168.x.x" bash scripts/training/train_qwen_english_l2t_scaled.sh
```

Or add hostname to `/etc/hosts` on all nodes:
```bash
echo "192.168.x.x job-13ee63a8-92fd-4de6-901f-cce230f05688-worker-0-0" >> /etc/hosts
```

### Issue 2: Port Blocked by Firewall

**Symptom**: Connection timeout or "connection refused"

**Solution**:
```bash
# Test port connectivity from worker nodes
nc -zv <master_addr> 23456

# If blocked, open the port (requires admin access)
sudo firewall-cmd --add-port=23456/tcp --permanent
sudo firewall-cmd --reload
```

### Issue 3: NCCL Network Interface Issues

**Symptom**: NCCL can't find the right network interface

**Solution**: Explicitly set the network interface:
```bash
# Find your network interface
ip addr show

# Set in training script (add to environment variables section)
export NCCL_SOCKET_IFNAME=eth0  # or your interface name
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

### Issue 4: Inconsistent Environment Across Nodes

**Symptom**: Some nodes fail while others succeed

**Solution**:
- Ensure all nodes have the same PyTorch version
- Ensure all nodes have the same CUDA version
- Ensure all nodes can access the same data paths
- Synchronize system clocks (NTP)

### Issue 5: TCPStore Timeout

**Symptom**: `recvValueWithTimeout failed`

**Solution**: Increase the timeout in `train_mmdit.py`:

```python
# In _init_distributed() function, change:
init_kwargs = dict(
    backend="nccl",
    timeout=datetime.timedelta(minutes=60),  # Increase from 30 to 60
    init_method="env://",
)
```

## Recommended Fixes for Your Training Script

### Fix 1: Add Better Error Handling

Add this to `train_qwen_english_l2t_scaled.sh` before the torchrun command:

```bash
# Verify master node is reachable
if ! ping -c 1 -W 5 "${MASTER_ADDR}" >/dev/null 2>&1; then
  echo "ERROR: Cannot reach master node at ${MASTER_ADDR}"
  exit 1
fi

# Verify port is open (on non-master nodes)
if [ "${NODE_RANK}" -ne 0 ]; then
  if ! timeout 5 bash -c "cat < /dev/null > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
    echo "ERROR: Cannot connect to master port ${MASTER_ADDR}:${MASTER_PORT}"
    exit 1
  fi
fi
```

### Fix 2: Use IP Address Instead of Hostname

Modify the MASTER_ADDR detection in your training script:

```bash
# Replace this line:
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}' | head -n1)}"

# With explicit IP or better detection:
if [ -z "${MASTER_ADDR:-}" ]; then
  # Try to get the primary IP address
  MASTER_ADDR=$(ip route get 1 | awk '{print $7; exit}')
  if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
  fi
fi
```

### Fix 3: Add Retry Logic

Add retry logic for transient network issues:

```bash
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  echo "Training attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"

  if torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    latentDLM_mmdit/train_mmdit.py \
    [... rest of args ...]; then
    echo "Training completed successfully"
    exit 0
  else
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      echo "Training failed, retrying in 30 seconds..."
      sleep 30
    fi
  fi
done

echo "Training failed after $MAX_RETRIES attempts"
exit 1
```

### Fix 4: Reduce Initial Scale

Start with fewer nodes to verify the setup works:

```bash
# Test with 2 nodes first
NNODES=2 bash scripts/training/train_qwen_english_l2t_scaled.sh

# Then scale up to 4 nodes
NNODES=4 bash scripts/training/train_qwen_english_l2t_scaled.sh

# Finally use all 8 nodes
NNODES=8 bash scripts/training/train_qwen_english_l2t_scaled.sh
```

## Alternative: Use SLURM or Other Job Scheduler

If your cluster uses SLURM, consider using it for better multi-node coordination:

```bash
#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00

# SLURM automatically sets up the distributed environment
srun python latentDLM_mmdit/train_mmdit.py [args...]
```

## Next Steps

1. **Run single-node test**: `bash scripts/utils/test_single_node.sh`
2. **If single-node works**: Test with 2 nodes, then scale up
3. **If single-node fails**: Check CUDA setup and training code
4. **Check NCCL logs**: Review `/tmp/nccl_debug_*.log` files
5. **Contact cluster admin**: If network/firewall issues persist

## Quick Reference Commands

```bash
# Test single node (2 GPUs)
bash scripts/utils/test_single_node.sh

# Test single node (4 GPUs)
NPROC_PER_NODE=4 bash scripts/utils/test_single_node.sh

# Check NCCL debug log
cat /tmp/nccl_debug_single_node.log

# Test network connectivity to master
nc -zv <master_ip> 23456

# Check GPU availability
nvidia-smi

# Check PyTorch and NCCL versions
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'NCCL: {torch.cuda.nccl.version()}')"
```

## Expected Output from Successful Test

When the diagnostic script runs successfully, you should see:

```
DIAGNOSTIC SUMMARY
================================================================================
Network connectivity: ✓ PASS
CUDA setup: ✓ PASS
Distributed init: ✓ PASS
Collective ops: ✓ PASS
================================================================================

✓ All tests passed! Distributed setup is working correctly.
```

If you see this, your distributed setup is working and you can proceed with training.
