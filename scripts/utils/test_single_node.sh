#!/bin/bash
# File: test_single_node.sh
# Test distributed training on a single node (simpler test)
set -xeuo pipefail

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Single node setup
NNODES=1
NODE_RANK=0
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"  # Test with 2 GPUs by default
MASTER_ADDR="localhost"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "=========================================="
echo "Single Node Test Configuration"
echo "=========================================="
echo "NNODES: ${NNODES}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "=========================================="

# Set NCCL environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE="/tmp/nccl_debug_single_node.log"

echo "Running single-node distributed diagnostic test..."
echo ""

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  test_distributed.py

echo ""
echo "=========================================="
echo "Single-node test completed!"
echo "Check /tmp/nccl_debug_single_node.log for NCCL details"
echo "=========================================="
