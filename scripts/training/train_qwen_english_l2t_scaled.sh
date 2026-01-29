#!/bin/bash
# File: train_l2t_scaled.sh
set -xeuo pipefail

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Distributed setup (override via env)
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}' | head -n1)}"
MASTER_PORT="${MASTER_PORT:-29500}"

# NPROC_PER_NODE behavior:
# - single-node: auto-detect GPUs unless explicitly set
# - multi-node: default to 8 GPUs per node unless explicitly set
if [ -z "${NPROC_PER_NODE:-}" ]; then
  if [ "${NNODES}" -gt 1 ]; then
    NPROC_PER_NODE=8
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _cdevs <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE=0
    for _dev in "${_cdevs[@]}"; do
      if [ -n "${_dev}" ]; then
        NPROC_PER_NODE=$((NPROC_PER_NODE + 1))
      fi
    done
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  else
    NPROC_PER_NODE=$(python - <<'PY'
try:
    import torch
    count = torch.cuda.device_count()
except Exception:
    count = 0
print(count)
PY
)
  fi
fi

if [ -z "${NPROC_PER_NODE}" ] || [ "${NPROC_PER_NODE}" -le 0 ]; then
  echo "ERROR: Could not determine GPU count. Set NPROC_PER_NODE explicitly."
  exit 1
fi

GLOBAL_WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
BASE_WORLD_SIZE="${BASE_WORLD_SIZE:-${NPROC_PER_NODE}}"

if [ "${GLOBAL_WORLD_SIZE}" -le 0 ]; then
  echo "ERROR: GLOBAL_WORLD_SIZE must be > 0"
  exit 1
fi
if [ "${BASE_WORLD_SIZE}" -le 0 ]; then
  echo "ERROR: BASE_WORLD_SIZE must be > 0"
  exit 1
fi

scale_value() {
  local base=$1
  # ceil(base * BASE_WORLD_SIZE / GLOBAL_WORLD_SIZE)
  echo $(( (base * BASE_WORLD_SIZE + GLOBAL_WORLD_SIZE - 1) / GLOBAL_WORLD_SIZE ))
}

# Base schedule (tuned for BASE_WORLD_SIZE)
L2T_BASE_STEPS="${L2T_BASE_STEPS:-1000000}"
L2T_BASE_SAVE_FREQ="${L2T_BASE_SAVE_FREQ:-50000}"
L2T_BASE_LOG_FREQ="${L2T_BASE_LOG_FREQ:-10000}"
L2T_BASE_EVAL_FREQ="${L2T_BASE_EVAL_FREQ:-10000}"

L2T_STEPS=$(scale_value "${L2T_BASE_STEPS}")
L2T_SAVE_FREQ=$(scale_value "${L2T_BASE_SAVE_FREQ}")
L2T_LOG_FREQ=$(scale_value "${L2T_BASE_LOG_FREQ}")
L2T_EVAL_FREQ=$(scale_value "${L2T_BASE_EVAL_FREQ}")

# Training knobs (override via env)
L2T_TRAIN_BS="${L2T_TRAIN_BS:-4}"
L2T_EVAL_BS="${L2T_EVAL_BS:-4}"
L2T_CONFIG="${L2T_CONFIG:-mmdit_preprocessed}"
L2T_RUN_NAME="${L2T_RUN_NAME:-mmdit-qwen-32d-l2t-scaled}"
L2T_SAVE_DIR="${L2T_SAVE_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved}"
L2T_COMPILE="${L2T_COMPILE:-false}"
LATENT_DIM="${LATENT_DIM:-32}"
DTYPE="${DTYPE:-bf16}"
DATA_WORKERS="${DATA_WORKERS:-16}"
TOKEN_DIR="${TOKEN_DIR:-/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/tokens/train}"
LATENT_DIR="${LATENT_DIR:-/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/latents/train}"

# Reduce allocator fragmentation and avoid wandb login prompts / unwritable paths
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_DIR="./output_dir/wandb"
mkdir -p "${WANDB_DIR}"

# Relax NCCL watchdog to mitigate false-positive hangs
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_ASYNC_ERROR_HANDLING=1

# 在现有的NCCL设置下面添加这些：
export NCCL_TIMEOUT=7200  # 7200秒 = 2小时
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO  # 用于调试，可以改为WARN或ERROR
export NCCL_DEBUG_FILE="/tmp/nccl_debug_${NODE_RANK}.log"

# 设置PyTorch的NCCL参数
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

echo "Training Latent-to-Text (l2t) model with scaled steps"
echo "Node Rank: ${NODE_RANK}"
echo "NNODES: ${NNODES} NPROC_PER_NODE: ${NPROC_PER_NODE} GLOBAL_WORLD_SIZE: ${GLOBAL_WORLD_SIZE}"
echo "BASE_WORLD_SIZE: ${BASE_WORLD_SIZE}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "Scaled steps: ${L2T_STEPS} save: ${L2T_SAVE_FREQ} log: ${L2T_LOG_FREQ} eval: ${L2T_EVAL_FREQ}"

# Create output directory
mkdir -p output_dir/l2t_logs

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  latentDLM_mmdit/train_mmdit.py \
  --config-name "${L2T_CONFIG}" \
  logging.run_name="${L2T_RUN_NAME}" \
  logging.save_dir="${L2T_SAVE_DIR}" \
  training.train_batch_size="${L2T_TRAIN_BS}" \
  training.eval_batch_size="${L2T_EVAL_BS}" \
  training.num_train_steps="${L2T_STEPS}" \
  training.compile_model="${L2T_COMPILE}" \
  loss.loss_type="l2t" \
  model.latent_dim="${LATENT_DIM}" \
  training.dtype="${DTYPE}" \
  logging.save_freq="${L2T_SAVE_FREQ}" \
  logging.log_freq="${L2T_LOG_FREQ}" \
  logging.eval_freq="${L2T_EVAL_FREQ}" \
  data.num_workers="${DATA_WORKERS}" \
  data.token_dir="${TOKEN_DIR}" \
  data.latent_dir="${LATENT_DIR}"
