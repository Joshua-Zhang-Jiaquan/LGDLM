#!/bin/bash
# File: train_qwen_english_l2t_scaled_improved.sh
# Improved version with better error handling and diagnostics
set -xeuo pipefail

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Distributed setup (override via env)
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Improved MASTER_ADDR detection
if [ -z "${MASTER_ADDR:-}" ]; then
  # Try to get the primary IP address
  MASTER_ADDR=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || true)
  if [ -z "${MASTER_ADDR}" ]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}' | head -n1)
  fi
  if [ -z "${MASTER_ADDR}" ]; then
    echo "ERROR: Could not determine MASTER_ADDR"
    exit 1
  fi
fi

# NPROC_PER_NODE behavior
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

# NCCL configuration
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN  # Changed from INFO to reduce log spam
export NCCL_DEBUG_FILE="/tmp/nccl_debug_${NODE_RANK}.log"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Additional NCCL tuning (optional, uncomment if needed)
# export NCCL_SOCKET_IFNAME=eth0  # Set to your network interface
# export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Training Latent-to-Text (l2t) model with scaled steps"
echo "Node Rank: ${NODE_RANK}"
echo "NNODES: ${NNODES} NPROC_PER_NODE: ${NPROC_PER_NODE} GLOBAL_WORLD_SIZE: ${GLOBAL_WORLD_SIZE}"
echo "BASE_WORLD_SIZE: ${BASE_WORLD_SIZE}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "Scaled steps: ${L2T_STEPS} save: ${L2T_SAVE_FREQ} log: ${L2T_LOG_FREQ} eval: ${L2T_EVAL_FREQ}"
echo "=========================================="

# Pre-flight checks
echo "Running pre-flight checks..."

# Check 1: Verify GPU availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. CUDA may not be available."
  exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "✓ Found ${GPU_COUNT} GPUs"

if [ "${NPROC_PER_NODE}" -gt "${GPU_COUNT}" ]; then
  echo "ERROR: NPROC_PER_NODE (${NPROC_PER_NODE}) > available GPUs (${GPU_COUNT})"
  exit 1
fi

# Check 2: Verify data paths exist
if [ ! -d "${TOKEN_DIR}" ]; then
  echo "ERROR: Token directory does not exist: ${TOKEN_DIR}"
  exit 1
fi
echo "✓ Token directory exists: ${TOKEN_DIR}"

if [ ! -d "${LATENT_DIR}" ]; then
  echo "ERROR: Latent directory does not exist: ${LATENT_DIR}"
  exit 1
fi
echo "✓ Latent directory exists: ${LATENT_DIR}"

# Check 3: Verify master node is reachable (for non-master nodes)
if [ "${NODE_RANK}" -ne 0 ]; then
  echo "Checking connectivity to master node..."
  if ! ping -c 1 -W 5 "${MASTER_ADDR}" >/dev/null 2>&1; then
    echo "WARNING: Cannot ping master node at ${MASTER_ADDR}"
    echo "This may be normal if ICMP is blocked. Continuing..."
  else
    echo "✓ Master node is reachable"
  fi

  # Try to connect to the port
  if timeout 5 bash -c "cat < /dev/null > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
    echo "✓ Master port ${MASTER_PORT} is accessible"
  else
    echo "WARNING: Cannot connect to master port ${MASTER_ADDR}:${MASTER_PORT}"
    echo "This may cause training to fail. Check firewall settings."
  fi
fi

# Check 4: Verify PyTorch and NCCL
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ NCCL version: {torch.cuda.nccl.version() if torch.cuda.is_available() else \"N/A\"}')"

echo "Pre-flight checks completed!"
echo "=========================================="

# Create output directory
mkdir -p output_dir/l2t_logs

# Launch training with retry logic
MAX_RETRIES="${MAX_RETRIES:-1}"  # Set to 3 for automatic retries
RETRY_COUNT=0
SUCCESS=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if [ $MAX_RETRIES -gt 1 ]; then
    echo "Training attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
  fi

  if torchrun \
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
    data.latent_dir="${LATENT_DIR}"; then

    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
    SUCCESS=1
    break
  else
    EXIT_CODE=$?
    RETRY_COUNT=$((RETRY_COUNT + 1))

    echo "=========================================="
    echo "✗ Training failed with exit code ${EXIT_CODE}"
    echo "=========================================="

    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      echo "Retrying in 30 seconds..."
      sleep 30
    fi
  fi
done

if [ $SUCCESS -eq 0 ]; then
  echo "=========================================="
  echo "✗ Training failed after ${MAX_RETRIES} attempt(s)"
  echo "Check NCCL debug log: /tmp/nccl_debug_${NODE_RANK}.log"
  echo "=========================================="
  exit 1
fi
