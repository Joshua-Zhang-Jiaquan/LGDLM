#!/bin/bash
# File: train_qwen_english_l2t_stable.sh
# Improved training script with fixes for:
# 1. Distributed training connectivity issues
# 2. NaN gradient prevention
set -xeuo pipefail

echo "=========================================="
echo "MM-LDLM Stable Training Script"
echo "=========================================="
echo "This script includes fixes for:"
echo "  - Distributed training connectivity"
echo "  - NaN gradient prevention"
echo "=========================================="
echo ""

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# ============================================================
# DISTRIBUTED SETUP
# ============================================================
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

# ============================================================
# TRAINING SCHEDULE (scaled)
# ============================================================
L2T_BASE_STEPS="${L2T_BASE_STEPS:-1000000}"
L2T_BASE_SAVE_FREQ="${L2T_BASE_SAVE_FREQ:-50000}"
L2T_BASE_LOG_FREQ="${L2T_BASE_LOG_FREQ:-10000}"
L2T_BASE_EVAL_FREQ="${L2T_BASE_EVAL_FREQ:-10000}"

L2T_STEPS=$(scale_value "${L2T_BASE_STEPS}")
L2T_SAVE_FREQ=$(scale_value "${L2T_BASE_SAVE_FREQ}")
L2T_LOG_FREQ=$(scale_value "${L2T_BASE_LOG_FREQ}")
L2T_EVAL_FREQ=$(scale_value "${L2T_BASE_EVAL_FREQ}")

# ============================================================
# TRAINING CONFIGURATION (NaN-safe defaults)
# ============================================================
L2T_TRAIN_BS="${L2T_TRAIN_BS:-4}"
L2T_EVAL_BS="${L2T_EVAL_BS:-4}"
L2T_CONFIG="${L2T_CONFIG:-mmdit_stable}"  # Use stable config by default
L2T_RUN_NAME="${L2T_RUN_NAME:-mmdit-qwen-32d-l2t-stable}"
L2T_SAVE_DIR="${L2T_SAVE_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved}"
L2T_COMPILE="${L2T_COMPILE:-false}"
LATENT_DIM="${LATENT_DIM:-32}"
DTYPE="${DTYPE:-bf16}"
DATA_WORKERS="${DATA_WORKERS:-16}"
TOKEN_DIR="${TOKEN_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/qwen-embeddings-32/tokens/train}"
LATENT_DIR="${LATENT_DIR:-/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/qwen-embeddings-32/latents/train}"

# NaN-safe hyperparameters (can be overridden)
LEARNING_RATE="${LEARNING_RATE:-5e-5}"  # Reduced from 1e-4
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"  # Reduced from 1.0
WARMUP_STEPS="${WARMUP_STEPS:-2000}"  # Increased from 1000
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-0.1}"  # Reduced from 1.0
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"  # Added for stability

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
# Reduce allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Disable wandb
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_DIR="./output_dir/wandb"
mkdir -p "${WANDB_DIR}"

# NCCL configuration (relaxed timeouts)
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=12000000
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN  # Changed from INFO to reduce log spam
export NCCL_DEBUG_FILE="/tmp/nccl_debug_${NODE_RANK}.log"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Optional: Set network interface if needed
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1

# ============================================================
# DISPLAY CONFIGURATION
# ============================================================
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Distributed Setup:"
echo "  Node Rank: ${NODE_RANK}"
echo "  NNODES: ${NNODES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  GLOBAL_WORLD_SIZE: ${GLOBAL_WORLD_SIZE}"
echo "  BASE_WORLD_SIZE: ${BASE_WORLD_SIZE}"
echo "  Master Addr: ${MASTER_ADDR}"
echo "  Master Port: ${MASTER_PORT}"
echo ""
echo "Training Schedule (scaled):"
echo "  Total steps: ${L2T_STEPS}"
echo "  Save freq: ${L2T_SAVE_FREQ}"
echo "  Log freq: ${L2T_LOG_FREQ}"
echo "  Eval freq: ${L2T_EVAL_FREQ}"
echo ""
echo "Hyperparameters (NaN-safe):"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Gradient clip: ${GRAD_CLIP_NORM}"
echo "  Warmup steps: ${WARMUP_STEPS}"
echo "  Latent loss weight: ${LATENT_LOSS_WEIGHT}"
echo "  Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "  Batch size: ${L2T_TRAIN_BS}"
echo "  Config: ${L2T_CONFIG}"
echo "=========================================="
echo ""

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo "Running pre-flight checks..."
echo ""

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

# Check 3: Verify config file exists
CONFIG_FILE="latentDLM_mmdit/configs/${L2T_CONFIG}.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "WARNING: Config file not found: ${CONFIG_FILE}"
  echo "  Will try to use default config"
else
  echo "✓ Config file exists: ${CONFIG_FILE}"
fi

# Check 4: Verify master node is reachable (for non-master nodes)
if [ "${NODE_RANK}" -ne 0 ]; then
  echo "Checking connectivity to master node..."
  if ! ping -c 1 -W 5 "${MASTER_ADDR}" >/dev/null 2>&1; then
    echo "WARNING: Cannot ping master node at ${MASTER_ADDR}"
    echo "  This may be normal if ICMP is blocked. Continuing..."
  else
    echo "✓ Master node is reachable"
  fi

  # Try to connect to the port
  if timeout 5 bash -c "cat < /dev/null > /dev/tcp/${MASTER_ADDR}/${MASTER_PORT}" 2>/dev/null; then
    echo "✓ Master port ${MASTER_PORT} is accessible"
  else
    echo "WARNING: Cannot connect to master port ${MASTER_ADDR}:${MASTER_PORT}"
    echo "  This may cause training to fail. Check firewall settings."
  fi
fi

# Check 5: Verify PyTorch and NCCL
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ NCCL version: {torch.cuda.nccl.version() if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Pre-flight checks completed!"
echo "=========================================="
echo ""

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================
mkdir -p output_dir/l2t_logs
mkdir -p "${L2T_SAVE_DIR}"

# Create a log file for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="train_logs/train_${TIMESTAMP}_node${NODE_RANK}.log"
mkdir -p train_logs

# ============================================================
# LAUNCH TRAINING
# ============================================================
echo "Launching training..."
echo "Logs will be saved to: ${LOG_FILE}"
echo ""

# Launch with error handling
set +e  # Don't exit on error, we want to handle it

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  latentDLM_mmdit/train_mmdit_stable.py \
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
  data.latent_dir="${LATENT_DIR}" \
  optimizer.lr="${LEARNING_RATE}" \
  optimizer.grad_clip_norm="${GRAD_CLIP_NORM}" \
  training.warmup_steps="${WARMUP_STEPS}" \
  loss.latent_loss_weight="${LATENT_LOSS_WEIGHT}" \
  training.gradient_accumulation_steps="${GRAD_ACCUM_STEPS}" \
  2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
set -e  # Re-enable exit on error

# ============================================================
# POST-TRAINING SUMMARY
# ============================================================
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Training completed successfully!"
else
  echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "=========================================="
echo ""
echo "Summary:"
echo "  Exit code: ${EXIT_CODE}"
echo "  Log file: ${LOG_FILE}"
echo "  NCCL debug: /tmp/nccl_debug_${NODE_RANK}.log"
echo "  Checkpoints: ${L2T_SAVE_DIR}/${L2T_RUN_NAME}/"
echo ""

if [ $EXIT_CODE -ne 0 ]; then
  echo "Troubleshooting:"
  echo "  1. Check log file for errors: ${LOG_FILE}"
  echo "  2. Check NCCL debug log: /tmp/nccl_debug_${NODE_RANK}.log"
  echo "  3. Look for 'ERROR' or 'NaN' in logs:"
  echo "     grep -E 'ERROR|NaN' ${LOG_FILE}"
  echo "  4. Check if distributed setup failed:"
  echo "     grep -E 'DistNetworkError|Connection' ${LOG_FILE}"
  echo ""
fi

echo "=========================================="

exit $EXIT_CODE
