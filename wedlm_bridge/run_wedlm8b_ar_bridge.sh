#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Optional venv activation
VENV_PATH="${VENV_PATH:-.venv_wedlm}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

# ---------------- Run settings ----------------
CONFIG_NAME="${CONFIG_NAME:-wedlm8b_ar_bridge}"
MODEL_ID="${MODEL_ID:-tencent/WeDLM-8B-Base}"
RUN_NAME="${RUN_NAME:-wedlm-8b-ar-latent-bridge}"
SAVE_DIR="${SAVE_DIR:-./outputs}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-./local_models/wedlm-8b}"

DATA_ROOT="${DATA_ROOT:-preprocessed_data/e5_1024d_full}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_ROOT}/train_data.json}"
VAL_JSON="${VAL_JSON:-${DATA_ROOT}/train_data.json}"

DTYPE="${DTYPE:-bf16}"
TASK="${TASK:-joint}" # l2t | t2l | joint

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_STEPS="${NUM_STEPS:-250000}"
LR="${LR:-1e-4}"

# Prefetch model repo to LOCAL_MODEL_DIR before launching training.
DOWNLOAD_ONLY="${DOWNLOAD_ONLY:-0}"
# After prefetch, force offline to avoid any unexpected hub calls.
OFFLINE_AFTER_DOWNLOAD="${OFFLINE_AFTER_DOWNLOAD:-1}"

# Optional: override HuggingFace endpoint (useful if your cluster sets HF_ENDPOINT to a rate-limited mirror).
HF_ENDPOINT_OVERRIDE="${HF_ENDPOINT_OVERRIDE:-}"
if [[ -n "${HF_ENDPOINT_OVERRIDE}" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT_OVERRIDE}"
fi

_abs_path() {
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

SAVE_DIR="$(_abs_path "${SAVE_DIR}")"
LOCAL_MODEL_DIR="$(_abs_path "${LOCAL_MODEL_DIR}")"
TRAIN_JSON="$(_abs_path "${TRAIN_JSON}")"
VAL_JSON="$(_abs_path "${VAL_JSON}")"

# ---------------- torchrun / DDP ----------------
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"

# Default to the number of visible GPUs if possible; otherwise 1.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Count comma-separated GPU ids.
    NPROC_PER_NODE="$(python - <<'PY'
import os
s = os.environ.get("CUDA_VISIBLE_DEVICES", "")
parts = [p for p in s.split(",") if p.strip() != ""]
print(len(parts) if parts else 0)
PY
)"
    if [[ "${NPROC_PER_NODE}" -le 0 ]]; then
      NPROC_PER_NODE=1
    fi
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${GPU_COUNT}" -gt 0 ]]; then
      NPROC_PER_NODE="${GPU_COUNT}"
    else
      NPROC_PER_NODE=1
    fi
  else
    NPROC_PER_NODE=1
  fi
else
  NPROC_PER_NODE="${NPROC_PER_NODE}"
fi

MASTER_PORT="${MASTER_PORT:-29500}"
if [[ "${NNODES}" -gt 1 ]]; then
  if [[ -z "${MASTER_ADDR:-}" ]]; then
    if [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
      MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
    elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
      MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
    else
      echo "ERROR: Multi-node run requires MASTER_ADDR (or SLURM vars)." >&2
      exit 1
    fi
  fi
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

# Disable W&B by default (can override outside)
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DIR="${WANDB_DIR:-${SAVE_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"

echo "WeDLM AR latent bridge"
echo "  NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "  MODEL_ID=${MODEL_ID}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR}"
echo "  TRAIN_JSON=${TRAIN_JSON}"
echo "  TASK=${TASK} DTYPE=${DTYPE} BS=${TRAIN_BATCH_SIZE} STEPS=${NUM_STEPS} LR=${LR}"

echo "Prefetching model files (download once):"
MODEL_ID="${MODEL_ID}" LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR}" python - <<'PY'
import os
import time
from pathlib import Path

model_id = os.environ["MODEL_ID"]
local_dir = Path(os.environ["LOCAL_MODEL_DIR"])
local_dir.mkdir(parents=True, exist_ok=True)

cfg = local_dir / "config.json"
if cfg.exists():
    print(f"  OK: already present: {cfg}")
else:
    print(f"  Downloading {model_id} -> {local_dir} ...")
    from huggingface_hub import snapshot_download
    from huggingface_hub import __version__ as hub_version

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    endpoint = os.getenv("HF_ENDPOINT")
    print(f"  huggingface_hub={hub_version} HF_ENDPOINT={endpoint!r} token={'set' if token else 'missing'}")
    if not token:
        print("  NOTE: If you hit 429 rate limits, set HF_TOKEN (or run `huggingface-cli login`).")

    max_retries = int(os.getenv("PREFETCH_MAX_RETRIES", "8"))
    max_workers = int(os.getenv("PREFETCH_MAX_WORKERS", "4"))

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token,
                max_workers=max_workers,
            )
            try:
                kwargs["resume_download"] = True
                snapshot_download(**kwargs)
            except TypeError:
                kwargs.pop("resume_download", None)
                snapshot_download(**kwargs)
            last_err = None
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            msg = str(e)
            is_429 = ("429" in msg) or ("Too Many Requests" in msg)
            if is_429 and attempt < max_retries:
                sleep_s = min(60 * (2 ** (attempt - 1)), 600)
                print(f"  Rate limited (429). Retry {attempt}/{max_retries} in {sleep_s}s...")
                time.sleep(sleep_s)
                continue
            raise
    if last_err is not None:
        raise last_err
    if not cfg.exists():
        raise RuntimeError(f"Download finished but {cfg} is still missing. Check local_dir contents.")
    print(f"  OK: downloaded: {cfg}")
PY

if [[ "${OFFLINE_AFTER_DOWNLOAD}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  echo "Offline mode enabled: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1"
fi

if [[ "${DOWNLOAD_ONLY}" == "1" ]]; then
  echo "DOWNLOAD_ONLY=1 set; exiting after prefetch."
  exit 0
fi

echo "Environment sanity check:"
python - <<'PY'
import os
import sys
try:
    import torch
except Exception as e:
    print("  ERROR: failed to import torch:", e)
    sys.exit(2)
print("  torch:", torch.__version__)
print("  CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("  torch.cuda.is_available():", torch.cuda.is_available())
print("  torch.cuda.device_count():", torch.cuda.device_count())
PY

if python - <<'PY'
import torch
import sys
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)
PY
then
  : # ok
else
  if [[ "${ALLOW_CPU:-0}" == "1" ]]; then
    echo "WARNING: CUDA not available; proceeding because ALLOW_CPU=1 (this will be extremely slow for 8B models)." >&2
  else
    echo "ERROR: CUDA is not available in this environment." >&2
    echo " - If you're on SLURM, run inside a GPU allocation (e.g., srun/sbatch with --gres=gpu:...)." >&2
    echo " - Ensure your PyTorch build has CUDA support and the container has GPU devices exposed." >&2
    echo " - Also ensure NPROC_PER_NODE matches the number of visible GPUs." >&2
    exit 1
  fi
fi

if [[ "${NNODES}" -gt 1 ]]; then
  torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    wedlm_bridge/train_wedlm_ar_bridge.py \
    --config-name "${CONFIG_NAME}" \
    model.pretrained_model_name_or_path="${MODEL_ID}" \
    model.pretrained_local_dir="${LOCAL_MODEL_DIR}" \
    model.local_files_only=true \
    data.data_files.train="${TRAIN_JSON}" \
    data.data_files.validation="${VAL_JSON}" \
    training.dtype="${DTYPE}" \
    training.task="${TASK}" \
    training.train_batch_size="${TRAIN_BATCH_SIZE}" \
    training.num_train_steps="${NUM_STEPS}" \
    optimizer.lr="${LR}" \
    logging.run_name="${RUN_NAME}" \
    logging.save_dir="${SAVE_DIR}"
else
  torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    wedlm_bridge/train_wedlm_ar_bridge.py \
    --config-name "${CONFIG_NAME}" \
    model.pretrained_model_name_or_path="${MODEL_ID}" \
    model.pretrained_local_dir="${LOCAL_MODEL_DIR}" \
    model.local_files_only=true \
    data.data_files.train="${TRAIN_JSON}" \
    data.data_files.validation="${VAL_JSON}" \
    training.dtype="${DTYPE}" \
    training.task="${TASK}" \
    training.train_batch_size="${TRAIN_BATCH_SIZE}" \
    training.num_train_steps="${NUM_STEPS}" \
    optimizer.lr="${LR}" \
    logging.run_name="${RUN_NAME}" \
    logging.save_dir="${SAVE_DIR}"
fi
