#!/bin/bash
# File: sample_l2t_1node.sh
set -xeuo pipefail

########################################
# Environment bootstrap
########################################
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

########################################
# Runtime hygiene
########################################
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export WANDB_DISABLED="true"
export WANDB_MODE="disabled"

########################################
# Paths & parameters
########################################
CKPT="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/outputs/2025-12-31/04-14-50/output_dir/l2t_models/mmdit-l2t-training/latest/checkpoint.pt"

CONFIG="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/configs/mmdit.yaml"

NPY_DIR="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/e5_1024d_full/latents/train/"

OUT_DIR="./fixed_l2t_results"

NUM_SAMPLES=3
BATCH_SIZE=1
STEPS=2000

mkdir -p "${OUT_DIR}"

########################################
# Run L2T sampling
########################################
python latentDLM_mmdit/sample_l2t_fixed.py \
  --checkpoint "${CKPT}" \
  --config "${CONFIG}" \
  --npy_dir "${NPY_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --output_dir "${OUT_DIR}"

echo "=========================================="
echo "L2T sampling completed."
echo "Results saved to: ${OUT_DIR}"
echo "=========================================="
