# WeDLM-8B Latent Bridge

This folder integrates **WeDLM-8B** (`tencent/WeDLM-8B-*`) as a pretrained text backbone and adds a minimal
**AR latent↔text bridge** for your E5 latents:

- `latent → text`: latent prefix conditioning + teacher-forcing CE loss
- `text → latent`: regression head (MSE) to your continuous E5 latent

All WeDLM-specific code lives under `wedlm_bridge/` (no edits to `latentDLM_mmdit/`).

## What this does (and does not do)

This is an **AR-conditioned** encoder/decoder bridge:

- **Encode (text→latent):** run WeDLM on text and regress an E5 vector from pooled hidden states.
- **Decode (latent→text):** project an E5 vector into a learned **prefix embedding** sequence and condition WeDLM to generate text.

It is **not** (yet) a unified diffusion model over `(text, latent)` and it does **not** use your repo’s MMDiT module.
It’s the simplest way to get usable text↔latent conversion while keeping WeDLM’s base language ability.

## Hardware planning (H200 141GB)

### Minimum GPUs
- **1 GPU is enough** (WeDLM-8B is replicated per GPU under DDP; adding GPUs increases throughput but does not shard the model).

### Recommended starting point
- On an **H200 141GB**, start with:
  - `DTYPE=bf16`
  - `TRAIN_BATCH_SIZE=2` (then increase)
  - `model.max_seq_len=1024` (or reduce to 512 for faster iteration)
  - keep `freeze_backbone=true` and `freeze_lm_head=true` for the first runs

### Why freezing first
With `freeze_backbone=true`, you train only:
- `latent_prefix` (latent→prefix conditioning)
- `text_to_latent` (text→latent regression)

This makes training more stable and preserves base fluency. You can later unfreeze selected layers if needed.

## Environment

WeDLM's HuggingFace compatibility targets newer `transformers` (their repo pins `4.56.1`).
Use a separate env for this folder.

Example (adjust torch wheel for your CUDA):

```bash
python -m venv .venv_wedlm
source .venv_wedlm/bin/activate

pip install torch
pip install "transformers>=4.56,<4.57" safetensors sentencepiece huggingface-hub
pip install -r requirements.txt
```

Notes:
- `trust_remote_code=true` is required for WeDLM HF loading; only do this in an environment you trust.
- The first run will download tens of GB of weights. Prefer local SSD for `model.pretrained_local_dir`.

## Data format (your JSON + .npy latents)

Training uses the existing JSON dataloader in `latentDLM_mmdit/data_simple.py`.

Your training file should look like:

```json
[
  {"text": "some text", "latent_path": "latents/000001.npy"},
  {"text": "another text", "latent_path": "latents/000002.npy"}
]
```

Rules:
- `latent_path` is resolved **relative to the JSON file’s directory**.
- Each `.npy` should be a float array of shape `[latent_dim]` or `[1, latent_dim]` (default `latent_dim=1024`).
- If you stored multiple vectors per sample, the loader pads them; the trainer averages them down to `[B, D]`.

Key knobs:
- `model.latent_dim` must match your E5 latent dimension.
- `model.max_seq_len` controls tokenization length (the loader pads to max length).

## Choose a backbone checkpoint

In `wedlm_bridge/configs/wedlm8b_ar_bridge.yaml`, set:
- Base model: `model.pretrained_model_name_or_path="tencent/WeDLM-8B-Base"`
- Instruct model: `model.pretrained_model_name_or_path="tencent/WeDLM-8B-Instruct"`

If you use the **Instruct** model, you typically want your dataset `text` to already be formatted as a chat prompt
(e.g., using the tokenizer’s chat template during preprocessing), because this training code treats `text` as plain text.

## Train

```bash
torchrun --nnodes 1 --nproc_per_node 8 wedlm_bridge/train_wedlm_ar_bridge.py \
  --config-name wedlm8b_ar_bridge
```

Switch base vs instruct:

```bash
torchrun --nproc_per_node 8 wedlm_bridge/train_wedlm_ar_bridge.py \
  --config-name wedlm8b_ar_bridge \
  model.pretrained_model_name_or_path="tencent/WeDLM-8B-Instruct"
```

## One-click run script

```bash
bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
```

The script auto-detects `NPROC_PER_NODE` from `nvidia-smi` when you don’t set it.
For a single H200, you typically want `NPROC_PER_NODE=1`.

### Prefetch (download first, then train offline)

`wedlm_bridge/run_wedlm8b_ar_bridge.sh` always **prefetches** the model repo into `LOCAL_MODEL_DIR`
before launching training, then sets:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`

So the actual training run does not hit the network.

To *only* download (e.g., on a login node) and exit:

```bash
DOWNLOAD_ONLY=1 bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
```

Common overrides:

```bash
NPROC_PER_NODE=8 bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
RUN_NAME="wedlm8b-latent-bridge-exp1" bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
MODEL_ID="tencent/WeDLM-8B-Instruct" bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
```

The run script supports (environment variables):
- **DDP:** `NNODES`, `NODE_RANK`, `NPROC_PER_NODE`, `MASTER_ADDR`, `MASTER_PORT`
- **Model:** `MODEL_ID`, `LOCAL_MODEL_DIR`
- **Data:** `DATA_ROOT`, `TRAIN_JSON`, `VAL_JSON`
- **Training:** `DTYPE`, `TASK`, `TRAIN_BATCH_SIZE`, `NUM_STEPS`, `LR`
- **Logging:** `RUN_NAME`, `SAVE_DIR`, plus standard `WANDB_*`

## Training modes (what “TASK” means)

`training.task` / `TASK` controls which direction(s) you train:

- `TASK=t2l` (text→latent only)
  - Loss: MSE between predicted latent and E5 latent.
  - Trains: `text_to_latent` head (and optionally any unfrozen backbone).
- `TASK=l2t` (latent→text only)
  - Loss: teacher-forcing CE on next-token prediction (shifted logits/labels).
  - Trains: `latent_prefix` (and optionally any unfrozen backbone).
- `TASK=joint` (both directions)
  - Loss: `text_loss_weight * CE + latent_loss_weight * MSE`.

Weights:
- `training.text_loss_weight` and `training.latent_loss_weight` (defaults 1.0/1.0).

## Recommended training recipe (practical schedule)

This schedule is designed to keep WeDLM fluent while learning text↔latent mapping.

### Stage 0: download once (optional but recommended)

Run with 1 process first so your local cache is populated:

```bash
NPROC_PER_NODE=1 TASK=t2l NUM_STEPS=1 bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
```

This creates `LOCAL_MODEL_DIR` (default `./local_models/wedlm-8b`) and avoids redundant downloads later.

### Stage 1: train text→latent (encoder)

```bash
TASK=t2l TRAIN_BATCH_SIZE=4 LR=1e-4 NUM_STEPS=20000 bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
```

What to watch:
- `latent_loss` should steadily decrease.

### Stage 2: train latent→text (decoder conditioning)

```bash
TASK=l2t TRAIN_BATCH_SIZE=2 LR=5e-5 NUM_STEPS=20000 bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
```

What to watch:
- `text_loss` should decrease; `text_acc` should increase.

### Stage 3: joint finetune

```bash
TASK=joint TRAIN_BATCH_SIZE=2 LR=5e-5 NUM_STEPS=50000 bash wedlm_bridge/run_wedlm8b_ar_bridge.sh
```

If you see one loss dominating, adjust `text_loss_weight` / `latent_loss_weight`.

### Optional Stage 4: unfreeze partial backbone (advanced)

Only if you need stronger latent conditioning / better reconstruction:
- set `model.freeze_backbone=false` (and optionally `model.freeze_lm_head=false`)
- reduce LR significantly (e.g., `LR=1e-5` or `5e-6`)
- keep batch small and watch for divergence

Full finetuning changes WeDLM’s base behavior; do this only after the bridge works.

## Local caching (download once)

`wedlm_bridge/pretrained_utils.py` downloads once into `model.pretrained_local_dir` (with a lock),
then loads locally on subsequent runs.

Tips:
- For multi-node, use a **shared** `LOCAL_MODEL_DIR` (shared filesystem) if you want a single download.
- If each node has its own filesystem, each node will download once (still safe).

## Outputs and checkpoints

Checkpoints are saved to:

`logging.save_dir/logging.run_name/checkpoint_step_<N>.pt`

Defaults:
- `SAVE_DIR=./outputs`
- `RUN_NAME=wedlm-8b-ar-latent-bridge`

This trainer does **not** implement resume yet. Treat checkpoints as “save for later / export” until resume is added.

## Troubleshooting

- **Hangs on startup (DDP):** ensure `MASTER_ADDR/MASTER_PORT` are correct; on SLURM, the run script auto-detects when possible.
- **OOM:** lower `TRAIN_BATCH_SIZE`, reduce `model.max_seq_len`, or keep `freeze_backbone=true`.
- **Model downloads repeatedly:** set `LOCAL_MODEL_DIR` to a stable path; make sure it contains `config.json` after the first download.
- **Instruct model outputs weird text:** your dataset text may need chat-template formatting; prefer `WeDLM-8B-Base` for plain text corpora.

## Smoke

The smoke script is gated (it will skip if the weights aren't already present):

```bash
python wedlm_bridge/scripts/smoke_wedlm8b_ar_bridge.py
```
