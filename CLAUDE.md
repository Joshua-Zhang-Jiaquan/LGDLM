# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MM-LDLM (Multimodal Latent Diffusion Language Models) is a research codebase for training multimodal models that bridge text and latent representations. The core innovation is using MMDiT (Multimodal Diffusion Transformer) architecture to enable bidirectional translation between discrete text tokens and continuous latent embeddings.

## Architecture

### Core Components

**MultimodalMMDiT** (`latentDLM_mmdit/models/multimodal_mmdit.py`):
- Dual-stream architecture with separate encoders for text tokens and latent vectors
- Text uses masked diffusion (MDLM), latents use continuous diffusion
- MMDiT backbone processes both modalities with cross-attention
- Separate output heads: `text_head` for token logits, `latent_head` for noise prediction

**Training Modes** (configured via `training.loss_type`):

1. **`unconditional`**: Train both text and latent generation simultaneously
   - All parameters trainable
   - Model learns to generate both text and latents from noise
   - Use case: General-purpose multimodal generation

2. **`l2t` (latent-to-text)**: Given clean latent embedding, generate text
   - Freezes latent-specific parameters (latent encoder/head)
   - Trains text encoder/head + shared MMDiT backbone
   - Use case: Generate text descriptions from semantic embeddings
   - Most common training mode

3. **`t2l` (text-to-latent)**: Given clean text, generate latent embedding
   - Freezes text-specific parameters (text encoder/head)
   - Trains latent encoder/head + shared MMDiT backbone
   - Use case: Generate semantic embeddings from text

4. **`mixed`**: Combination of all modes with configurable weights
   - All parameters trainable
   - Weights specified in `training.loss_type_weights`
   - Use case: Joint training across all tasks

**Diffusion Processes**:
- Text: Masked diffusion via `MaskedDiffusion` class (replaces tokens with [MASK])
- Latents: Continuous Gaussian diffusion via `ContinuousDiffusion` class (adds noise to embeddings)

### Directory Structure

- `latentDLM_mmdit/` - Main MMDiT implementation (text ↔ latent)
  - `train_mmdit.py` - Primary training script with DDP support
  - `models/multimodal_mmdit.py` - Model architecture
  - `improved_trainer.py` - Training logic with selective parameter freezing
  - `sample_l2t_fixed.py`, `sample_t2l_fixed.py` - Inference scripts
  - `configs/` - Hydra configuration files
- `preprocessed_data/` - Data preprocessing utilities
- `baseline/` - Baseline implementations (HDLM, MDLM, AR)
- `baseline_latent/` - Latent-focused baseline models
- `latentIMG_mmdit/` - Image+text MMDiT (separate from main text-latent work)
- `scripts/` - All executable scripts
  - `training/` - Training shell scripts for various configurations
  - `utils/` - Utility scripts (testing, monitoring, validation, fixes)
- `docs/` - Comprehensive documentation (guides, troubleshooting, references)
- `results/` - Experiment outputs and archived configurations

## Common Commands

### Environment Setup

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # Note: requirements.txt may not exist, check for pyproject.toml or setup.py
```

### Quick Start (Recommended)

For stable training with NaN gradient protection, use the stable training script:

```bash
# Test on single node with 2 GPUs (recommended first step)
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# Scale to multiple nodes once stable
NNODES=2 bash scripts/training/train_qwen_english_l2t_stable.sh  # 16 GPUs
NNODES=4 bash scripts/training/train_qwen_english_l2t_stable.sh  # 32 GPUs
```

The stable script uses `train_mmdit_stable.py` and `improved_trainer_stable.py` which include fixes for NaN gradients and numerical stability issues.

### Data Preprocessing

Extract text embeddings using various encoder models (SONAR, E5, Qwen, etc.):

```bash
# E5 embeddings (1024-dim) on 4 GPUs
torchrun --nnodes=1 --nproc_per_node=4 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model e5 \
  --batch-size 256 \
  --max-samples 10000000 \
  --output-dir preprocessed_data/e5_1024d_full

# Qwen embeddings (32-dim or 1024-dim) on 2 GPUs
torchrun --nnodes=1 --nproc_per_node=2 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model qwen \
  --batch-size 8 \
  --max-samples 10000000 \
  --output-dir preprocessed_data/qwen_embeddings
```

**Output structure**: Creates `texts/train/*.txt`, `latents/train/*.npy`, `train_data.json`, and optionally `validation_data.json`.

**Critical**: Ensure embedding dimensionality matches `model.latent_dim` in training config.

### Training MMDiT

**Single-node training** (L2T mode - latent to text):

```bash
torchrun --nnodes=1 --nproc_per_node=2 latentDLM_mmdit/train_mmdit.py \
  --config-name mmdit \
  logging.run_name="mmdit-l2t-experiment" \
  training.train_batch_size=16 \
  training.eval_batch_size=16 \
  training.num_epochs=50 \
  training.compile_model=false \
  training.dtype=bf16 \
  training.loss_type="l2t" \
  model.latent_dim=1024 \
  data.latent_data_root="/path/to/preprocessed_data/e5_1024d_full"
```

**Multi-node training** (2 nodes, 8 GPUs each):

```bash
# On each node, set environment variables:
export RANK=0  # 0 for master, 1 for worker
export WORLD_SIZE=2
export MASTER_ADDR="<master-node-ip>"
export MASTER_PORT=29500

torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  latentDLM_mmdit/train_mmdit.py \
  --config-name mmdit \
  training.loss_type="l2t" \
  model.latent_dim=1024
```

**Key training parameters**:
- `training.loss_type`: "unconditional", "l2t", "t2l", or "mixed"
- `training.dtype`: "fp32", "fp16", or "bf16" (bf16 recommended for H100/A100)
- `model.latent_dim`: Must match embedding dimension from preprocessing
- `logging.save_freq`: Checkpoint save frequency (steps)
- `logging.eval_freq`: Evaluation frequency (steps)

### Sampling/Inference

**Latent-to-Text (L2T)**:

```bash
python latentDLM_mmdit/sample_l2t_fixed.py \
  --checkpoint /path/to/checkpoint/model.pt \
  --latent-file /path/to/latent.npy \
  --num-samples 10 \
  --num-steps 256 \
  --output-dir results_l2t/
```

**Text-to-Latent (T2L)**:

```bash
python latentDLM_mmdit/sample_t2l_fixed.py \
  --checkpoint /path/to/checkpoint/model.pt \
  --text "Your input text here" \
  --num-steps 256 \
  --output-dir results_t2l/
```

### Evaluation

```bash
python latentDLM_mmdit/evaluate_mmdit_sts.py \
  --checkpoint /path/to/checkpoint/model.pt \
  --eval-dataset stsb \
  --batch-size 32
```

## Configuration System

Uses Hydra for hierarchical configuration. Main config: `latentDLM_mmdit/configs/mmdit.yaml`

**Override syntax**:
```bash
# Override nested config values
python train_mmdit.py \
  --config-name mmdit \
  model.latent_dim=512 \
  training.train_batch_size=32 \
  data.latent_data_root="/new/path"
```

**Important config paths**:
- `data.latent_data_root`: Root directory containing preprocessed data
- `data.data_files.train`: Path to train_data.json (usually auto-set from latent_data_root)
- `tokenizer.name`: Tokenizer to use (default: "bert-base-uncased")
- `tokenizer.cache_dir`: HuggingFace cache directory

## Development Notes

### Training Script Versions

The repository contains two versions of the training code:

**Original versions** (may encounter NaN gradients):
- `latentDLM_mmdit/train_mmdit.py`
- `latentDLM_mmdit/improved_trainer.py`

**Stable versions** (recommended - includes NaN fixes):
- `latentDLM_mmdit/train_mmdit_stable.py`
- `latentDLM_mmdit/improved_trainer_stable.py`

The stable versions include:
- Loss validation before backward pass (skip batches with NaN/inf loss)
- Gradient clipping before NaN detection
- Epsilon values in denominators to prevent divide-by-zero
- Latent normalization for numerical stability
- Loss clamping to prevent overflow
- Conservative default hyperparameters

**Recommendation**: Always use the stable training script (`scripts/training/train_qwen_english_l2t_stable.sh`) which automatically uses the stable versions.

### Pre-flight Checks

The stable training script runs automatic pre-flight checks before starting:
- GPU availability and count verification
- Data directory existence (tokens and latents)
- Config file existence
- Master node reachability (for multi-node setups)
- PyTorch and NCCL version checks

If any check fails, the script exits with a clear error message.

### Distributed Training

- Uses PyTorch DDP with NCCL backend
- Training script (`train_mmdit.py`) automatically detects distributed environment via `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars
- Checkpoints saved only on rank 0, but RNG state saved per-rank for reproducibility
- Use `safe_barrier()` helper for synchronization to handle older PyTorch versions

### Parameter Freezing

The `improved_trainer.py` implements selective parameter freezing:
- **L2T mode**: Freezes latent encoder/head, trains text encoder/head + shared MMDiT
- **T2L mode**: Freezes text encoder/head, trains latent encoder/head + shared MMDiT
- **Unconditional/Mixed**: All parameters trainable

Verify freezing with `verify_parameter_freezing()` method during training.

### Data Format

Preprocessed data structure:
```
preprocessed_data/e5_1024d_full/
├── texts/train/*.txt          # Raw text files
├── latents/train/*.npy        # Numpy arrays [latent_dim]
├── train_data.json            # Metadata mapping
└── validation_data.json       # Optional validation split
```

Each `.npy` file contains a single latent vector of shape `[latent_dim]`.

### Checkpointing

Checkpoints saved to `{logging.save_dir}/{logging.run_name}/latest/`:
- `model.pt` - Model state dict
- `optimizer.pt` - Optimizer state
- `training_state.json` - Training metadata (epoch, step, etc.)
- `rng_state_rank{N}.pt` - Per-rank RNG state for reproducibility

Resume training with `training.resume=/path/to/checkpoint/dir`.

**Important**: The checkpoint directory should point to the folder containing these files (e.g., `saved/mmdit-experiment/latest/`), not to individual `.pt` files.

### Training Logs

The stable training script creates timestamped log files:
```
train_logs/train_YYYYMMDD_HHMMSS_node0.log  # Main training log
/tmp/nccl_debug_0.log                        # NCCL debug output
saved/{run_name}/training_log.jsonl          # Per-step metrics (JSON lines)
```

The `training_log.jsonl` file contains one JSON object per training step with fields like:
- `epoch`, `step`, `batch_in_epoch`
- `loss`, `lr`
- `text_loss`, `latent_loss`, `text_accuracy` (when available)

### Troubleshooting Quick Reference

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| "ERROR: Invalid loss" | NaN/inf in loss computation | Use stable training script, reduce LR |
| "ERROR: Invalid gradient norm" | NaN gradients after backward | Reduce gradient clip, increase warmup |
| "Cannot connect to master" | Network/firewall issue | Set `MASTER_ADDR` explicitly, check port |
| "DistNetworkError" | Node connectivity problem | Verify all nodes can reach master node |
| CUDA OOM | Batch size too large | Reduce `train_batch_size`, increase `grad_accum_steps` |
| Training very slow | Too many data workers | Reduce `data.num_workers` |
| Loss not decreasing | Learning rate too low | Increase `LEARNING_RATE` (but watch for NaN) |
| Vocab size mismatch | Tokenizer/model mismatch | Check `text_head.weight` shape in checkpoint |
| Latent dim mismatch | Preprocessing/config mismatch | Ensure `model.latent_dim` matches encoder output |

### Common Issues

1. **Vocab size mismatch**: Ensure tokenizer vocab size matches model's `rounded_vocab_size`. Check `text_head.weight` shape in checkpoint.

2. **Latent dimension mismatch**: Preprocessing encoder output dim must equal `model.latent_dim` in training config.

3. **OOM errors**: Reduce `training.train_batch_size`, use `training.dtype=bf16`, or reduce `data.num_workers`.

4. **NaN gradients**: This is a known issue. Use the stable training versions (`train_mmdit_stable.py` and `improved_trainer_stable.py`) which include:
   - Loss validation before backward pass
   - Gradient clipping before NaN checks
   - Numerical stability improvements (epsilon values, normalization)
   - Conservative hyperparameters (lower LR, reduced gradient clip)
   - Automatic bad batch skipping

5. **Distributed hangs**: Verify NCCL environment variables (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`) are consistent across nodes.

### Stable Training Configuration

The stable training script uses these conservative hyperparameters to prevent NaN gradients:

```bash
# Environment variable overrides for stable training
LEARNING_RATE=5e-5              # Default: 5e-5 (reduced from 1e-4)
GRAD_CLIP_NORM=0.5              # Default: 0.5 (reduced from 1.0)
WARMUP_STEPS=2000               # Default: 2000 (increased from 1000)
LATENT_LOSS_WEIGHT=0.1          # Default: 0.1 (reduced from 1.0)
GRAD_ACCUM_STEPS=2              # Default: 2 (added for stability)

# Example: Even more conservative settings
LEARNING_RATE=3e-5 GRAD_CLIP_NORM=0.3 WARMUP_STEPS=3000 \
bash train_qwen_english_l2t_stable.sh
```

### Monitoring Training

```bash
# Watch logs in real-time
tail -f train_logs/train_*_node0.log

# Monitor for errors and NaN
tail -f train_logs/train_*_node0.log | grep -E "Loss:|ERROR|NaN"

# Check gradient norms (should be < 1.0)
tail -f train_logs/train_*_node0.log | grep "grad_norm"

# Find all errors in completed run
grep -E "ERROR|FAILED" train_logs/train_*_node0.log
```

## Supported Embedding Models

The preprocessing pipeline supports multiple embedding models for latent extraction:

| Model | Dimension | Command Flag | Notes |
|-------|-----------|--------------|-------|
| **SONAR** | 1024 | `--latent-model sonar` | Facebook SONAR, good multilingual support |
| **E5** | 1024 | `--latent-model e5` | intfloat/multilingual-e5-large, strong performance |
| **Qwen** | 32-1024 | `--latent-model qwen` | Qwen/Qwen3-Embedding-0.6B, lower memory |
| **T5** | 1024 | `--latent-model t5` | google/t5-v1_1-large encoder |
| **BGE** | varies | `--latent-model bge` | BAAI/bge-m3, supports long sequences |

**Critical**: The embedding dimension from preprocessing must match `model.latent_dim` in your training config. For example, if you preprocess with Qwen at 32 dimensions, set `model.latent_dim=32` in training.

## Getting Started Workflow

### 1. First-Time Setup

```bash
# Clone and setup environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM
source /path/to/.venv/bin/activate  # Or create new venv

# Verify GPU access
nvidia-smi
```

### 2. Preprocess Data

```bash
# Extract embeddings (example with E5 on 4 GPUs)
torchrun --nnodes=1 --nproc_per_node=4 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model e5 \
  --batch-size 256 \
  --max-samples 10000000 \
  --output-dir preprocessed_data/e5_1024d_full

# Verify output structure
ls preprocessed_data/e5_1024d_full/texts/train/
ls preprocessed_data/e5_1024d_full/latents/train/
cat preprocessed_data/e5_1024d_full/train_data.json | head
```

### 3. Test Training (Single Node)

```bash
# Quick test with 2 GPUs
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# Monitor logs
tail -f train_logs/train_*_node0.log
```

### 4. Monitor for Stability

Watch the first 1000 steps for:
- ✅ No "ERROR" messages
- ✅ Loss decreasing (should drop from ~3.0 to ~2.0 in first few hundred steps)
- ✅ Gradient norms < 1.0
- ✅ No NaN warnings

### 5. Scale Up

Once stable on single node:
```bash
# Scale to 2 nodes (16 GPUs)
NNODES=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# Continue monitoring
tail -f train_logs/train_*_node0.log | grep -E "Loss:|ERROR"
```

## Testing

Run single-node test:
```bash
bash scripts/utils/test_single_node.sh
```

Run distributed setup test:
```bash
bash scripts/utils/test_distributed_setup.sh
```

## Baseline Models

To run baseline MDLM or latent DIT models, see `baseline/` and `baseline_latent/` directories. These use separate training scripts (`baseline/train.py`, `baseline_latent/train_latent_dit.py`) with their own config systems.

Example baseline commands:

```bash
# MDLM baseline
torchrun --nnodes 1 --nproc_per_node 1 baseline/train.py \
  --config-name mdlm \
  logging.run_name="test-openwebtext" \
  data.dataset_name="openwebtext" \
  training.train_batch_size=4

# Latent DIT baseline
torchrun --nnodes 1 --nproc_per_node 1 baseline_latent/train_latent_dit.py \
  --config-name mdlm_latent \
  logging.run_name="latent-full-8M" \
  model.latent_dim=768
```

## Additional Documentation

The repository contains several detailed documentation files in the `docs/` directory:

- **docs/STABLE_TRAINING_GUIDE.md** - Comprehensive guide for the stable training script
- **docs/START_HERE.md** - Quick start guide with complete overview
- **docs/QUICK_REFERENCE.md** - One-page cheat sheet for common commands
- **docs/NAN_FIX_SUMMARY.md** - Summary of NaN gradient fixes
- **docs/NAN_GRADIENT_FIXES.md** - Detailed technical analysis of NaN issues
- **docs/STABLE_FILES_COMPLETE.md** - Overview of stable version files
- **docs/DISTRIBUTED_DEBUG_GUIDE.md** - Debugging distributed training issues
- **docs/TROUBLESHOOTING_GUIDE.md** - Common issues and solutions
- **docs/MIGRATION_GUIDE.md** - Guide for migrating between versions
- **docs/OPTIMIZATION_CHEATSHEET.md** - Quick optimization reference

Refer to these files for more detailed information on specific topics.
