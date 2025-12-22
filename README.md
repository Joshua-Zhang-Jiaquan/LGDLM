# MM-DLM — Multimodal Latent Diffusion Language Models (focused)

This repository contains code for latent reasoning and multimodal MMDiT models (text ↔ latent, image ↔ latent) and accompanying preprocessing and baseline code. The README here focuses on the latent/text/image MMDiT workflows and how to preprocess data, run encoders, and start training the latent models. Baseline code (HDLM, MDLM, AR, GIDD+) is preserved in the `baseline` and `baseline_latent` folders.

This repo now contains (high-level):

- `baseline/` and `baseline_latent/` — baseline reproductions (HDLM, MDLM, AR, GIDD+, and other baselines).
- `latentDLM_mmdit/` — latent MMDiT (text-to-latent / latent reasoning) training and utilities (train_mmdit.py, trainer_multimodal.py, etc.).
- `latentIMG_mmdit/` — image+text MMDiT training scripts and image patch encoders (continuous & discrete training variants).
- `preprocessed_data/` — data preprocessing helpers and distributed latent extraction: `prepare_data_multi_gpu.py` and usage examples in its README.
- `baseline_latent/` — additional latent-focused baseline code (train_latent_dit.py, trainer_latent.py, etc.).

Note: This top-level README intentionally focuses on the latent/text/image workflows. If you need to run the original HDLM unconditional generation or NeurIPS experiment reproductions, see the `baseline/` and `baseline_latent/` folders — those baselines are retained.

---

## Quick start

1. Create a Python virtual environment and install requirements:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optionally: pip install -e .
```

2. Preprocess / extract text latents (if you plan to train continuous latent models): follow `preprocessed_data/README` (it contains example `torchrun` commands using `prepare_data_multi_gpu.py`).

Example preprocessing (from `preprocessed_data/README`):

```bash
# SONAR on 2 GPUs
torchrun --nnodes=1 --nproc_per_node=2 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model sonar \
  --batch-size 128 \
  --max-samples 10000000 \
  --output-dir sonar_1024d_full

# E5 (multilingual-e5-large) on 4 GPUs
torchrun --nnodes=1 --nproc_per_node=4 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model e5 \
  --batch-size 256 \
  --max-samples 10000000 \
  --output-dir e5_1024d_full

# Qwen embedding model on 2 GPUs (lower memory than large causal models)
torchrun --nnodes=1 --nproc_per_node=2 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model qwen \
  --batch-size 8 \
  --max-samples 10000000 \
  --output-dir qwen_1024d_full
```

The `preprocessed_data/README` has more details and checks.

---

## Data format and preprocessing

- The continuous training scripts expect precomputed continuous text embeddings saved as `.npy` files (and a JSON manifest `train_data.json` / `validation_data.json`). Use `prepare_data_multi_gpu.py` to create these.
- The discrete/masked training script expects tokenized captions (token IDs + attention masks). The `latentIMG_mmdit/README` explains differences between discrete and continuous pipelines.

Files created by `prepare_data_multi_gpu.py` (typical):

- output-dir/
  - texts/train/*.txt
  - latents/train/*.npy
  - train_data.json
  - validation_data.json (optional)

Make sure embedding dimensionality from the encoder matches the `--dim-text` / model config used by training scripts.

---

## Training latent MMDiT (text ↔ latent reasoning)

Primary training entrypoints for latent reasoning in this repo:

- `latentDLM_mmdit/train_mmdit.py` — main training script for multimodal latent MMDiT (text-to-latent and multimodal setups).
- `latentIMG_mmdit/train_image_continuous.py` — continuous diffusion training using precomputed text embeddings and learnable image encoder.
- `latentIMG_mmdit/train_image_discrete.py` — masked/discrete diffusion training on tokens.
- `baseline_latent/train_latent_dit.py` and `baseline_latent/train_cross_dit.py` — latent baseline training scripts.

Example single-node, single-process command (adjust for your environment):

```bash
# Latent multimodal MMDiT (example)
python latentDLM_mmdit/train_mmdit.py \
  --data-root /path/to/preprocessed_data/e5_1024d_full \
  --epochs 50 \
  --batch-size 32 \
  --dim-text 1024 \
  --dim-cond 256 \
  --output-dir outputs_latent_mmdit

# Image-continuous training (uses precomputed text latents)
python latentIMG_mmdit/train_image_continuous.py \
  --data-root /path/to/preprocessed_data/e5_1024d_full \
  --epochs 50 \
  --batch-size 8 \
  --dim-text 1024 \
  --dim-image 512 \
  --output-dir outputs_image_continuous

# Image-discrete (masked) training
python latentIMG_mmdit/train_image_discrete.py \
  --data-root /path/to/tokenized_coco_like_data \
  --epochs 50 \
  --batch-size 8 \
  --dim-text 768 \
  --dim-image 512 \
  --output-dir outputs_image_masked
```

For multi-GPU runs, wrap the training script with `torchrun` and ensure the training scripts initialize DDP correctly (the preprocessing script already supports distributed extraction).

---

## Baselines (kept for comparison)

The repo preserves baseline implementations. See:

- `baseline/` — HDLM (original hierarchical diffusion language model) and other baselines.
- `baseline_latent/` — latent-focused baselines for DIT-style latent training.

You can run baseline training via the scripts in those folders. The original README content for the baselines has been preserved there; consult the respective `README` or `configs/` files in each baseline folder for exact training commands.

### Example baseline run commands

Below are example single-node, single-process `torchrun` commands for common baseline runs (copied from the provided commands, with minor fixes and `training.dtype=bf16` applied to the `mmdit` example). Adjust paths, `--nnodes` and `--nproc_per_node` for your hardware.

```bash
# MDLM baseline (quick test)
torchrun --nnodes 1 --nproc_per_node 1 hdlm/train.py \
  --config-name mdlm \
  logging.run_name="'test-openwebtext'" \
  data.dataset_name="openwebtext" \
  data.dataset_subset=null \
  data.data_files.train=null \
  data.cache_dir="./data" \
  data.test_size=1000 \
  training.train_batch_size=4 \
  training.eval_batch_size=4 \
  training.num_train_steps=20 \
  logging.eval_freq=10 \
  logging.log_freq=5
```

```bash
# Latent DIT baseline (full latent training)
torchrun --nnodes 1 --nproc_per_node 1 hdlm/train_latent_dit.py \
  --config-name hdlm_latent \
  logging.run_name="latent-full-8M" \
  data.dataset_name="openwebtext" \
  data.latent_data_root="./data_root" \
  model.latent_dim=768 \
  model.use_latent_conditioning=true \
  training.train_batch_size=32 \
  training.eval_batch_size=32 \
  training.num_train_steps=250000 \
  training.compile_model=false
```

```bash
# Cross-attention latent baseline
torchrun --nnodes 1 --nproc_per_node 1 hdlm/train_cross_dit.py \
  --config-name hdlm_cross_attention \
  logging.run_name="cross-attention-training" \
  data.dataset_name="json" \
  data.latent_data_root="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM/mmdit/data_root" \
  data.data_files.train="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM/mmdit/data_root/train_data.json" \
  data.data_files.validation="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM/mmdit/data_root/train_data.json" \
  training.train_batch_size=16 \
  training.eval_batch_size=16 \
  training.num_train_steps=250000 \
  training.compile_model=false
```

```bash
# MMDiT training run (use bf16 for faster/memory-efficient runs on supported hardware)
torchrun --nnodes 1 --nproc_per_node 1 hdlm/train_mmdit.py \
  --config-name mmdit \
  logging.run_name="mmdit-training-h200" \
  training.train_batch_size=80 \
  training.eval_batch_size=80 \
  training.num_train_steps=25000 \
  training.compile_model=false \
  model.latent_dim=768 \
  training.dtype=bf16
```

---

## Notes and tips

- Always verify embedding dimension when switching encoders: mismatch between encoder embedding size and `--dim-text` will break training.
- For quick pipeline checks, set `--max-samples 10` to validate model loading, tokenizer/embedding loading, and I/O.
- If you plan to use Qwen or BGE models, ensure your environment has the necessary packages (device_map handling, FlagEmbedding if using BGE wrapper).

---

If you'd like, I can add a small script to validate encoder dimensions automatically and fail fast if dimensions mismatch; or I can add example `torchrun` multi-GPU training wrappers for the latent and image scripts. Tell me which and I'll add it.
<div align="center">

<h1>NeurIPS 2025 | Next Semantic Scale Prediction via<br>Hierarchical Diffusion Language Models</h1>

<div>
    <a href="https://homepage.zhouc.ai/" target="_blank">Cai Zhou</a><sup>1,*</sup> | 
    <a href="https://chenyuwang-monica.github.io/" target="_blank">Chenyu Wang</a><sup>1,*</sup> | 
    <a href="https://zdhnarsil.github.io/" target="_blank">Dinghuai Zhang</a><sup>2,*</sup> | 
    <a href="https://shangyuantong.github.io/" target="_blank">Shangyuan Tong</a><sup>1</sup> | 
    <a href="https://yifeiwang77.com/" target="_blank">Yifei Wang</a><sup>1</sup> |<br>
    <a href="https://stephenbates19.github.io/index.html" target="_blank">Stephen Bates</a><sup>1,†</sup> |
    <a href="https://people.csail.mit.edu/tommi/tommi.html" target="_blank">Tommi Jaakkola</a><sup>1,†</sup>
</div>
<br>
<div>
    <sup></sup><sup>1</sup> Massachusetts Institute of Technology   <sup>2</sup> Microsoft Research
</div>
<div>
    <sup>*</sup> Equal Contribution  <sup>†</sup> Equal Senior Supervision
</div>
<br>


[![arXiv](https://img.shields.io/badge/arXiv-2510.08632-b31b1b.svg)](https://arxiv.org/abs/2510.08632)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-4b44ce.svg)](https://neurips.cc/virtual/2025/poster/116380)


<div align="left"> 

## 📢 News
- [2025/09/18] HDLM is accepted to [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/116380)!
- [2025/10/12] Paper is available on [arXiv](https://arxiv.org/abs/2510.08632)!
- [2025/10/12] Code is released!

## 💻 Overview
<br>

<div style="width: 100%; text-align: center; margin:auto;">
    <img style="width:100%" src="HDLM.png">
</div>
<br>

We present Hierarchical Diffusion Language Model (HDLM), a novel framework for training discrete diffusion models via time-varying next-semantic scale prediction.
HDLM extends standard Masked Diffusion Model (MDM) by introducing intermediate hierarchies (termed cluster tokens) in between clean tokens and masked tokens.
In the forward process, each token is independently perturbed to its higher-level ancestor with more abstract semantics according to the scheduler, while in the reverse process the model progressively predicts the next, more detailed semantics. 
Taken together, HDLM provides a general time-varying next semantic scale prediction process for language modeling. We derive closed-form expressions for the diffusion Evidence Lower Bound (ELBO), and show that HDLM can be implemented in a flexible manner while including the existing MDM as a special case.
This repository contains all training and evaluation code necessary for reproducing the results in the paper.


## 🔧 Quick Start

Set up the environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## 🎈 Reproducing Experiments

### Training

#### Precomputing cluster dicts and embeddings
You can download our precalculated files in `hdlm/clusters` for existing numbers of clusters in [1, 2, 4, 8, 16, 32, 64, 128, 256] with [GPT-2 tokenizer](https://huggingface.co/openai-community/gpt2) on [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset using [GIDD](https://arxiv.org/abs/2503.04482) pretrained models, or preprocess by running `hdlm/compute_cluster.py` for customed numbers of clusters / tokenizers / datasets / pretrained models. Make sure the names / paths of these cluster files match `cluster_dict_path`, `cluster_embed_path` and `pretrained_model_name` in your training configs as in the examples.

#### Configs

To reproduce the training runs from the paper, you can use the following commands.
In this example, we are training on a single node with 8 GPUs, feel free to adjust the `--nnodes` and `--nproc_per_node` arguments to match your setup.

Whenever needed, feel free to change the checkpoint saving directory by adjusting `save_dir` in `hdlm/configs/logging/default.yaml`, and data storage directory by `cache_dir` in `hdlm/configs/data/defaults.yaml`.

Key hyperparameters include:
 * `cluster_size`: number of clusters ($n$ in the paper)
 * `gamma`: forward process schedule ($\gamma$ in the paper)
 * `p_perturb`: probability of stochastic perturbations ($1-\xi$ in the paper)

 You are also welcome to try out other model / training / loss hyperparameters.

(optional) Log into W&B with `wandb login` for experiment tracking or other disable via `wandb disabled`.

```bash
# HDLM-small-64
torchrun --nnodes 1 --nproc_per_node 8 hdlm/train.py --config-name hdlm-small-cluster_64-gamma_1.0-xi_1.0 logging.run_name="'small-hdlm-cluster_64-gamma_1.0-xi_1.0-owt'"

# GIDD+ baseline
torchrun --nnodes 1 --nproc_per_node 8 hdlm/train.py --config-name gidd logging.run_name="'small-gidd+-owt-pu=0.0'"

# MDLM baseline
torchrun --nnodes 1 --nproc_per_node 8 hdlm/train.py --config-name mdlm logging.run_name="'small-mdlm-owt'"

# AR baseline
torchrun --nnodes 1 --nproc_per_node 8 hdlm/train.py --config-name ar logging.run_name="'small-ar-owt'"
```


### Evaluation

There are also a couple of scripts to run inference and evaluate the trained models.

#### Generate samples
The following command will generate `num_samples=256` samples in `num_denoising_steps=512` iterations from the model checkpoint located at `path` and save them to `samples_dir=samples.pt`.
```bash
python hdlm/eval/generate_samples.py path=./outputs/path/to/checkpoint/ samples_dir=samples.pt num_samples=256 num_denoising_steps=512 batch_size=16
```

#### Generative PPL
Given a file containing samples generated with the `generate_samples.py` script, the following command will compute the generative PPL.
Here we assume that the diffusion model used to generate samples located at `samples.pt` uses the `gpt2` tokenizer, and we compute generative PPL using `gpt2-large` as a reference model.
The results will be saved to `metrics_path=metrics.json`.
```bash
python hdlm/eval/generative_ppl.py samples_path=samples.pt model_tokenizer=gpt2 pretrained_model=gpt2-large batch_size=1 metrics_path=metrics.json
```

#### Validation loss
A simple helper script to compute the loss of a trained model on the entire validation split.
```bash
python hdlm/eval/loss.py path=./outputs/path/to/checkpoint/ batch_size=32
```


## 📎 Citation 

If you find our work helpful, please consider giving a star ⭐ and citation 📝 
```bibtex
@article{zhou2025next,
  title={Next Semantic Scale Prediction via Hierarchical Diffusion Language Models},
  author={Zhou, Cai and Wang, Chenyu and Zhang, Dinghuai and Tong, Shangyuan and Wang, Yifei and Bates, Stephen and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2510.08632},
  year={2025}
}
```

## 💞 Acknowledgements
The code is built upon the below repositories, we thank all the contributors for open-sourcing.
* [GIDD](https://github.com/dvruette/gidd/)
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
