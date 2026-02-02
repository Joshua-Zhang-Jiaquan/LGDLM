# cycle_vae_impl cluster runbook

This runbook is for running `cycle_vae_impl` on a GPU training machine/cluster.

Scope:
- Uses the isolated entrypoint `cycle_vae_impl/entrypoints/train_cycle_vae.py`.
- Does not depend on editing `latentDLM_mmdit/`.

References:
- torchrun docs: https://docs.pytorch.org/docs/stable/elastic/run.html
- DDP fault tolerance tutorial: https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html

## 0) Preflight

1) Verify python environment

```bash
python -V
python -c "import torch; print(torch.__version__)"
python -c "import yaml; print('pyyaml ok')"
```

2) Verify GPUs (on training machine)

```bash
nvidia-smi
python -c "import torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count())"
```

3) Verify the real MMDiT dependency exists

```bash
python -c "from mmdit.mmdit_generalized_pytorch import MMDiT; print('mmdit ok')"
```

If this fails, install/provide the `mmdit` python package in your training environment.

## 1) Prepare config

Use:
- `cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml`

You must set (via editing the yaml or overrides):
- `tokenizer.path`
- `data.token_dir`, `data.latent_dir` (and optionally `data.data_dir`, `data.data_files.*`)

Hard safety:
- `model.use_stub_if_no_mmdit` is `false` in the cluster config; jobs fail fast if the real model is unavailable.

## 2) Single-node launch (recommended first)

One GPU:

```bash
torchrun --standalone --nproc_per_node=1 \
  cycle_vae_impl/entrypoints/train_cycle_vae.py \
  --config cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml \
  --override tokenizer.path=/path/to/tokenizer \
  --override data.token_dir=/path/to/tokens/train \
  --override data.latent_dir=/path/to/latents/train \
  --override logging.run_name=cycle-vae-cluster-test \
  --override training.num_train_steps=500 \
  --override logging.log_every=10 \
  --override logging.save_every=100
```

Multiple GPUs on one node:

```bash
torchrun --standalone --nproc_per_node=8 \
  cycle_vae_impl/entrypoints/train_cycle_vae.py \
  --config cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml \
  --override tokenizer.path=/path/to/tokenizer \
  --override data.token_dir=/path/to/tokens/train \
  --override data.latent_dir=/path/to/latents/train
```

Outputs:
- `cycle_vae_impl/outputs/<run_name>/training_log.jsonl`
- future checkpoints (once enabled): `cycle_vae_impl/outputs/<run_name>/checkpoints/`

## 3) Multi-node launch (template)

Node 0:

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=node0:29500 \
  --node_rank=0 \
  cycle_vae_impl/entrypoints/train_cycle_vae.py \
  --config cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml \
  --override tokenizer.path=/path/to/tokenizer \
  --override data.token_dir=/path/to/tokens/train \
  --override data.latent_dir=/path/to/latents/train
```

Node 1:

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=node0:29500 \
  --node_rank=1 \
  cycle_vae_impl/entrypoints/train_cycle_vae.py \
  --config cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml \
  --override tokenizer.path=/path/to/tokenizer \
  --override data.token_dir=/path/to/tokens/train \
  --override data.latent_dir=/path/to/latents/train
```

## 4) Recommended override knobs

Stability knobs:

```bash
--override loss.latent_t_mode=full \
--override loss.cycle_latent_t_max=0.90 \
--override loss.cycle_text_weight=0.1
```

Pretrained compatibility knob (important if your T2L checkpoint was trained with `latent_t=1`):

```bash
--override loss.latent_t_mode=full
```

Enable categorical pseudo-labeling for L2T2L (once implemented):

```bash
--override loss.cycle_latent_sampling=categorical
```

## 5) Debugging

If distributed hangs:

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
```

If you need sync CUDA errors:

```bash
export CUDA_LAUNCH_BLOCKING=1
```
