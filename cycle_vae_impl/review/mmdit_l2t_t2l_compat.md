# Compatibility review: cycle_vae_impl vs existing MMDiT L2T/T2L

Date: 2026-02-02

Scope:
- Compare the isolated implementation in `cycle_vae_impl/` with the existing MMDiT-based L2T/T2L trainer under `latentDLM_mmdit/`.
- Focus on forward contract, timestep semantics, masking policy, parameterization, and pretrained-weights compatibility.

Primary reference files (existing):
- `latentDLM_mmdit/models/multimodal_mmdit.py`
- `latentDLM_mmdit/improved_trainer_stable.py`
- `latentDLM_mmdit/continuous_diffusion.py`

Primary implementation files (isolated):
- `cycle_vae_impl/trainers/cycle_vae_trainer.py`
- `cycle_vae_impl/entrypoints/train_cycle_vae.py`
- `cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml`

## 1) Model forward contract: OK

Existing `MultimodalMMDiT.forward` signature:
- `forward(text_tokens, latents, text_timesteps, latent_timesteps, attention_mask=None)`
- Returns:
  - `(text_logits, latent_pred)`
  - or `(text_logits, latent_pred, text_cluster_logits)` if `cluster_size > 0`

Key behaviors:
- `text_timesteps` and `latent_timesteps` can be `None`. `MultimodalConditioning` maps `None` to zeros.
- `attention_mask` is converted to boolean inside the model.
- Latents accept `[B,D]` or `[B,1,D]` (and some `[B,S,D]` reduced by mean).

Note on external `mmdit` expectations:
- The upstream `mmdit` library expects a `time_cond` embedding vector, not raw scalar timesteps.
- This repo satisfies that by computing `time_cond` via `MultimodalConditioning(TimestepEmbedder(...))` inside `MultimodalMMDiT`.

cycle_vae_impl usage:
- Calls use keyword args matching the real model.
- Latents are passed as `[B,1,D]`.
- `attention_mask` is passed through.

## 2) Existing L2T/T2L semantics (stable trainer)

From `latentDLM_mmdit/improved_trainer_stable.py`:

L2T:
- `text_t = ones(B)` (full noise)
- `noisy_input_ids = [MASK]` everywhere
- latents are clean conditioning; `latent_t = None`
- text loss is computed over all positions (mask is all-true)

T2L:
- text is clean conditioning; `text_t = None`
- latents are noised at `latent_t = ones(B)` (full noise)
- latent loss is computed (normalized MSE), scaled by `loss.latent_loss_weight`.

## 3) cycle_vae_impl semantics vs existing: matches and mismatches

## 3.0 Compatibility matrix (existing vs isolated)

| Topic | Existing (latentDLM_mmdit stable) | cycle_vae_impl default | Risk | Recommended for pretrained reuse |
|------|-----------------------------------|------------------------|------|----------------------------------|
| Model forward signature | `text_tokens, latents, text_timesteps, latent_timesteps, attention_mask` | same keywords | low | OK |
| None timesteps | allowed (mapped to zeros) | uses tensors; can use zeros | low | OK |
| L2T text masking | full mask, CE over all positions | full mask, CE over all positions | low | OK |
| T2L latent timestep | `latent_t = 1` (full noise) | configurable via `loss.latent_t_mode` | high | set `loss.latent_t_mode=full` |
| Latent loss form | normalized MSE, scaled by `latent_loss_weight` | normalized MSE, scaled by `latent_loss_weight` | low | OK |
| Cycle terms | none | adds `cycle_text` and `cycle_latent` | medium | start weights low + ramp |
| x0 reconstruction | not used in base trainer | used for `cycle_text` via `predict_x0_from_{eps,v}` | medium | keep stop-grad; start small |

### 3.1 Matches

- L2T masking: cycle_vae_impl uses full-mask L2T (all tokens masked; CE over all positions). This matches stable L2T.
- Latent loss normalization: cycle_vae_impl uses normalized MSE (same style as stable trainer).
- Parameterization handling for base T2L loss: cycle_vae_impl uses `ContinuousDiffusion.add_noise(...)` with `latent_parameterization` set in config.

### 3.2 Mismatch that matters for pretrained reuse (fixed)

Pretrained T2L compatibility depends heavily on the latent timestep distribution.

Existing stable T2L trains at:
- `latent_t = ones(B)` (t=1) only.

Old cycle_vae_impl behavior:
- sampled random `t_lat in [t_min, t_max]` for T2L passes.
This is a distribution shift if you reuse pretrained weights trained at t=1.

Fix implemented in cycle_vae_impl:
- Added `loss.latent_t_mode` in `cycle_vae_impl/trainers/cycle_vae_trainer.py`:
  - `random`: sample `t` in `[cycle_latent_t_min, cycle_latent_t_max]`
  - `full`: force `t = 1` (matches existing stable T2L)

Cluster default:
- `cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml` sets `loss.latent_t_mode: full`.

## 4) Cycle losses are additive (not present in existing trainer)

cycle_vae_impl adds:
- cycle_text (T2L2T): uses `x0_pred` derived from the T2L prediction and conditions an L2T pass.
- cycle_latent (L2T2L): pseudo-label tokens from L2T logits (argmax/categorical), then runs a T2L loss conditioned on pseudo tokens.

These are not part of the existing stable objective; treat them as new regularizers. To reduce destabilization risk:
- keep `loss.cycle_stop_grad_latent: true`
- start with low cycle weights and ramp.

## 5) Recommended settings for first real run on cluster

If you are reusing pretrained T2L weights trained with the stable trainer:
- `loss.latent_t_mode: full`
- ramp cycle weights:
  - `loss.cycle_text_warmup_steps: 0` and `loss.cycle_text_ramp_steps: N`
  - `loss.cycle_latent_warmup_steps: 0` and `loss.cycle_latent_ramp_steps: N`
- start small:
  - `loss.cycle_text_weight: 0.1`
  - `loss.cycle_latent_weight: 0.1`

Only after stable training, consider `loss.latent_t_mode: random` with a curriculum that stays near 1 initially.

## 6) What this review does NOT guarantee

- It does not prove training quality or convergence.
- It does not guarantee that cycle objectives improve downstream sampling.
- It does not validate the external `mmdit` dependency installation.
