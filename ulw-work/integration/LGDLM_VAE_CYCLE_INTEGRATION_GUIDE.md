# LGDLM VAE/Cycle Integration Guide (LLM-friendly)

Goal
- Integrate a VAE-style bidirectional objective into the original LGDLM training framework (`latentDLM_mmdit/`) using existing L2T and T2L modes as decoder/encoder.
- The objective is cycle-consistent:
  - T2L2T (text -> latent -> text) = cycle_text
  - L2T2L (latent -> text -> latent) = cycle_latent

Non-goals
- This guide does not claim training will converge; it gives the minimal, correct wiring plan.
- This guide does not require changing the MMDiT architecture.

Where this fits in the repo
- Entry points:
  - `latentDLM_mmdit/train_mmdit.py`
  - `latentDLM_mmdit/train_mmdit_stable.py` (recommended)
- Trainer/loss wiring:
  - `latentDLM_mmdit/improved_trainer.py`
  - `latentDLM_mmdit/improved_trainer_stable.py` (recommended)
- Model forward contract:
  - `latentDLM_mmdit/models/multimodal_mmdit.py`

Working reference implementation (already in this repo)
- Isolated cycle objective implementation:
  - `cycle_vae_impl/trainers/cycle_vae_trainer.py`
  - `cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml`
  - `cycle_vae_impl/review/mmdit_l2t_t2l_compat.md`

The core design

1) Map to VAE terms
- Decoder p_theta(y|z): L2T mode (clean latents, fully masked text -> reconstruct tokens)
- Encoder q_phi(z|y): T2L mode (clean text, noisy latents -> reconstruct latent)

2) Cycle losses

Cycle-text (T2L2T)
- Input: clean tokens y
- Step 1 (T2L): predict latent x0 (or equivalent) from noisy latent z_t at timestep t
- Step 2 (L2T): feed predicted latent (x0_pred) as conditioning and reconstruct y
- Loss: CE(y_logits, y)

Cycle-latent (L2T2L)
- Input: clean latent z
- Step 1 (L2T): predict text logits from fully masked text conditioned on z
- Step 2 (pseudo-label): choose tokens y_hat via argmax or categorical sampling
- Step 3 (T2L): predict latent from y_hat
- Loss: MSE(latent_pred, latent_target)

3) Stability rule for pretrained compatibility
- Existing stable T2L trains with latent_t = 1 (full noise).
- If you reuse pretrained T2L weights, random t is a distribution shift.
- Add a config knob for latent timestep policy:
  - full: latent_t = ones(B)
  - random: latent_t sampled in [t_min, t_max]

Minimal integration plan (recommended: stable path)

## Step A: Add new config file

Create: `latentDLM_mmdit/configs/mmdit_cycle_vae.yaml`

Minimum keys to add under `loss:`
- loss_type: cycle_vae
- latent_loss_weight
- cycle_text_weight
- cycle_text_warmup_steps
- cycle_text_ramp_steps
- cycle_latent_weight
- cycle_latent_warmup_steps
- cycle_latent_ramp_steps
- cycle_latent_sampling: argmax|categorical
- latent_t_mode: full|random
- cycle_latent_t_min
- cycle_latent_t_max
- cycle_stop_grad_latent: true|false

Recommended first values (pretrained-safe)
- latent_t_mode: full
- cycle_stop_grad_latent: true
- cycle_text_weight: 0.1
- cycle_latent_weight: 0.1

## Step B: Implement cycle_vae in the stable trainer

Modify: `latentDLM_mmdit/improved_trainer_stable.py`

Target location
- In `MultimodalDiffusionTrainer_new.forward(self, batch, step=None)`, after you compute:
  - base text loss (when L2T active)
  - base latent loss (when T2L active)

Add a new loss_type branch
- If `self.loss_type == "cycle_vae"`:
  - Compute base_l2t_text_loss and base_t2l_latent_loss for the same batch.
  - Compute cycle_text_loss (T2L2T).
  - Compute cycle_latent_loss (L2T2L).
  - Combine:

Pseudocode (use existing code patterns; avoid refactors)

```python
if self.loss_type == "cycle_vae":
    # 1) base L2T (text CE): full mask + clean latents
    # 2) base T2L (latent MSE): clean text + noisy latents
    # 3) cycle_text: T2L -> x0_pred -> L2T CE
    # 4) cycle_latent: L2T logits -> y_hat -> T2L latent MSE

    # IMPORTANT: latent_t distribution
    if config.loss.latent_t_mode == "full":
        t_lat = ones(B)
    else:
        t_lat = uniform(config.loss.cycle_latent_t_min, config.loss.cycle_latent_t_max)

    # x0_pred reconstruction depends on latent_parameterization
    if param == "epsilon":
        x0_pred = latent_diffusion.predict_x0_from_eps(z_t, t_lat, eps_pred)
    elif param == "x0":
        x0_pred = latent_pred
    elif param == "v_param":
        x0_pred = latent_diffusion.predict_x0_from_v(z_t, t_lat, v_pred)

    if config.loss.cycle_stop_grad_latent:
        x0_cycle = x0_pred.detach()
    else:
        x0_cycle = x0_pred

    # cycle weights with warmup/ramp
    w_text = ramp(step, base=cycle_text_weight, warmup=cycle_text_warmup_steps, ramp=cycle_text_ramp_steps)
    w_lat  = ramp(step, base=cycle_latent_weight, warmup=cycle_latent_warmup_steps, ramp=cycle_latent_ramp_steps)

    total = base_l2t_text_loss + base_t2l_latent_loss_weighted + w_text*cycle_text_loss + w_lat*cycle_latent_loss
    metrics.update({
        "cycle_text_loss": float(cycle_text_loss.item()),
        "cycle_latent_loss": float(cycle_latent_loss.item()),
        "cycle_text_weight": float(w_text),
        "cycle_latent_weight": float(w_lat),
        "latent_t_mode": str(config.loss.latent_t_mode),
    })
    return total, metrics
```

Implementation shortcut (allowed)
- You can copy the concrete logic from `cycle_vae_impl/trainers/cycle_vae_trainer.py` into the stable trainer branch.
- Keep it minimal: do not refactor existing trainer modes.

## Step C: Make the stable entrypoint runnable

Use the stable entrypoint with your new config:

```bash
torchrun --standalone --nproc_per_node=1 \
  latentDLM_mmdit/train_mmdit_stable.py \
  --config-name mmdit_cycle_vae \
  loss.loss_type=cycle_vae
```

Note:
- `latentDLM_mmdit/train_mmdit_stable.py` constructs `MultimodalDiffusionTrainer_new` and calls `ddp_trainer(batch, step=state.step)`.
- That means your cycle_vae implementation must live in the trainer and accept `step`.

## Step D: Logging requirements

Existing behavior
- `latentDLM_mmdit/train_mmdit_stable.py` writes `training_log.jsonl` with `loss` and numeric `metrics` keys.

Requirements
- Ensure cycle losses are numeric floats in metrics.
- Avoid non-numeric metrics (strings) in eval aggregation.

Recommended metric keys
- cycle_text_loss
- cycle_latent_loss
- cycle_text_weight
- cycle_latent_weight
- latent_t_mode (string; do not include in eval aggregation or gate it)

Testing and verification

## Local Mac (no training)
- Use the isolated smoke tests:
  - `cycle_vae_impl/entrypoints/smoke_cycle_vae_forward.py`
  - `cycle_vae_impl/entrypoints/train_cycle_vae.py --synthetic`

## Training machine
- First run should match pretrained semantics:
  - `loss.latent_t_mode=full`
  - low cycle weights and ramps

Common failure modes
- mmdit import missing: `latentDLM_mmdit/models/multimodal_mmdit.py` raises ImportError if mmdit is not installed.
- x0_pred explosion: if you use random t with a T2L checkpoint trained at t=1, x0_pred may be off-manifold.
- feedback loops: do not allow gradients through pseudo tokens; keep stop-grad on x0_pred at first.
