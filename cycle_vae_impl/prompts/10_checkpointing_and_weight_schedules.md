# Prompt 10: Checkpointing and weight schedules

TASK
- Add isolated checkpoints and weight schedules in the training entrypoint.

CHECKPOINTS
- Save under: `cycle_vae_impl/outputs/<run_name>/checkpoints/`
- Support resume from a checkpoint path.
- Save fields:
  - model_state_dict
  - optimizer_state_dict
  - step
  - cfg
  - RNG state (torch + cuda)

WEIGHT SCHEDULES
- Add warmup/ramp support for cycle weights.
- Config keys:
  - `loss.cycle_text_weight`
  - `loss.cycle_text_warmup_steps`
  - `loss.cycle_text_ramp_steps`
  - `loss.cycle_latent_weight`
  - `loss.cycle_latent_warmup_steps`
  - `loss.cycle_latent_ramp_steps`

VERIFY
- Smoke forward still works.
- Synthetic train can resume from a saved checkpoint.
