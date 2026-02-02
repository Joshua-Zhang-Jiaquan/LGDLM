# Prompt 00: Overview and constraints

TASK
- Implement a new training variant named `cycle_vae` in an isolated folder.

HARD CONSTRAINTS
- Do not edit any existing files under `latentDLM_mmdit/`.
- Only add new files under `cycle_vae_impl/`.
- No commits.
- Keep files ASCII-only.

DEFINITION OF DONE
- There is a runnable entrypoint `cycle_vae_impl/entrypoints/train_cycle_vae.py` that uses a new trainer and can be launched without touching the original entrypoints.
- There is a smoke forward script that runs on CPU with a synthetic batch and produces finite losses.
- The new trainer computes:
  - base_t2l_latent_loss (random latent timestep)
  - base_l2t_text_loss (full-mask text conditioned on ground-truth latents)
  - cycle_text_loss (T2L2T): predicted latent x0 (from T2L) -> L2T text CE

NON-GOALS (for this implementation)
- No L2T2L cycle (latent cycle) yet.
- No new dependencies.
- No full training quality claims.
