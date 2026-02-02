# Prompt 09: Add cycle_latent (L2T2L)

TASK
- Extend CycleVAETrainer to compute an L2T2L cycle loss.

DEFINITION
- L2T2L (latent -> text -> latent)
  - Start from ground-truth clean latents z.
  - Compute L2T logits with full-mask text conditioned on z.
  - Produce pseudo tokens y_hat via argmax (default) or categorical sampling.
  - Run a T2L latent denoising loss conditioned on y_hat.
  - Compare latent prediction to target like the base T2L loss.

GRADIENT POLICY
- Tokens are discrete pseudo-labels; default is stop-grad implicitly.

CONFIG KEYS
- `loss.cycle_latent_weight`
- `loss.cycle_latent_sampling` in {"argmax", "categorical"}

LOG METRICS
- `cycle_latent_loss`, `cycle_latent_weight`, `cycle_latent_sampling`
