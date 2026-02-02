# Prompt 04: Implement base losses

TASK
- Implement base losses inside `CycleVAETrainer.forward`.

BASE T2L
- Use clean text (`input_ids`) as conditioning.
- Sample `t_lat` randomly.
- Add noise to ground-truth latents and compute the latent denoising loss.

BASE L2T
- Use full-mask `text_tokens` and `text_timesteps = ones(B)`.
- Condition on ground-truth latents.
- Compute text CE over all tokens.

MUST NOT DO
- Do not use dataset sampling scripts.
