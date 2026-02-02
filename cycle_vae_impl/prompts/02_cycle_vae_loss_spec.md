# Prompt 02: cycle_vae loss spec

LOSS TYPE
- `loss.loss_type = cycle_vae`

TIMESTEP POLICY (latent)
- Sample `t_lat ~ Uniform(loss.cycle_latent_t_min, loss.cycle_latent_t_max)` and clamp to < 1.0.
- Do not use `t_lat == 1.0`.

TERMS
1) Base T2L (text -> latent) loss
- Condition on clean text.
- Diffuse ground-truth latents using sampled `t_lat`.
- Predict diffusion parameterization target (epsilon/x0/v).
- Use normalized MSE (match stable trainer style) and weight by `loss.latent_loss_weight`.

2) Base L2T (latent -> text) loss
- Fully mask the text input.
- Condition on ground-truth clean latents.
- Compute CE over all positions (mask is all True).

3) Cycle-text (T2L2T)
- From the T2L pass, reconstruct predicted latent x0:
  - if latent_parameterization == epsilon: `x0_pred = predict_x0_from_eps(xt, t_lat, eps_pred)`
  - if x0: `x0_pred = latent_pred`
  - if v_param: `x0_pred = predict_x0_from_v(xt, t_lat, v_pred)`
- Stop-grad default: `x0_cycle = x0_pred.detach()`.
- Run L2T again conditioned on `x0_cycle` and compute CE to the original tokens.

TOTAL
- `total = base_t2l + base_l2t + loss.cycle_text_weight * cycle_text`

LOG METRICS
- `base_t2l_latent_loss`, `base_l2t_text_loss`, `cycle_text_loss`, `cycle_text_weight`, `t_lat_mean`, `latent_pred_norm`, `x0_pred_norm`
