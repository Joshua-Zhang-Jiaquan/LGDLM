# Prompt 01: Repo reference map

TASK
- Pin the exact existing modules this implementation will import and mirror.

MUST READ (reference only; do not modify)
- `latentDLM_mmdit/train_mmdit_stable.py` (overall wiring: model/tokenizer/dataloader)
- `latentDLM_mmdit/improved_trainer_stable.py` (loss math + stability clamps)
- `latentDLM_mmdit/models/multimodal_mmdit.py` (model forward signature + return tuple)
- `latentDLM_mmdit/continuous_diffusion.py` (latent diffusion add_noise + predict_x0 helpers)
- `latentDLM_mmdit/diffusion_process.py` (MaskedDiffusion for text)
- `latentDLM_mmdit/modeling_mmdit.py` (get_tokenizer)

CONTRACTS TO COPY
- Model forward:
  - `model(text_tokens, latents, text_timesteps, latent_timesteps, attention_mask)`
  - returns `(text_logits, latent_pred)` or `(text_logits, latent_pred, cluster_logits)`
- Latents shapes:
  - dataset batch provides `latent` as [B,D] or [B,1,D]
  - model expects `latents` as [B,1,D]
- Timesteps:
  - text full-mask uses `text_t = ones(B)`
  - latent diffusion should avoid `t=1` for stability in cycle training; sample `t ~ U(t_min, t_max)` with `t_max < 1`
