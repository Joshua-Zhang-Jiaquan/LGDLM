# Prompt 07: Verification smoke

TASK
- Create a smoke script and verification commands.

REQUIREMENTS
- Script: `cycle_vae_impl/entrypoints/smoke_cycle_vae_forward.py`
- Must build model + trainer on CPU by default.
- Must create a synthetic batch (input_ids, attention_mask, latent) and run `trainer(batch)`.

PASS/FAIL
- Loss is finite.
- Metrics dict includes:
  - base_t2l_latent_loss
  - base_l2t_text_loss
  - cycle_text_loss
