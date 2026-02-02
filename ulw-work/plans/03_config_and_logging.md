# Plan 03: Config + logging plan

TASK
- Define configuration keys and logging needed to support cycle training.

FILES (to be edited in the future implementation)
- `latentDLM_mmdit/configs/mmdit_stable.yaml`
- optionally `latentDLM_mmdit/configs/mmdit.yaml`

WHAT TO DO
1) Define config keys under `loss:`:
   - cycle_text_weight
   - cycle_latent_weight
   - cycle_warmup_steps
   - cycle_ramp_steps
   - stopgrad_cycle_text (bool)
   - stopgrad_cycle_latent (bool)
   - latent_prior_weight
2) Define which metrics to log:
   - train/cycle_text_loss
   - train/cycle_latent_loss
   - train/latent_prior_penalty
   - train/active_params_ratio (already exists in stable trainer)

MUST NOT DO
- Do not hardcode cluster paths or local machine paths in the plans.

ACCEPTANCE CRITERIA
- The report includes a config snippet example that can be pasted into YAML.

VERIFY
- Confirm `latentDLM_mmdit/configs/mmdit_stable.yaml` currently uses `loss:` and that adding keys there is consistent.
