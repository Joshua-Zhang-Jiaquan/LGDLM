# Plan 02: Trainer wiring plan (no code yet)

TASK
- Design how to integrate the cycle losses into the existing training code.

FILES (to be edited in the future implementation)
- `latentDLM_mmdit/improved_trainer.py`
- `latentDLM_mmdit/improved_trainer_stable.py`

WHAT TO DO
1) Identify the exact control point:
   - `MultimodalDiffusionTrainer_new.forward` computes mode and builds `noisy_input_ids`, `noisy_latents`, then runs the model and computes base losses.
2) Specify a new `loss_type` option (e.g., `cycle_vae`) that:
   - still computes base losses
   - additionally computes cycle losses based on the same batch
3) Specify how to avoid shape mismatches:
   - latents are either [B, D] or [B, 1, D] in the model call; be explicit about reshaping.
4) Specify how to log:
   - add numeric entries to `metrics` (cycle_text_loss, cycle_latent_loss, etc.)

MUST NOT DO
- Do not add `as any` or type suppression.

ACCEPTANCE CRITERIA
- `ulw-work/research/vae_cycle_report.md` Section 6 lists:
  - the exact class/function names to modify
  - the exact metric keys to add
  - an explicit pseudocode snippet (not actual code) for the new loss_type branch

VERIFY
- Confirm the training loop in `latentDLM_mmdit/train_mmdit_stable.py` logs all numeric metrics (so adding keys is sufficient).
