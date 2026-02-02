# Prompt 05: Add cycle_text (T2L2T)

TASK
- Add a second L2T pass conditioned on predicted x0 from the T2L pass.

STOP-GRAD DEFAULT
- Detach predicted x0 for cycle_text by default.

MUST DO
- Make the latent parameterization explicit and correct:
  - epsilon/x0/v_param
