# Plan 01: Objective specification (base + cycle + regularizers)

TASK
- Specify an implementable objective that uses existing denoising losses plus cycle-consistency terms.

REQUIRED OUTPUT
- A crisp equation-level spec with weights and gradient-flow policy.

WHAT TO DO
1) Define base losses using repo semantics:
   - L_l2t_base: CE reconstruction in L2T mode.
   - L_t2l_base: latent denoising MSE in T2L mode.
2) Define cycle losses:
   - L_cycle_text (T2L2T): y -> z_hat -> y_recon.
   - L_cycle_latent (L2T2L): z -> y_hat -> z_recon.
3) For each cycle, define the default gradient policy:
   - start with stop-grad on the bridge variable (z_hat or y_hat).
   - specify when and how to relax it (optional).
4) Add ELBO-inspired regularization:
   - R_latent_prior (moment matching / norm constraints) as a first implementation.
5) Specify scheduling:
   - cycle_warmup_steps
   - cycle_ramp_steps

MUST NOT DO
- Do not require full diffusion sampling inside the training graph as the first implementation.

ACCEPTANCE CRITERIA
- `ulw-work/research/vae_cycle_report.md` Section 4 contains:
  - explicit formulas for L_total
  - a table listing each weight and its default value range
  - a table listing stop-grad flags and defaults

VERIFY
- All referenced loss names and modes exist in the trainer.
