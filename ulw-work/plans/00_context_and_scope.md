# Plan 00: Context and scope lock

TASK
- Create a VAE-style cycle-consistency optimization design for existing L2T/T2L modes.

CONTEXT (repo facts)
- Training modes are implemented in `latentDLM_mmdit/improved_trainer.py` and `latentDLM_mmdit/improved_trainer_stable.py`.
- L2T = text reconstruction conditioned on clean latents.
- T2L = latent denoising conditioned on clean text.

WHAT TO DO
1) Read and summarize mode semantics from `latentDLM_mmdit/improved_trainer.py` and `latentDLM_mmdit/improved_trainer_stable.py`.
2) Lock the naming (decision already made):
   - Use T2L2T (text -> latent -> text).
   - Treat "L2T2T" as a typo; use L2T2L (latent -> text -> latent).
3) Write a short scope section for the design:
   - in-scope: objective design + trainer integration plan + ablations + smoke tests
   - out-of-scope (for now): full-scale training runs; new large dependencies; RL gradients through discrete sampling

MUST NOT DO
- Do not edit training code yet.

ACCEPTANCE CRITERIA
- A one-page statement added to `ulw-work/research/vae_cycle_report.md` Section 3 is precise enough that no one can misunderstand what "cycle" means.

VERIFY
- Paths referenced exist:
  - `latentDLM_mmdit/improved_trainer.py`
  - `latentDLM_mmdit/improved_trainer_stable.py`
