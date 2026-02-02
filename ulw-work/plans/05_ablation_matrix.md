# Plan 05: Ablation matrix and decision gates

TASK
- Define an ablation matrix with decision gates to determine whether cycle training is helping.

WHAT TO DO
1) Baselines:
   - l2t-only
   - t2l-only
   - unconditional (optional)
2) Additive:
   - + cycle_text only
   - + cycle_latent only
   - + both
3) Define decision gates:
   - cycle loss decreases without exploding base loss
   - text accuracy does not collapse
   - latent norms stay bounded

ACCEPTANCE CRITERIA
- `ulw-work/research/vae_cycle_report.md` Section 7 has a table listing these runs and metrics.
