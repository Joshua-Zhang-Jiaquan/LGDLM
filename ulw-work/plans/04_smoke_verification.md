# Plan 04: Smoke verification plan (must be executable)

TASK
- Define smoke tests and verification commands for the eventual implementation.

WHAT TO DO
1) Define a minimal run that executes 50-200 steps with small batch size.
2) Define pass/fail criteria:
   - no NaN/INF loss
   - cycle losses computed and logged
   - stop-grad policy holds (gradients only where expected)
3) Define offline evaluation:
   - run `latentDLM_mmdit/sample_l2t_fixed.py` and `latentDLM_mmdit/sample_t2l_fixed.py` on a small set
   - compare reconstruction quality before/after (qualitative + numeric summary if available)

MUST NOT DO
- Do not require multi-node training as the first validation.

ACCEPTANCE CRITERIA
- A future implementer can copy/paste the commands and observe the expected logs/artifacts.
