# VAE-Structured Cycle Optimization for L2T/T2L (MM-LDLM)

Date: 2026-01-29

## 0) Executive intent

Goal: Use existing bidirectional modes in `latentDLM_mmdit/` as an encoder/decoder pair and design a VAE-structured optimization with cycle consistency to improve both directions.

Repo reality:
- "L2T" (latent-to-text) is implemented as a text denoising objective conditioned on clean latents.
- "T2L" (text-to-latent) is implemented as a latent denoising objective conditioned on clean text.
- Both are diffusion-style denoisers, not explicit likelihood models with tractable log p.

Therefore, the practical design is:
- Keep the existing diffusion denoising losses as the base objective.
- Add ELBO-inspired regularization terms and cycle-consistency auxiliary losses that are consistent with the repo's training graph.
- Use a gradient policy (stop-grad / alternating updates) to avoid unstable feedback loops.

Deliverable produced by this report:
- A precise objective spec with two interpretations of the user's naming (see Section 3).
- A mapping to this codebase (exact files and hook points).
- A minimal ablation matrix and smoke verification plan.

## 1) Ground truth: where L2T/T2L are in this repo

### 1.1 Training entrypoints
- `latentDLM_mmdit/train_mmdit.py`: Hydra entrypoint; builds model, dataloaders, trainer; logs metrics; runs train loop.
- `latentDLM_mmdit/train_mmdit_stable.py`: stable variant with skip-batch logic for NaN/INF and safer gradient clipping.

### 1.2 Trainer and mode semantics
Core logic lives in:
- `latentDLM_mmdit/improved_trainer.py`
- `latentDLM_mmdit/improved_trainer_stable.py`

Key facts (from `MultimodalDiffusionTrainer_new.forward`):
- Modes are selected by `config.loss.loss_type` (trainer) or by `training.loss_type` (config), depending on code path; in practice, the trainer reads `config.loss.loss_type` and supports fixed modes plus `random` and `sequential` schedules.

Mode behavior:
- `unconditional`:
  - Text is partially noised via masked diffusion, and latent is noised via continuous diffusion.
  - Computes both text loss (CE over masked positions) and latent loss (MSE for latent denoising target).
- `l2t`:
  - Text is fully masked (all mask tokens) and must be reconstructed.
  - Latents are clean and used as conditioning; latent loss is not computed.
- `t2l`:
  - Text is clean and used as conditioning; text loss is not computed.
  - Latents are fully noised and must be reconstructed; latent loss is computed.

Implication:
- In VAE notation, a useful mapping is:
  - Decoder: p_theta(y | z) ~ "l2t" (text reconstruction conditioned on latent)
  - Encoder/inference: q_phi(z | y) ~ "t2l" (latent reconstruction conditioned on text)
  - Prior model p(z) is not explicitly modeled unless we leverage unconditional latent diffusion or add a separate prior.

### 1.3 Data contract
Primary dataloader used by the MMDiT training flow:
- `latentDLM_mmdit/data_simple.py`: provides batches with keys:
  - `input_ids` (token ids)
  - `attention_mask` (optional)
  - `latent` (latent vectors)

Preprocessing contract (continuous latents):
- `preprocessed_data/README`: latents are stored as `.npy` vectors; JSON maps text to latent paths.

### 1.4 Sampling scripts (for offline evaluation)
- `latentDLM_mmdit/sample_l2t_fixed.py`: generate text from latent vectors (offline, not used in training loop)
- `latentDLM_mmdit/sample_t2l_fixed.py`: generate latent vectors from text prompts (offline)
- `latentDLM_mmdit/sample_unconditional_fixed.py`: unconditional latent generation then text generation

Important: these sampling scripts implement their own sampling loops and are not currently a differentiable training path. For cycle training, we should reuse the trainer/model forward passes instead of using these sampling scripts.

## 2) What "VAE-structured" means here (pragmatic definition)

### 2.1 Classical ELBO (reference framing)
Standard latent-variable model:
- Prior p(z)
- Decoder p(y | z)
- Encoder q(z | y)

ELBO:
  L_ELBO(y) = E_{q(z|y)}[ log p(y|z) ] - KL(q(z|y) || p(z))

### 2.2 Diffusion models are compatible with variational views (but log-likelihood is not cheap)
For diffusion/score models, several works provide variational interpretations or likelihood bounds:
- Huang et al., "A Variational Perspective on Diffusion-Based Generative Models and Score Matching" (arXiv:2106.02808)
  - https://arxiv.org/abs/2106.02808
- Kingma et al., "Variational Diffusion Models" (arXiv:2107.00630)
  - https://arxiv.org/abs/2107.00630
- Song et al., "Maximum Likelihood Training of Score-Based Diffusion Models" (NeurIPS 2021; arXiv preprint)
  - https://arxiv.org/abs/2101.09258
- Piriyakulkij et al., "Denoising Diffusion Variational Inference" (arXiv:2401.02739)
  - https://arxiv.org/abs/2401.02739

Practical implication for this repo:
- We do not need to compute exact log p(y|z) or KL(q||p) to get useful training signal.
- We can build an ELBO-inspired objective by:
  - Using existing denoising losses as reconstruction surrogates.
  - Adding auxiliary regularizers that approximate a prior match on z (moment matching / MMD / norm constraints) OR by using unconditional latent diffusion as a learned prior.

### 2.3 Internal repo theory already proposes ELBO + cycle
This repository already contains a relevant internal document:
- `Latent_Unified_DLM/main/method_outline.md`
  - Stage II discusses ELBO (Eq. 12) and cycle consistency (Eq. 29).

We will align our practical proposal with that internal framing, but constrained to current code capabilities.

## 3) Define the cycles (naming locked)

User requested: "L2T2T and T2L2T".

Naming decision (locked): treat "L2T2T" as a typo for "L2T2L".

Cycles used in this design:
- Text cycle: T2L2T (text -> latent -> text)
- Latent cycle: L2T2L (latent -> text -> latent)

Cycle-Text (T2L2T):
- Input: y (clean tokens)
- Step 1: z_hat = T2L(y)
- Step 2: y_recon = L2T(z_hat)
- Loss: cycle_text_loss = CE(y_recon_logits, y)

Cycle-Latent (L2T2L):
- Input: z (clean latent vector)
- Step 1: y_hat = L2T(z)
- Step 2: z_recon = T2L(y_hat)
- Loss: cycle_latent_loss = MSE(z_recon, z)

### 3.3 What we can implement without adding new sampling machinery

Key constraint: Text sampling is discrete and non-differentiable.

Therefore we separate cycle terms into:

(A) Differentiable cycle via latent bridge (recommended start)
- T2L2T can be implemented in a differentiable way if we avoid discrete sampling:
  - Make the latent-side diffusion state explicit (timestep + noise). In practice, the current `t2l` mode uses "full noise" (`latent_t = 1`) and samples Gaussian noise inside the forward path.
  - Use the T2L forward pass to produce a continuous latent prediction z_hat (a tensor).
  - Feed z_hat into L2T forward pass to compute logits over tokens.
  - Compute CE against the original y.
- Gradient policy:
  - Start with stop-grad on z_hat so only L2T learns from the cycle.
  - Later optionally allow gradient into T2L (careful: feedback loops).

Important precision note:
- "No sampling" can be true for discrete tokens (we avoid sampling y_hat), but latent noise epsilon ~ N(0, I) is still sampled in T2L unless you fix it deterministically. This sampling is reparameterizable and remains fully differentiable.

(B) Non-differentiable cycle via sampled text bridge (dual learning / back-translation style)
- L2T2L requires turning L2T outputs into tokens. Options:
  - Stop-grad: sample/argmax y_hat, treat it as data, then train T2L to reconstruct z.
  - Soft/STE: Gumbel-softmax or straight-through argmax to pass approximate gradients.
  - RL: REINFORCE-type gradients (not recommended as first implementation).
- Recommended start: stop-grad with alternating updates.

## 4) Proposed objective (VAE-structured, but implementable)

We define base losses already in the code:
- L_l2t_base: text reconstruction CE in `l2t` mode.
- L_t2l_base: latent denoising MSE in `t2l` mode.

We add cycle auxiliary losses:
- L_cycle_text (T2L2T) as CE against original tokens.
- L_cycle_latent (L2T2L) as MSE against original latent vectors.

We add ELBO-inspired regularizers:
- R_latent_prior: encourage a stable latent space and prevent drift.
  - Option 1 (cheap): match first/second moments of predicted latents to dataset latents or to N(0, I) if latents are normalized.
  - Option 2 (stronger): treat `unconditional` latent diffusion as a learned prior p(z) and add a "prior matching" term via denoising score/loss.
  - Option 3 (future): diffusion-VI style regularized ELBO (Piriyakulkij et al., arXiv:2401.02739).

Total objective (for a minibatch of paired (y, z)):

  L_total = w_l2t * L_l2t_base
          + w_t2l * L_t2l_base
          + w_cyc_text * L_cycle_text
          + w_cyc_latent * L_cycle_latent
          + w_prior * R_latent_prior

Important: this is an objective decomposition, not a single monolithic "ELBO".

## 5) Training algorithm (stable and incremental)

### 5.1 Warm start
Start from existing training behavior:
- Train L2T only or T2L only with the existing modes.
- Ensure stable training (recommend: stable trainer + stable config).

### 5.2 Add cycle-text first (safe, mostly differentiable)
Phase P1:
- Enable computation of L_cycle_text using the T2L forward latent prediction z_hat and L2T logits.
- Default gradient policy:
  - stop-grad on z_hat to avoid destabilizing T2L early.
- Update:
  - L2T gets additional supervision to reconstruct y from latents that are "on-manifold" of T2L predictions.

Shared-weights caveat (important):
- In this repo, L2T/T2L are modes of a single model + trainer with shared parameters. Detaching z_hat prevents gradients flowing through the T2L-produced tensor, but the L2T forward pass still updates shared backbone parameters that also affect T2L behavior.
- If you truly require "only L2T learns from cycle_text", you must use parameter freezing / selective `requires_grad` during the cycle loss computation.

### 5.3 Add cycle-latent via stop-grad pseudo text (dual learning)
Phase P2:
- Generate y_hat (tokens) from z using L2T.
- Treat y_hat as a pseudo-label (no gradient).
- Train T2L to reconstruct z from y_hat with an MSE loss.

Implementation detail:
- This phase can be implemented as alternating steps or as a stochastic mixture per batch:
  - With probability p_dual, do pseudo-label cycle step; else do supervised step.

### 5.4 Optional: allow gradients through both models (advanced)
Only after stability:
- Consider allowing gradient into T2L via the differentiable cycle-text.
- Consider soft-token techniques for the latent cycle.

## 6) Where future implementation would hook into this repo

Primary integration points:
- Add a new loss_type, e.g. `cycle_vae`, in:
  - `latentDLM_mmdit/improved_trainer.py`
  - `latentDLM_mmdit/improved_trainer_stable.py`
- The correct place is inside `MultimodalDiffusionTrainer_new.forward` after obtaining base `text_logits` and `latent_pred`.

Logging:
- `latentDLM_mmdit/train_mmdit.py` and `latentDLM_mmdit/train_mmdit_stable.py` already:
  - write `metrics` to JSONL
  - log numeric metrics to W&B
- Therefore, adding cycle losses only requires adding numeric keys to the `metrics` dict.

Config:
- Extend the `loss:` section in:
  - `latentDLM_mmdit/configs/mmdit_stable.yaml`
  - optionally `latentDLM_mmdit/configs/mmdit.yaml`
with new weights and schedules:
  - cycle_text_weight, cycle_latent_weight
  - cycle_warmup_steps, cycle_ramp_steps
  - stopgrad flags

## 7) Minimal ablation matrix (required to validate gains)

Baseline runs (paired data):
1) L2T only: fixed `loss_type=l2t`
2) T2L only: fixed `loss_type=t2l`
3) Unconditional: `loss_type=unconditional` (if used)
4) Existing random/sequential schedule (if used in your experiments)

Incremental cycle runs:
5) + cycle_text only (T2L2T), stop-grad on z_hat
6) + cycle_latent only (L2T2L), stop-grad on y_hat
7) + both cycles

What to measure (log as metrics):
- Base: text_loss, latent_loss, text_accuracy, grad_norm
- Cycle: cycle_text_loss, cycle_latent_loss, cycle_weight
- Distribution: latent_norm stats; optionally moment match stats

## 8) Test plan (smoke and correctness)

### Objective
Verify that:
- New objective terms compute without shape errors.
- Loss remains finite (NaN/INF guards catch issues).
- Gradients only flow where expected (stop-grad policy).

### Prerequisites
- A minimal dataset with small batch size.
- Stable config baseline available.

### Test cases (future implementation verification)
1) Forward-only unit smoke:
  - Run one forward pass for each mode and ensure losses are finite.
2) Cycle-text on:
  - Ensure cycle_text_loss decreases slightly over a short run.
3) Cycle-latent on:
  - Ensure z_recon has correct shape and loss is finite.
4) Stop-grad audit:
  - Ensure T2L params do not receive gradients from cycle_text when stop-grad enabled.

## 9) Known pitfalls and mitigations

Pitfall: feedback loop instability (model teaches itself garbage).
- Mitigation: stop-grad defaults + warmup + low cycle weights + alternating updates.

Pitfall: posterior collapse / ignoring latents.
- Mitigation: maintain L2T conditioning strength; consider MI-like auxiliary terms later.

Pitfall: discrete sampling breaks gradients.
- Mitigation: start with stop-grad pseudo-labeling for L2T2L.

Pitfall: compute cost explosion if cycle requires full diffusion sampling.
- Mitigation: define cycle losses using the existing forward denoising predictions rather than sampling loops.

Pitfall: the cycle is ill-defined unless you specify the latent diffusion state.
- Mitigation: for the first implementation, match the existing `t2l` mode (full-noise latent_t=1). For stronger regularization, generalize to random latent timesteps (t sampled uniformly) and define z_hat as an x0 prediction at that timestep.

Pitfall: latent scale mismatch / drift.
- Mitigation: explicitly define the latent normalization contract (are dataset latents normalized?). Add a lightweight prior/stat regularizer (mean/var match or norm penalty) early.

## 10) References (externally verifiable)

- Chin-Wei Huang, Jae Hyun Lim, Aaron Courville. "A Variational Perspective on Diffusion-Based Generative Models and Score Matching." arXiv:2106.02808.
  https://arxiv.org/abs/2106.02808

- Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho. "Variational Diffusion Models." arXiv:2107.00630.
  https://arxiv.org/abs/2107.00630

- Yang Song, Conor Durkan, Iain Murray, Stefano Ermon. "Maximum Likelihood Training of Score-Based Diffusion Models." arXiv:2101.09258.
  https://arxiv.org/abs/2101.09258

- Wasu Top Piriyakulkij, Yingheng Wang, Volodymyr Kuleshov. "Denoising Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors." arXiv:2401.02739.
  https://arxiv.org/abs/2401.02739

Internal reference (already in repo):
- `Latent_Unified_DLM/main/method_outline.md` (ELBO Eq. 12; cycle Eq. 29)
