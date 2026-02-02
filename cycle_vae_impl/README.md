# cycle_vae_impl

Goal: implement a minimal "cycle_vae" training variant in an isolated folder.

Implemented objectives:
- cycle_text (T2L2T)
- cycle_latent (L2T2L; pseudo-label by argmax or categorical sampling)

Hard constraint: this folder must not modify or depend on edits to `latentDLM_mmdit/`. It may import modules from `latentDLM_mmdit/`.

Layout:
- `cycle_vae_impl/prompts/`: implementation prompts (written first; then followed to implement).
- `cycle_vae_impl/configs/mmdit_cycle_vae.yaml`: standalone YAML config (loaded by `cycle_vae_impl/utils/config.py`), not Hydra.
- `cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml`: cluster-oriented config template.
- `cycle_vae_impl/trainers/cycle_vae_trainer.py`: trainer that computes base losses + cycle_text.
- `cycle_vae_impl/entrypoints/train_cycle_vae.py`: runnable training entrypoint (minimal CLI; optional DDP).
- `cycle_vae_impl/entrypoints/smoke_cycle_vae_forward.py`: smoke forward test (no dataset required).

Quick smoke:
```bash
python cycle_vae_impl/entrypoints/smoke_cycle_vae_forward.py --config cycle_vae_impl/configs/mmdit_cycle_vae.yaml
```

Tiny synthetic train (also writes `training_log.jsonl`):
```bash
python cycle_vae_impl/entrypoints/train_cycle_vae.py --config cycle_vae_impl/configs/mmdit_cycle_vae.yaml --synthetic --override training.num_train_steps=10
```
