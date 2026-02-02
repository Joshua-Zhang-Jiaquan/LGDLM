# Prompt 06: DDP and checkpointing (optional)

TASK
- Make `train_cycle_vae.py` runnable under `torchrun` without importing original entrypoints.

REQUIREMENTS
- If `WORLD_SIZE > 1`, initialize torch.distributed.
- Wrap trainer in DDP.
- Save checkpoints only under `cycle_vae_impl/outputs/...`.

OPTIONAL
- Reuse checkpoint helpers from `latentDLM_mmdit/checkpoints_mmdit.py` if convenient.
