# Prompt 03: Scaffold configs and entrypoint

TASK
- Create a standalone config and a minimal CLI entrypoint (YAML + lightweight loader), avoiding Hydra search path issues.

REQUIREMENTS
- Config file: `cycle_vae_impl/configs/mmdit_cycle_vae.yaml`
- Entrypoint: `cycle_vae_impl/entrypoints/train_cycle_vae.py`

CONFIG RULES
- Copy key defaults from `latentDLM_mmdit/configs/mmdit_stable.yaml` as a starting point.
- Add the new `loss.*` keys for cycle_vae.
- Ensure output directory is under `cycle_vae_impl/outputs/`.

ENTRYPOINT RULES
- Must import (not copy) the existing model/tokenizer/dataloader utilities:
  - tokenizer: `latentDLM_mmdit.modeling_mmdit.get_tokenizer`
  - dataloader: `latentDLM_mmdit.data_simple.get_simple_dataloaders`
  - model: `latentDLM_mmdit.models.multimodal_mmdit.MultimodalMMDiT`
- Must instantiate `CycleVAETrainer` from `cycle_vae_impl/trainers/cycle_vae_trainer.py`.
