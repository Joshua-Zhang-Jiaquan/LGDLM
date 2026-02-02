# Prompt 08: Cluster config and runbook

TASK
- Create a cluster-ready config and a runbook for running on a training machine.

FILES TO CREATE
- `cycle_vae_impl/configs/mmdit_cycle_vae_cluster.yaml`
- `cycle_vae_impl/RUNBOOK_CLUSTER.md`

MUST DO
- Ensure cluster config disables the stub:
  - `model.use_stub_if_no_mmdit: false`
- Ensure cluster config uses real tokenizer + dataset paths (fill with placeholders and require overrides).
- Ensure output paths are under `cycle_vae_impl/outputs/`.

RUNBOOK MUST INCLUDE
- Preflight checks (python imports, GPU visibility, mmdit import)
- torchrun examples (single node; multi node with rdzv)
- Resume example
- Override examples for cycle weights and t ranges
