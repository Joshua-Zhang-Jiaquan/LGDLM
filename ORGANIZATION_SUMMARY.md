# Repository Organization Summary

This document describes the reorganization completed on 2026-01-29.

## Directory Structure

```
MM-LDLM/
├── README.md                    # Main documentation
├── CLAUDE.md                    # Claude Code instructions
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
│
├── docs/                        # All documentation (16 files)
│   ├── STABLE_TRAINING_GUIDE.md
│   ├── START_HERE.md
│   ├── QUICK_REFERENCE.md
│   ├── TROUBLESHOOTING_GUIDE.md
│   ├── NAN_FIX_SUMMARY.md
│   ├── NAN_GRADIENT_FIXES.md
│   ├── DISTRIBUTED_DEBUG_GUIDE.md
│   ├── MIGRATION_GUIDE.md
│   ├── OPTIMIZATION_CHEATSHEET.md
│   └── ... (other guides)
│
├── scripts/
│   ├── training/                # Training scripts (16 files)
│   │   ├── train_qwen_english_l2t_stable.sh
│   │   ├── train_qwen_english_l2t.sh
│   │   ├── train_sonar_english_l2t.sh
│   │   └── ... (other training scripts)
│   │
│   └── utils/                   # Utility scripts (11 files)
│       ├── test_single_node.sh
│       ├── test_distributed_setup.sh
│       ├── apply_nan_fixes.sh
│       ├── apply_critical_fixes.sh
│       ├── benchmark_training.sh
│       ├── monitor_training.py
│       ├── validate_optimizations.py
│       └── ... (other utilities)
│
├── results/                     # Experiment outputs
│   ├── archive/                 # Archived configs and patches
│   │   ├── TRAIN_MMDIT_PATCH.py
│   │   ├── IMPROVED_TRAINER_PATCH.py
│   │   ├── mmdit_stable_config.yaml
│   │   └── prompts.txt
│   ├── fixed_l2t_results/
│   ├── t2l_results/
│   ├── unconditional_results/
│   └── outputs/
│
├── latentDLM_mmdit/            # Main MMDiT implementation
├── latentIMG_mmdit/            # Image MMDiT
├── baseline/                    # Baseline models
├── baseline_latent/            # Latent baselines
├── preprocessed_data/          # Data preprocessing
├── saved/                       # Training checkpoints
├── train_logs/                 # Training logs
└── wedlm_bridge/               # Standalone (preserved)
```

## Changes Made

### Files Moved

1. **Documentation** (16 files → `docs/`)
   - All .md files except README.md and CLAUDE.md
   - Includes training guides, troubleshooting, and references

2. **Training Scripts** (16 files → `scripts/training/`)
   - train_qwen_*.sh
   - train_sonar_*.sh

3. **Utility Scripts** (11 files → `scripts/utils/`)
   - test_*.sh
   - apply_*.sh
   - benchmark_training.sh
   - monitor_training.py
   - validate_optimizations.py

4. **Results** (7 directories → `results/`)
   - fixed_l2t_results/
   - t2l_results/
   - unconditional_results/
   - outputs/
   - output_dir/

5. **Archive** (4 files → `results/archive/`)
   - TRAIN_MMDIT_PATCH.py
   - IMPROVED_TRAINER_PATCH.py
   - mmdit_stable_config.yaml
   - prompts.txt

### Files Removed

- .DS_Store (macOS metadata)
- =0.4.0 (corrupted file)
- .~preprocess_data_qwen.sh (temp file)
- nohup.out (old log)
- docker_test.sh (empty file)

### Documentation Updated

All documentation files have been updated with correct paths:
- README.md - Updated directory structure and quick start
- CLAUDE.md - Updated all script paths and references
- docs/*.md - Updated all internal references to scripts and files

## Usage

### Running Training Scripts

All training scripts should be run from the repository root:

```bash
# From repository root
bash scripts/training/train_qwen_english_l2t_stable.sh

# Or with environment variables
NNODES=2 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Running Utility Scripts

```bash
# Test single node
bash scripts/utils/test_single_node.sh

# Apply fixes
bash scripts/utils/apply_nan_fixes.sh

# Monitor training
python scripts/utils/monitor_training.py
```

### Accessing Documentation

```bash
# View guides
cat docs/STABLE_TRAINING_GUIDE.md
cat docs/QUICK_REFERENCE.md
cat docs/TROUBLESHOOTING_GUIDE.md
```

## Notes

- All scripts change to the repository root directory before executing, so relative paths to `latentDLM_mmdit/`, `preprocessed_data/`, etc. still work correctly
- The `wedlm_bridge/` directory was preserved as a standalone folder as requested
- Original training code in `latentDLM_mmdit/` remains untouched
- All checkpoints in `saved/` and logs in `train_logs/` remain in place

## Benefits

1. **Cleaner root directory** - Only essential files visible
2. **Logical grouping** - Related files organized together
3. **Easier navigation** - Clear separation of docs, scripts, and results
4. **Better maintainability** - Easier to find and update files
5. **Preserved functionality** - All scripts still work as before
