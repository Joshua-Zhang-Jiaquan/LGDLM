# Quick Start Guide

This repository has been reorganized for better maintainability. Everything is ready to use!

## 🚀 Start Training Immediately

```bash
# Recommended: Test on single node first
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# Scale to multiple nodes
NNODES=2 bash scripts/training/train_qwen_english_l2t_stable.sh
```

## 📁 New Directory Structure

```
MM-LDLM/
├── docs/                    # 16 documentation files
├── scripts/
│   ├── training/           # 16 training scripts
│   └── utils/              # 11 utility scripts
├── results/                # Experiment outputs
│   └── archive/           # Patches and configs
├── latentDLM_mmdit/       # Main code (unchanged)
└── [other core directories]
```

## 📚 Key Documentation

- **docs/QUICK_REFERENCE.md** - One-page cheat sheet
- **docs/STABLE_TRAINING_GUIDE.md** - Complete training guide
- **docs/TROUBLESHOOTING_GUIDE.md** - Common issues
- **ORGANIZATION_SUMMARY.md** - Details of reorganization

## 🔧 Common Commands

```bash
# Test single node
bash scripts/utils/test_single_node.sh

# Monitor training
tail -f train_logs/train_*_node0.log

# Apply fixes
bash scripts/utils/apply_nan_fixes.sh
```

## ✅ Verification Status

All smoke tests passed (18/18):
- ✓ Directory structure correct
- ✓ All files in place
- ✓ Documentation updated
- ✓ Scripts validated
- ✓ Dependencies verified

Ready to train!
