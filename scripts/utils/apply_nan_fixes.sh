#!/bin/bash
# File: apply_nan_fixes.sh
# Script to help apply NaN gradient fixes to your training code

set -e

echo "=========================================="
echo "NaN Gradient Fix Application Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "latentDLM_mmdit/train_mmdit.py" ]; then
    echo "ERROR: Must run this script from MM-LDLM root directory"
    exit 1
fi

echo "This script will help you apply fixes to prevent NaN gradients."
echo ""
echo "Files that will be modified:"
echo "  1. latentDLM_mmdit/train_mmdit.py"
echo "  2. latentDLM_mmdit/improved_trainer.py"
echo "  3. latentDLM_mmdit/configs/mmdit_preprocessed.yaml"
echo ""
echo "Backups will be created with .backup extension"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create backups
echo ""
echo "Creating backups..."
cp latentDLM_mmdit/train_mmdit.py latentDLM_mmdit/train_mmdit.py.backup
cp latentDLM_mmdit/improved_trainer.py latentDLM_mmdit/improved_trainer.py.backup
if [ -f "latentDLM_mmdit/configs/mmdit_preprocessed.yaml" ]; then
    cp latentDLM_mmdit/configs/mmdit_preprocessed.yaml latentDLM_mmdit/configs/mmdit_preprocessed.yaml.backup
fi
echo "✓ Backups created"

# Copy stable config
echo ""
echo "Installing stable configuration..."
if [ -f "mmdit_stable_config.yaml" ]; then
    cp mmdit_stable_config.yaml latentDLM_mmdit/configs/mmdit_stable.yaml
    echo "✓ Stable config installed at latentDLM_mmdit/configs/mmdit_stable.yaml"
else
    echo "⚠ mmdit_stable_config.yaml not found, skipping"
fi

echo ""
echo "=========================================="
echo "Manual Steps Required"
echo "=========================================="
echo ""
echo "The following changes need to be applied manually:"
echo ""
echo "1. Edit latentDLM_mmdit/train_mmdit.py"
echo "   - See TRAIN_MMDIT_PATCH.py for detailed changes"
echo "   - Replace lines 420-437 with improved loss validation and gradient handling"
echo ""
echo "2. Edit latentDLM_mmdit/improved_trainer.py"
echo "   - See IMPROVED_TRAINER_PATCH.py for detailed changes"
echo "   - Replace lines 336-374 with numerically stable loss computation"
echo ""
echo "3. Update your training script to use stable config:"
echo "   --config-name mmdit_stable"
echo ""
echo "4. Recommended hyperparameters:"
echo "   - Learning rate: 5e-5 (reduced from 1e-4)"
echo "   - Gradient clip: 0.5 (reduced from 1.0)"
echo "   - Warmup steps: 2000 (increased)"
echo "   - Latent loss weight: 0.1 (reduced from 1.0)"
echo ""
echo "=========================================="
echo "Testing"
echo "=========================================="
echo ""
echo "After applying fixes, test with:"
echo ""
echo "  # Single node test (2 GPUs)"
echo "  NNODES=1 NPROC_PER_NODE=2 bash train_qwen_english_l2t_scaled_improved.sh"
echo ""
echo "  # Monitor for NaN"
echo "  tail -f train_logs/latest.log | grep -E 'NaN|ERROR|loss'"
echo ""
echo "=========================================="
echo "Documentation"
echo "=========================================="
echo ""
echo "For detailed information, see:"
echo "  - NAN_GRADIENT_FIXES.md - Complete analysis and solutions"
echo "  - TRAIN_MMDIT_PATCH.py - Training script patches"
echo "  - IMPROVED_TRAINER_PATCH.py - Trainer patches"
echo ""
echo "Done! Please apply the manual changes as described above."
