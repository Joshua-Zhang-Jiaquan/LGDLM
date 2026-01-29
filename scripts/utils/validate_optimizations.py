#!/usr/bin/env python3
"""
Validate that all optimization fixes have been applied correctly
"""

import sys
from pathlib import Path
import re

class ValidationResult:
    def __init__(self, name, passed, message):
        self.name = name
        self.passed = passed
        self.message = message

def check_nccl_timeout():
    """Check if NCCL timeout is fixed."""
    file_path = Path("train_qwen_english_l2t_stable.sh")
    if not file_path.exists():
        return ValidationResult("NCCL Timeout", False, "File not found")

    content = file_path.read_text()

    # Check for the bad timeout
    if "12000000" in content:
        return ValidationResult("NCCL Timeout", False, "Still set to 138 days (12000000)")

    # Check for the good timeout
    if "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800" in content:
        return ValidationResult("NCCL Timeout", True, "Correctly set to 30 minutes")

    return ValidationResult("NCCL Timeout", False, "Timeout value unclear")

def check_data_loading():
    """Check if data loading is optimized."""
    file_path = Path("latentDLM_mmdit/data_simple.py")
    if not file_path.exists():
        return ValidationResult("Data Loading", False, "File not found")

    content = file_path.read_text()

    issues = []
    if "prefetch_factor" not in content:
        issues.append("Missing prefetch_factor")
    if "persistent_workers=True" not in content:
        issues.append("Missing persistent_workers")

    if issues:
        return ValidationResult("Data Loading", False, ", ".join(issues))

    return ValidationResult("Data Loading", True, "Optimized with prefetch and persistent workers")

def check_gradient_checkpointing():
    """Check if gradient checkpointing is enabled."""
    file_path = Path("latentDLM_mmdit/models/multimodal_mmdit.py")
    if not file_path.exists():
        return ValidationResult("Gradient Checkpointing", False, "File not found")

    content = file_path.read_text()

    if "use_gradient_checkpointing" not in content:
        return ValidationResult("Gradient Checkpointing", False, "Not implemented")

    if "checkpoint(" in content and "use_reentrant=False" in content:
        return ValidationResult("Gradient Checkpointing", True, "Implemented correctly")

    return ValidationResult("Gradient Checkpointing", False, "Partially implemented")

def check_ddp_optimization():
    """Check if DDP is optimized."""
    file_path = Path("latentDLM_mmdit/train_mmdit_stable.py")
    if not file_path.exists():
        return ValidationResult("DDP Optimization", False, "File not found")

    content = file_path.read_text()

    issues = []
    if "gradient_as_bucket_view=True" not in content:
        issues.append("Missing gradient_as_bucket_view")
    if "broadcast_buffers=False" not in content:
        issues.append("Missing broadcast_buffers")

    if issues:
        return ValidationResult("DDP Optimization", False, ", ".join(issues))

    return ValidationResult("DDP Optimization", True, "Optimized with bucket view and buffer settings")

def check_fused_optimizer():
    """Check if fused optimizer is enabled."""
    file_path = Path("latentDLM_mmdit/optimizer.py")
    if not file_path.exists():
        return ValidationResult("Fused Optimizer", False, "File not found")

    content = file_path.read_text()

    if "fused=True" in content:
        return ValidationResult("Fused Optimizer", True, "Enabled")

    return ValidationResult("Fused Optimizer", False, "Not enabled")

def check_checkpoint_cleanup():
    """Check if checkpoint cleanup is implemented."""
    file_path = Path("latentDLM_mmdit/train_mmdit_stable.py")
    if not file_path.exists():
        return ValidationResult("Checkpoint Cleanup", False, "File not found")

    content = file_path.read_text()

    if "cleanup_old_checkpoints" in content:
        return ValidationResult("Checkpoint Cleanup", True, "Implemented")

    return ValidationResult("Checkpoint Cleanup", False, "Not implemented")

def check_loss_function():
    """Check if loss function is improved."""
    file_path = Path("latentDLM_mmdit/improved_trainer_stable.py")
    if not file_path.exists():
        return ValidationResult("Loss Function", False, "File not found")

    content = file_path.read_text()

    if "cosine_similarity" in content and "latent_loss" in content:
        return ValidationResult("Loss Function", True, "Using cosine similarity")

    return ValidationResult("Loss Function", False, "Still using MSE")

def main():
    print("="*70)
    print("MM-LDLM Training Optimization - Validation Report")
    print("="*70)
    print()

    # Run all checks
    checks = [
        check_nccl_timeout(),
        check_data_loading(),
        check_gradient_checkpointing(),
        check_ddp_optimization(),
        check_fused_optimizer(),
        check_checkpoint_cleanup(),
        check_loss_function(),
    ]

    # Print results
    passed = 0
    failed = 0

    for result in checks:
        status = "✓" if result.passed else "✗"
        color = "\033[92m" if result.passed else "\033[91m"
        reset = "\033[0m"

        print(f"{color}{status}{reset} {result.name:25} - {result.message}")

        if result.passed:
            passed += 1
        else:
            failed += 1

    print()
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
    print()

    if failed == 0:
        print("✓ All optimizations applied successfully!")
        print()
        print("Next steps:")
        print("  1. Test training: NNODES=1 NPROC_PER_NODE=2 bash train_qwen_english_l2t_stable.sh")
        print("  2. Monitor: python monitor_training.py train_logs/train_*.log")
        print("  3. Benchmark: bash benchmark_training.sh")
        return 0
    else:
        print("✗ Some optimizations are missing.")
        print()
        print("To apply all fixes:")
        print("  bash apply_critical_fixes.sh")
        print()
        print("Or follow the manual guide:")
        print("  cat MIGRATION_GUIDE.md")
        return 1

if __name__ == '__main__':
    sys.exit(main())
