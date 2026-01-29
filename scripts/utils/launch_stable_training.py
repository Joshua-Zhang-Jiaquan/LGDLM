#!/usr/bin/env python3
"""
Stable Training Launcher - Uses NaN-safe versions of training code

This script is a simple wrapper that launches train_mmdit_stable.py
which uses improved_trainer_stable.py with all NaN gradient fixes applied.

Usage:
    python launch_stable_training.py [hydra args...]

Example:
    python launch_stable_training.py --config-name mmdit_stable \
        training.train_batch_size=4 \
        optimizer.lr=5e-5
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the stable training script
from latentDLM_mmdit import train_mmdit_stable

if __name__ == "__main__":
    # The stable training script will handle all arguments via Hydra
    train_mmdit_stable.main()
