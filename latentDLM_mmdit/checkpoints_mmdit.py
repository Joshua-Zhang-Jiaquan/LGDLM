# File: latentDLM_mmdit/checkpoints.py (FIXED)
import json
import torch
from pathlib import Path
import shutil
from omegaconf import OmegaConf
from dataclasses import dataclass

# FIX: Import from modeling_mmdit
from latentDLM_mmdit.modeling_mmdit import get_model, get_tokenizer

@dataclass
class TrainingState:
    epoch: int = 0
    epoch_start_step: int = 0
    step: int = 0
    total_tokens: int = 0
    total_flops: float = 0.0
    start_time: float = 0.0
    curr_time: float = 0.0


def save_checkpoint(path, model, optimizer, state, config=None):
    """Save checkpoint for MMDiT training."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Handle DDP model
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'state': state,
        'config': OmegaConf.to_container(config) if config else None,
    }
    
    torch.save(checkpoint, path / "checkpoint.pt")
    print(f"Saved checkpoint to {path / 'checkpoint.pt'}")
    
    # Also save config if provided
    if config:
        with open(path / "config.yaml", 'w') as f:
            OmegaConf.save(config, f)


def load_checkpoint_for_training(path, device=None, dtype=None):
    """Load checkpoint for resuming MMDiT training."""
    path = Path(path)
    
    if not (path / "checkpoint.pt").exists():
        raise FileNotFoundError(f"Checkpoint not found at {path / 'checkpoint.pt'}")
    
    checkpoint = torch.load(path / "checkpoint.pt", map_location='cpu')
    
    # Load config
    if 'config' in checkpoint:
        config = OmegaConf.create(checkpoint['config'])
    else:
        # Try to load from config file
        config_file = path / "config.yaml"
        if config_file.exists():
            config = OmegaConf.load(config_file)
        else:
            raise FileNotFoundError(f"Config not found in checkpoint or at {config_file}")
    
    # Get tokenizer and model - FIXED: Use imported functions
    tokenizer = get_tokenizer(config)
    model = get_model(config, tokenizer, device, dtype)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer
    from latentDLM_mmdit.optimizer import get_optimizer
    optimizer = get_optimizer(config, model)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load training state
    state = checkpoint.get('state', TrainingState())
    
    # Text diffusion (masked diffusion)
    from latentDLM_mmdit.diffusion_process import MaskedDiffusion
    text_noise_schedule = MaskedDiffusion(tokenizer)
    
    return model, text_noise_schedule, tokenizer, config, model, optimizer, state


def save_rng_state(path, rank):
    """Save random number generator state."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    rng_state = {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    
    torch.save(rng_state, path / f"rng_state_{rank}.pt")


def load_rng_state(path, rank):
    """Load random number generator state."""
    path = Path(path)
    rng_file = path / f"rng_state_{rank}.pt"
    
    if rng_file.exists():
        rng_state = torch.load(rng_file, map_location='cpu')
        torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and rng_state['cuda'] is not None:
            torch.cuda.set_rng_state(rng_state['cuda'])
        print(f"Loaded RNG state for rank {rank}")