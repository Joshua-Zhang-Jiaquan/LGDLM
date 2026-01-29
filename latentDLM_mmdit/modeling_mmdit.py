# File: latentDLM_mmdit/modeling_mmdit.py
from transformers import AutoTokenizer
import torch.nn as nn
from .models.multimodal_mmdit import MultimodalMMDiT
import os



# File: latentDLM_mmdit/modeling_mmdit.py (UPDATED)
from transformers import AutoTokenizer
import os
from pathlib import Path

from transformers import AutoTokenizer
import os
from pathlib import Path

from transformers import AutoTokenizer
import os
from pathlib import Path

# File: latentDLM_mmdit/modeling_mmdit.py
from transformers import AutoTokenizer
import os
from pathlib import Path

def get_tokenizer(config=None, tokenizer_path=None):
    """Get tokenizer - simplified version for local offline use"""
    # SET THIS FIRST - BEFORE ANY TOKENIZER IMPORTS!
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # If tokenizer_path is provided, use it directly
    if tokenizer_path is None:
        # Try to get path from config if no direct path provided
        if config is not None:
            # Handle config (could be dict or object)
            if isinstance(config, dict):
                tokenizer_config = config.get("tokenizer", {})
                tokenizer_path = tokenizer_config.get("path", None)
            else:
                # Object config
                if hasattr(config, 'tokenizer') and hasattr(config.tokenizer, 'path'):
                    tokenizer_path = config.tokenizer.path
                else:
                    tokenizer_path = None
        
        # If still no path, use default
        if tokenizer_path is None:
            tokenizer_path = "/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/local_data/tokenizers/bert-base-uncased"
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    # Always use local files only (no internet)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True  # Critical for offline
    )

    # Set model max length (default 4096)
    tokenizer.model_max_length = 4096

    print(f"✓ Tokenizer loaded successfully!")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print(f"  Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # Ensure mask token is set (for masked diffusion)
    if tokenizer.mask_token is None:
        print("  Adding [MASK] token...")
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
        print(f"  Mask token added: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

    return tokenizer

   

def get_model(config, tokenizer, device=None, dtype=None):
    vocab_size = len(tokenizer)
    
    if config.model.type == "multimodal_mmdit":
        print(f"Using Multimodal MMDiT for joint text-latent generation")
        model = MultimodalMMDiT(
            config=config.model,
            vocab_size=vocab_size,
            latent_dim=config.model.get("latent_dim", 768),
            cluster_size=config.model.get("cluster_size", 0)
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}. Use 'multimodal_mmdit' for MMDiT training.")

    if device is not None:
        model = model.to(device, dtype=dtype)
    elif dtype is not None:
        model = model.to(dtype=dtype)

    return model