# File: latentDLM_mmdit/modeling_mmdit.py
from transformers import AutoTokenizer
import torch.nn as nn
from .models.multimodal_mmdit import MultimodalMMDiT


def get_tokenizer(config):
    tokenizer_name = config.data.tokenizer_name
    
    # Check if tokenizer_name is a local path that exists
    import os
    local_path = None
    if os.path.exists(tokenizer_name):
        local_path = tokenizer_name
    else:
        # Check in local_models directory
        local_models_dir = os.path.join(os.path.dirname(__file__), "..", "local_models")
        potential_local_path = os.path.join(local_models_dir, tokenizer_name)
        if os.path.exists(potential_local_path):
            local_path = potential_local_path
    
    if local_path is not None:
        print(f"Loading tokenizer from local path: {local_path}")
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    else:
        print(f"Loading tokenizer from Hugging Face Hub: {tokenizer_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            # Try with offline mode and local files only as fallback
            print(f"Failed to download tokenizer from HF Hub: {e}")
            print("Trying to load from cache or local files only...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.model_max_length = config.model.max_seq_len
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