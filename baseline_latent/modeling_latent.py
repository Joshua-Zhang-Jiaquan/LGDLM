# File: baseline_latent/modeling_latent.py (MODIFIED version)
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import baseline_latent.models.dit as dit
from .models.dit_latent import DITWithLatentConditioning  # NEW import

try:
    import flash_attn
    has_flash_attn = True
except ImportError:
    has_flash_attn = False


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    tokenizer.model_max_length = config.model.max_seq_len
    return tokenizer



def get_model(config, tokenizer, device=None, dtype=None):
    vocab_size = len(tokenizer)
    model_config = config.model
    
    # NEW: Unified model selection logic
    # Check what type of conditioning is requested
    conditioning_type = model_config.get("conditioning_type", None)
    
    # Fallback to old style for compatibility
    if conditioning_type is None and model_config.get("use_latent_conditioning", False):
        conditioning_type = "adaln"  # Default to AdaLN
    
    # Select model based on conditioning type
    if conditioning_type == "cross_attention":
        print(f"Using DIT with CROSS-ATTENTION latent conditioning")
        from .models.dit_latent import DITWithCrossAttention
        model = DITWithCrossAttention(
            model_config,
            vocab_size=vocab_size,
            latent_dim=model_config.get("latent_dim", 768),
            cluster_size=model_config.get("cluster_size", 0)
        )
    
    elif conditioning_type == "adaln":
        print(f"Using DIT with AdaLN latent conditioning (dim={model_config.get('latent_dim', 768)})")
        from .models.dit_latent import DITWithLatentConditioning
        model = DITWithLatentConditioning(
            model_config,
            vocab_size=vocab_size,
            latent_dim=model_config.get("latent_dim", 768),
            cluster_size=model_config.get("cluster_size", 0)
        )
    
    elif model_config.type == "diffusion":
        print("Using vanilla DIT (no latent conditioning)")
        from .models import dit
        model = dit.DIT(model_config, vocab_size, model_config.get("cluster_size", 0))
    
    elif model_config.type == "autoregressive":
        # Handle autoregressive model
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(
            vocab_size=vocab_size,
            num_hidden_layers=model_config.n_blocks,
            hidden_size=model_config.hidden_size,
            intermediate_size=4*model_config.hidden_size,
            num_attention_heads=model_config.n_heads,
            max_position_embeddings=model_config.max_seq_len,
            attn_implementation="flash_attention_2" if has_flash_attn else "sdpa",
            torch_dtype=dtype,
        )
        model = LlamaForCausalLM(cfg)
    
    else:
        raise ValueError(f"Unknown model type: {model_config.type}")

    if device is not None:
        model = model.to(device, dtype=dtype)
    elif dtype is not None:
        model = model.to(dtype=dtype)

    return model