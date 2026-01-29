from transformers import AutoTokenizer

from wedlm_bridge.pretrained_utils import resolve_pretrained_path


def get_tokenizer(config):
    tokenizer_name = None
    if hasattr(config, "data") and hasattr(config.data, "tokenizer_name"):
        tokenizer_name = config.data.tokenizer_name
    if tokenizer_name is None and hasattr(config, "tokenizer") and hasattr(config.tokenizer, "name"):
        tokenizer_name = config.tokenizer.name
    if tokenizer_name is None and hasattr(config, "model") and hasattr(config.model, "pretrained_model_name_or_path"):
        tokenizer_name = config.model.pretrained_model_name_or_path
    tokenizer_name = tokenizer_name or "tencent/WeDLM-8B-Base"

    cache_dir = getattr(getattr(config, "tokenizer", None), "cache_dir", None)
    local_files_only = bool(getattr(getattr(config, "tokenizer", None), "local_files_only", False))
    if hasattr(config, "model") and getattr(config.model, "local_files_only", None) is not None:
        local_files_only = bool(config.model.local_files_only)

    tokenizer_name, resolved_local_only = resolve_pretrained_path(config, tokenizer_name)
    local_files_only = local_files_only or resolved_local_only

    trust_remote_code = True
    if hasattr(config, "model") and getattr(config.model, "trust_remote_code", None) is not None:
        trust_remote_code = bool(config.model.trust_remote_code)
    if hasattr(config, "tokenizer") and getattr(config.tokenizer, "trust_remote_code", None) is not None:
        trust_remote_code = bool(config.tokenizer.trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    if tokenizer.pad_token_id is None or tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id and mask_token_id.")

    if hasattr(config, "model") and hasattr(config.model, "max_seq_len"):
        tokenizer.model_max_length = int(config.model.max_seq_len)

    return tokenizer


def get_model(config, tokenizer, device=None, dtype=None):
    model_type = config.model.type
    if model_type != "wedlm8b_latent_ar_bridge":
        raise ValueError(f"Unsupported model type: {model_type}. Expected 'wedlm8b_latent_ar_bridge'.")

    from wedlm_bridge.models.wedlm8b_latent_ar_bridge import WeDLM8BLatentARBridge

    model = WeDLM8BLatentARBridge(config.model, tokenizer)
    if device is not None:
        model = model.to(device, dtype=dtype)
    elif dtype is not None:
        model = model.to(dtype=dtype)
    return model

