from pathlib import Path

import torch
from omegaconf import OmegaConf

from wedlm_bridge.modeling_wedlm_bridge import get_model, get_tokenizer


def main():
    config = OmegaConf.load("wedlm_bridge/configs/wedlm8b_ar_bridge.yaml")
    local_dir = Path(config.model.pretrained_local_dir)
    if not (local_dir / "config.json").exists():
        print(f"SKIP: {local_dir} does not look like a local WeDLM checkout (missing config.json).")
        print("Run training once (or pre-download) to populate `model.pretrained_local_dir`.")
        return

    tokenizer = get_tokenizer(config)
    model = get_model(config, tokenizer, device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    batch_size = 1
    seq_len = 8
    latent_dim = int(config.model.latent_dim)

    input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    latents = torch.randn((batch_size, latent_dim))

    with torch.no_grad():
        text_logits, latent_pred = model(
            text_tokens=input_ids,
            attention_mask=attention_mask,
            latents=latents,
            mode="l2t",
        )
    assert text_logits is not None and latent_pred is None
    assert text_logits.shape == (batch_size, seq_len, len(tokenizer)), text_logits.shape

    with torch.no_grad():
        text_logits, latent_pred = model(
            text_tokens=input_ids,
            attention_mask=attention_mask,
            latents=None,
            mode="t2l",
        )
    assert text_logits is None and latent_pred is not None
    assert latent_pred.shape == (batch_size, latent_dim), latent_pred.shape

    print("Smoke test passed.")


if __name__ == "__main__":
    main()

