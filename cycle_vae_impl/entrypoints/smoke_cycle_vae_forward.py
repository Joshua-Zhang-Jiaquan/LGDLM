import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from cycle_vae_impl.models.mmdit_stub import MMDiTStub
from latentDLM_mmdit.utils import parse_dtype

from cycle_vae_impl.trainers.cycle_vae_trainer import CycleVAETrainer
from cycle_vae_impl.utils.config import load_config
from cycle_vae_impl.utils.dummy_tokenizer import DummyTokenizer


def _build_synthetic_batch(tokenizer, cfg, device, dtype):
    batch_size = int(cfg.training.train_batch_size)
    seq_len = int(cfg.model.max_seq_len)
    latent_dim = int(cfg.model.latent_dim)

    vocab_size = len(tokenizer)
    input_ids = torch.randint(low=0, high=max(1, vocab_size), size=(batch_size, seq_len), device=device)

    # ensure pad id is used a bit if defined
    if tokenizer.pad_token_id is not None:
        input_ids[:, -1] = tokenizer.pad_token_id

    attention_mask = (input_ids != (tokenizer.pad_token_id or 0)).to(dtype=torch.bool)
    latents = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "latent": latents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dtype = parse_dtype(cfg.training.dtype)
    device = torch.device(args.device)

    if cfg.tokenizer.get("path"):
        from latentDLM_mmdit.modeling_mmdit import get_tokenizer
        tokenizer = get_tokenizer(cfg)
    else:
        tokenizer = DummyTokenizer(
            vocab_size=int(cfg.tokenizer.vocab_size),
            mask_token_id=int(cfg.tokenizer.mask_token_id),
            pad_token_id=int(cfg.tokenizer.pad_token_id),
        )
    vocab_size = len(tokenizer)
    latent_dim = int(cfg.model.get("latent_dim", 32))

    try:
        from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
        model = MultimodalMMDiT(
            config=cfg.model,
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            cluster_size=cfg.model.get("cluster_size", 0),
        )
    except ImportError:
        if not bool(cfg.model.get("use_stub_if_no_mmdit", True)):
            raise
        model = MMDiTStub(
            vocab_size=vocab_size,
            latent_dim=latent_dim,
            hidden_size=int(cfg.model.hidden_size),
            max_seq_len=int(cfg.model.max_seq_len),
        )

    model = model.to(device=device, dtype=dtype)

    text_noise_schedule = None
    trainer = CycleVAETrainer(
        model=model,
        tokenizer=tokenizer,
        text_noise_schedule=text_noise_schedule,
        dtype=dtype,
        config=cfg,
    ).to(device=device)

    batch = _build_synthetic_batch(tokenizer, cfg, device, dtype)

    loss, metrics = trainer(batch, step=0)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss}")
    for k in ["base_t2l_latent_loss", "base_l2t_text_loss", "cycle_text_loss"]:
        if k not in metrics:
            raise RuntimeError(f"Missing metric: {k}")

    print("ok")
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
