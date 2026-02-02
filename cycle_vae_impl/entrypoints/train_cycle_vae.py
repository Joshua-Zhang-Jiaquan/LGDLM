import argparse
import json
import os
import time
from pathlib import Path

import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.distributed as dist

from latentDLM_mmdit.data_simple import get_simple_dataloaders
from cycle_vae_impl.models.mmdit_stub import MMDiTStub
from latentDLM_mmdit.utils import parse_dtype

from cycle_vae_impl.trainers.cycle_vae_trainer import CycleVAETrainer
from cycle_vae_impl.utils.config import load_config, apply_overrides
from cycle_vae_impl.utils.config import to_plain_dict
from cycle_vae_impl.utils.dummy_tokenizer import DummyTokenizer
from cycle_vae_impl.utils.lr import get_lr
from cycle_vae_impl.utils.optimizer import build_optimizer
from cycle_vae_impl.utils.checkpoint import save_checkpoint, load_checkpoint


def _init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return True, rank, local_rank, world_size


def _apply_overrides(cfg, overrides):
    return apply_overrides(cfg, overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--synthetic", action="store_true", help="use synthetic batches (no dataset required)")
    ap.add_argument("--resume", default="", help="path to checkpoint .pt to resume")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args.override)

    use_ddp, rank, local_rank, world_size = _init_distributed()
    is_main = (not use_ddp) or rank == 0

    torch.manual_seed(int(cfg.training.seed))

    dtype = parse_dtype(cfg.training.dtype)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

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

    if use_ddp:
        trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = build_optimizer(cfg, trainer)

    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(args.resume, trainer, optimizer=optimizer)

    def _synthetic_batches():
        batch_size = int(cfg.training.train_batch_size)
        seq_len = int(cfg.model.max_seq_len)
        latent_dim = int(cfg.model.latent_dim)
        while True:
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
            latents = torch.randn(batch_size, latent_dim, device=device, dtype=dtype)
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "latent": latents}

    if args.synthetic or not cfg.data.get("token_dir"):
        train_iter = _synthetic_batches()
        train_dl = None
    else:
        train_dl, _ = get_simple_dataloaders(cfg, tokenizer=None)
        train_iter = iter(train_dl)

    out_root = Path(cfg.logging.log_dir) / cfg.logging.run_name
    log_file = out_root / "training_log.jsonl"
    if is_main:
        out_root.mkdir(parents=True, exist_ok=True)

    ckpt_dir = out_root / "checkpoints"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_steps = int(cfg.training.num_train_steps)
    log_every = int(cfg.logging.log_every)
    save_every = int(cfg.logging.get("save_every", 0))
    grad_accum = int(cfg.training.gradient_accumulation_steps)
    max_lr = float(cfg.optimizer.lr)

    step = int(start_step)
    start = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            if train_dl is None:
                raise
            train_iter = iter(train_dl)
            batch = next(train_iter)

        if train_dl is not None:
            batch = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in batch.items()}

        lr = get_lr(cfg, max_lr, step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss, metrics = trainer(batch, step=step)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainer.parameters(), float(cfg.optimizer.grad_clip_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if is_main and (step % log_every == 0):
            entry = {"step": step, "lr": lr, "time": time.time() - start}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    entry[k] = float(v)
            with open(str(log_file), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            print(entry)

        if is_main and save_every and (step % save_every == 0) and step > 0:
            ckpt_path = ckpt_dir / f"step_{step}.pt"
            cfg_dict = to_plain_dict(cfg)
            save_checkpoint(str(ckpt_path), trainer, optimizer, step=step, cfg=cfg_dict)
            save_checkpoint(str(ckpt_dir / "latest.pt"), trainer, optimizer, step=step, cfg=cfg_dict)

        step += 1

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
