import datetime
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(".")
sys.path.append("..")

from latentDLM_mmdit.data_simple import get_simple_dataloaders
from latentDLM_mmdit.utils import get_lr, parse_dtype
from wedlm_bridge.modeling_wedlm_bridge import get_model, get_tokenizer


class Logger:
    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process

    def init(self, *args, **kwargs):
        if self.is_main_process:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_main_process:
            wandb.log(*args, **kwargs)


def _init_distributed() -> tuple[int, int, int, bool, bool]:
    env_has_ddp = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not env_has_ddp:
        return 0, 0, 1, True, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch detected, but CUDA is not available.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    init_kwargs = dict(
        backend="nccl",
        timeout=datetime.timedelta(minutes=30),
        init_method="env://",
    )
    try:
        dist.init_process_group(**init_kwargs, device_id=torch.device("cuda", local_rank))  # type: ignore[arg-type]
    except TypeError:
        dist.init_process_group(**init_kwargs)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = (global_rank == 0)
    is_distributed = dist.is_available() and dist.is_initialized()
    return local_rank, global_rank, world_size, is_main_process, is_distributed


def safe_barrier(local_rank: int | None = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if local_rank is not None:
            dist.barrier(device_ids=[local_rank])  # type: ignore[arg-type]
        else:
            dist.barrier()
    except TypeError:
        dist.barrier()


@contextmanager
def main_process_first(local_rank: int | None = None):
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            safe_barrier(local_rank)
        else:
            safe_barrier(local_rank)
            yield
    else:
        yield


def _freeze_module(module: nn.Module | None) -> bool:
    if module is None:
        return False
    for param in module.parameters():
        param.requires_grad = False
    return True


def _get_text_head(model: nn.Module) -> nn.Module | None:
    if hasattr(model, "text_head") and model.text_head is not None:
        return model.text_head
    if hasattr(model, "lm_head"):
        return model.lm_head  # type: ignore[attr-defined]
    lm = getattr(model, "lm", None)
    if lm is not None:
        if hasattr(lm, "get_output_embeddings"):
            return lm.get_output_embeddings()
        if hasattr(lm, "lm_head"):
            return lm.lm_head  # type: ignore[attr-defined]
    return None


def _maybe_freeze_unused_heads(trainer: nn.Module, config) -> bool:
    freeze_unused = bool(getattr(config.training, "freeze_unused_heads", True))
    if not freeze_unused:
        return False

    model = getattr(trainer, "model", None)
    if model is None:
        return False

    task = str(getattr(config.training, "task", "joint")).lower()
    frozen = False
    if task == "l2t":
        frozen |= _freeze_module(getattr(model, "text_to_latent", None))
    elif task == "t2l":
        frozen |= _freeze_module(getattr(model, "latent_prefix", None))
        frozen |= _freeze_module(_get_text_head(model))
    return frozen


class ARLatentBridgeTrainer(nn.Module):
    def __init__(self, model: nn.Module, tokenizer, dtype: torch.dtype, config):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.config = config

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        latents = batch.get("latent", None)

        device = input_ids.device
        task = str(getattr(self.config.training, "task", "joint")).lower()
        text_w = float(getattr(self.config.training, "text_loss_weight", 1.0))
        latent_w = float(getattr(self.config.training, "latent_loss_weight", 1.0))

        text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        text_acc = 0.0
        if task in {"l2t", "joint"} and text_w > 0:
            if latents is None:
                raise ValueError("Batch is missing `latent`, required for l2t conditioning.")
            latents_in = latents.squeeze(1) if latents.dim() == 3 and latents.shape[1] == 1 else latents

            text_logits, _ = self.model(
                text_tokens=input_ids,
                attention_mask=attention_mask,
                latents=latents_in,
                mode="l2t",
            )
            if text_logits is None:
                raise RuntimeError("Model returned no text logits for mode='l2t'.")
            vocab_size = text_logits.shape[-1]

            labels = input_ids.clone()
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)

            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            text_loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            with torch.no_grad():
                pred = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                if mask.any():
                    text_acc = (pred[mask] == shift_labels[mask]).float().mean().item()

        latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        if task in {"t2l", "joint"} and latent_w > 0:
            if latents is None:
                raise ValueError("Batch is missing `latent`, required for t2l regression.")
            latent_target = latents
            if latent_target.dim() == 3 and latent_target.shape[1] == 1:
                latent_target = latent_target.squeeze(1)
            elif latent_target.dim() == 3 and latent_target.shape[1] > 1:
                latent_target = latent_target.mean(dim=1)
            latent_target = latent_target.to(device=device, dtype=self.dtype)

            _, latent_pred = self.model(
                text_tokens=input_ids,
                attention_mask=attention_mask,
                latents=None,
                mode="t2l",
            )
            if latent_pred is None:
                raise RuntimeError("Model returned no latent prediction for mode='t2l'.")
            latent_pred = latent_pred.to(dtype=self.dtype)
            latent_loss = F.mse_loss(latent_pred, latent_target)

        total_loss = text_w * text_loss + latent_w * latent_loss
        metrics = {
            "loss": float(total_loss.item()),
            "text_loss": float(text_loss.item()),
            "latent_loss": float(latent_loss.item()),
            "text_acc": float(text_acc),
        }
        return total_loss, metrics


def get_optimizer(config, trainer):
    params = [p for p in trainer.parameters() if p.requires_grad]
    if config.optimizer.name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=config.optimizer.eps,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer.name}")


def save_checkpoint(save_dir: Path, step: int, trainer: nn.Module, config) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "config": OmegaConf.to_container(config, resolve=True),
        "model": getattr(trainer, "model", trainer).state_dict(),
    }
    torch.save(payload, save_dir / f"checkpoint_step_{step}.pt")


@hydra.main(config_path="configs", config_name="wedlm8b_ar_bridge", version_base="1.1")
def main(config):
    local_rank, global_rank, world_size, is_main_process, is_distributed = _init_distributed()

    with open_dict(config):
        config.training.world_size = world_size
        config.training.local_rank = local_rank
        config.training.global_rank = global_rank

    seed = int(config.training.seed) + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype = parse_dtype(config.training.dtype)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device={device} dtype={dtype}")

    if config.training.resume is not None:
        raise NotImplementedError("Resume not implemented for WeDLM AR bridge trainer.")

    with main_process_first(local_rank):
        tokenizer = get_tokenizer(config)
    with main_process_first(local_rank):
        model = get_model(config, tokenizer, device=device, dtype=dtype)

    trainer = ARLatentBridgeTrainer(model=model, tokenizer=tokenizer, dtype=dtype, config=config).to(device=device)
    frozen_any = _maybe_freeze_unused_heads(trainer, config)
    optimizer = get_optimizer(config, trainer)

    with main_process_first(local_rank):
        train_dl, _ = get_simple_dataloaders(config, tokenizer)

    logger = Logger(is_main_process)
    os.environ.setdefault("WANDB_DIR", config.logging.get("wandb_dir", "./outputs/"))
    logger.init(
        name=config.logging.run_name,
        entity=config.logging.wandb_entity,
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )

    if config.training.compile_model:
        opt_trainer = torch.compile(trainer)
    else:
        opt_trainer = trainer

    if is_distributed:
        ddp_find_unused = getattr(config.training, "ddp_find_unused_parameters", None)
        if ddp_find_unused is None:
            task = str(getattr(config.training, "task", "joint")).lower()
            ddp_find_unused = not (bool(frozen_any) and task in {"l2t", "t2l"})
        ddp_kwargs = dict(
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(ddp_find_unused),
        )
        ddp_static_graph = getattr(config.training, "ddp_static_graph", None)
        if ddp_static_graph is None:
            task = str(getattr(config.training, "task", "joint")).lower()
            ddp_static_graph = bool(frozen_any) and task in {"l2t", "t2l"}
        if ddp_static_graph:
            ddp_kwargs["static_graph"] = True
        try:
            ddp_trainer = DDP(opt_trainer, **ddp_kwargs)
        except TypeError:
            ddp_kwargs.pop("static_graph", None)
            ddp_trainer = DDP(opt_trainer, **ddp_kwargs)
    else:
        ddp_trainer = opt_trainer

    max_lr = float(config.optimizer.lr)
    save_dir = Path(config.logging.save_dir) / config.logging.run_name

    batch_iterator = iter(train_dl)
    prev_time = time.time()

    with tqdm.tqdm(
        total=int(config.training.num_train_steps),
        initial=0,
        desc="Training",
        ncols=100,
        disable=not is_main_process,
        leave=True,
    ) as pbar:
        for step in range(int(config.training.num_train_steps)):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(train_dl)
                batch = next(batch_iterator)

            curr_lr = get_lr(config, max_lr, step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            loss, metrics = ddp_trainer(batch)
            optimizer.zero_grad(set_to_none=True)
            (loss * float(config.loss.loss_scale)).backward()

            if config.optimizer.get("grad_clip_norm", 0.0) and float(config.optimizer.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(trainer.parameters(), float(config.optimizer.grad_clip_norm))
            optimizer.step()

            if is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "text": f"{metrics.get('text_loss', 0.0):.4f}",
                        "latent": f"{metrics.get('latent_loss', 0.0):.4f}",
                        "acc": f"{metrics.get('text_acc', 0.0):.4f}",
                    }
                )
                pbar.update(1)

                if step % int(config.logging.log_freq) == 0:
                    now = time.time()
                    wall = now - prev_time
                    prev_time = now
                    logger.log({"lr": curr_lr, "wall_sec": wall, **metrics, "step": step})

                if step % int(config.logging.save_freq) == 0 and step > 0:
                    save_checkpoint(save_dir, step, trainer, config)

    if is_main_process:
        save_checkpoint(save_dir, int(config.training.num_train_steps), trainer, config)


if __name__ == "__main__":
    main()

