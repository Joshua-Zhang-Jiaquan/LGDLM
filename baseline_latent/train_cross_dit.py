# File: baseline_latent/train_cross_dit.py
import datetime
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import hydra
import tqdm
import wandb
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn

import sys
sys.path.append("..")
sys.path.append(".")

# Import CROSS-ATTENTION components
from baseline_latent.models.dit_cross_attention import DITWithCrossAttention  # CHANGED
from baseline_latent.checkpoints import (
    save_checkpoint,
    load_checkpoint_for_training,
    TrainingState,
    save_rng_state,
    load_rng_state,
)
from baseline_latent.diffusion_process import get_noise_schedule, sample_t
from baseline_latent.modeling_latent import get_tokenizer
from baseline_latent.loss import get_loss
from baseline_latent.optimizer import get_optimizer
from baseline_latent.utils import (
    get_lr,
    parse_dtype,
    calculate_flops_per_batch,
)
from baseline_latent.data_simple import get_simple_dataloaders

class Logger:
    def __init__(self, is_main_process):
        self.is_main_process = is_main_process

    def init(self, *args, **kwargs):
        if self.is_main_process:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_main_process:
            wandb.log(*args, **kwargs)


@contextmanager
def main_process_first():
    if dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
    else:
        yield


class CrossAttentionTrainer(nn.Module):
    """Trainer for DIT with cross-attention latent conditioning."""
    
    def __init__(self, config, model, tokenizer, noise_schedule, loss_fn, dtype):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn
        self.dtype = dtype
        
        self.device = next(model.parameters()).device

    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)
        
    def forward(self, batch, force_transitting=False):
        batch_size = batch["input_ids"].size(0)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            # Sample timesteps
            t = sample_t(self.config, batch_size, device=self.device)

            # Apply noise schedule
            z_t = self.noise_schedule.sample_zt(batch["input_ids"], t)

            # Extract latent data
            latents = batch.get("latent", None)
            latent_mask = batch.get("latent_mask", None)

            # Forward pass with cross-attention
            if latents is not None:
                latents = latents.to(device=self.device)
                if latent_mask is not None:
                    latent_mask = latent_mask.to(device=self.device)

                # Cross-attention model expects latent_mask
                outputs = self.model(
                    indices=z_t,
                    sigma=t,
                    latents=latents,
                    latent_mask=latent_mask,  # NEW: Pass mask
                    attention_mask=batch["attention_mask"],
                    return_latent_pred=self.config.model.get('predict_latents', False)
                )
            else:
                outputs = self.model(
                    indices=z_t,
                    sigma=t,
                    attention_mask=batch["attention_mask"]
                )

            # Handle outputs: could be (logits,) or (logits, latent_pred)
            if isinstance(outputs, tuple):
                if len(outputs) == 2:
                    # (logits, latent_pred)
                    logits, latent_pred = outputs
                    cluster_logits = None
                elif len(outputs) == 3:
                    # (logits, cluster_logits, latent_pred)
                    logits, cluster_logits, latent_pred = outputs
                else:
                    logits = outputs[0]
                    cluster_logits = outputs[1] if len(outputs) > 1 else None
                    latent_pred = None
            else:
                logits = outputs
                cluster_logits = None
                latent_pred = None

            # Prepare loss arguments
            loss_args = {
                "logits": logits,
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "z_t": z_t,
                "t": t,
                "reduction": self.config.loss.reduction,
                "force_transitting": force_transitting,
            }
            
            if cluster_logits is not None and self.config.model.diffusion_process == "baseline_latent":
                loss_args["cluster_logits"] = cluster_logits

            # Compute loss
            loss, _, metrics = self.loss_fn.forward(**loss_args)

            # Add latent prediction loss if available
            if latent_pred is not None and latents is not None:
                latent_loss = torch.nn.functional.mse_loss(
                    latent_pred, 
                    latents.mean(dim=1) if latents.dim() == 3 else latents
                )
                loss = loss + self.config.model.get('latent_prediction_weight', 0.1) * latent_loss
                metrics["latent_pred_loss"] = latent_loss.item()

            # Add latent metrics
            if latents is not None:
                metrics["latent_norm"] = latents.norm(dim=-1).mean().item()
                metrics["latent_std"] = latents.std().item()

        return loss, metrics


@hydra.main(config_path="configs", config_name="baseline_latent_cross_attention", version_base="1.1")  # NEW config
def main(config):
    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=30),
            device_id=torch.device("cuda", local_rank),
        )
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        is_main_process = (global_rank == 0)
    except RuntimeError:
        print("Distributed training not available, running on single device.")
        world_size = 1
        local_rank = 0
        global_rank = 0
        is_main_process = True
    
    with open_dict(config):
        config.training.world_size = world_size
    
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    seed = config.training.seed + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    dtype = parse_dtype(config.training.dtype)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=} and {dtype=}")

    if config.training.resume is None:
        tokenizer = get_tokenizer(config)
        vocab_size = len(tokenizer)
        
        # Use CROSS-ATTENTION DIT
        model = DITWithCrossAttention(
            config.model,
            vocab_size=vocab_size,
            latent_dim=config.model.get("latent_dim", 768),
            cluster_size=config.model.get("cluster_size", 0)
        ).to(dtype).to(device)
        
        noise_schedule = get_noise_schedule(config, tokenizer)
        loss_fn = get_loss(config, tokenizer, noise_schedule)
        
        trainer = CrossAttentionTrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            noise_schedule=noise_schedule,
            loss_fn=loss_fn,
            dtype=dtype
        ).to(device)
        
        optimizer = get_optimizer(config, trainer)
        
        state = TrainingState(
            epoch=0,
            epoch_start_step=0,
            step=0,
        )
    else:
        # Handle checkpoint loading
        (
            model,
            noise_schedule,
            tokenizer,
            old_config,
            trainer,
            optimizer,
            state
        ) = load_checkpoint_for_training(config.training.resume, device=device, dtype=dtype)
    
    # Get dataloaders
    with main_process_first():
        train_dl, test_dl = get_simple_dataloaders(config, tokenizer)

    max_lr = config.optimizer.lr

    logger = Logger(is_main_process)
    os.environ["WANDB_DIR"] = config.logging.get("wandb_dir", "./outputs/")
    logger.init(
        name=config.logging.run_name,
        entity=config.logging.wandb_entity,
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )

    if is_main_process:
        pwd = Path(".").resolve()
        wandb.config.update({"pwd": pwd})
        print(f"Working directory: {pwd}")
        
        # Print cross-attention specific info
        print(f"\n{'='*60}")
        print("TRAINING DIT WITH CROSS-ATTENTION LATENT CONDITIONING")
        print(f"{'='*60}")
        print(f"Cross-attention frequency: every {config.model.get('cross_attention_frequency', 2)} blocks")
        print(f"Latent prediction: {config.model.get('predict_latents', False)}")
        print(f"Latent prediction weight: {config.model.get('latent_prediction_weight', 0.1)}")
        print(f"{'='*60}")

    non_emb_params = sum(p.numel() for p in model.blocks.parameters())
    flops_per_batch = calculate_flops_per_batch(config, model, len(tokenizer), non_emb_params, method="hoffmann")
    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)

    if config.training.compile_model:
        opt_trainer = torch.compile(trainer)
    else:
        opt_trainer = trainer

    if is_distributed:
        ddp_trainer = DDP(opt_trainer, device_ids=[device.index])
    else:
        ddp_trainer = opt_trainer

    if is_main_process:
        non_emb_params_str = f"{non_emb_params / 1e6:.1f}M" if non_emb_params < 500 * 1e6 else f"{non_emb_params / 1e9:.1f}B"
        trainable_params_str = f"{trainable_params / 1e6:.1f}M" if trainable_params < 500 * 1e6 else f"{trainable_params / 1e9:.1f}B"
        print(f"*** Starting DIT with CROSS-ATTENTION Latent Conditioning ***")
        print(f"* World size: {world_size}")
        print(f"* FLOPS per batch: {flops_per_batch:.3g}")
        print(f"* Per-device batch size: {config.training.train_batch_size}")
        print(f"* Total batch size: {config.training.train_batch_size * world_size}")
        print(f"* Non-embedding parameters: {non_emb_params_str}")
        print(f"* Trainable parameters: {trainable_params_str}")
        print(f"* Model dtype: {next(iter(model.parameters())).dtype}")
        print(f"* Latent dimension: {config.model.get('latent_dim', 768)}")
        print(f"*************************")

    # ... rest of training loop is SAME as train_latent_dit.py ...
    # Copy the training loop from your existing train_latent_dit.py
    # (Everything from "if is_distributed and hasattr(train_dl.sampler, "set_epoch"):" onward)

    # Training loop
    if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
        train_dl.sampler.set_epoch(state.epoch)
    batch_iterator = iter(train_dl)

    # Initialize eval dataloader
    _ = next(iter(test_dl))

    if state.step - state.epoch_start_step > 0:
        for _ in tqdm.trange(state.step - state.epoch_start_step, desc="Skipping batches", dynamic_ncols=True, disable=not is_main_process):
            next(batch_iterator)

    curr_time = time.time()
    trained_time = 0 if config.training.resume is None else (state.start_time - state.curr_time)
    state.start_time = curr_time - trained_time
    state.curr_time = curr_time
    prev_time = curr_time

    log_buffer = []

    if config.training.resume is not None:
        load_rng_state(config.training.resume, global_rank)

    with tqdm.tqdm(total=config.training.num_train_steps, initial=state.step, desc="Training", dynamic_ncols=True, disable=not is_main_process) as pbar:
        for step in range(state.step, config.training.num_train_steps):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                state.epoch += 1
                state.epoch_start_step = step
                if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
                    train_dl.sampler.set_epoch(state.epoch)
                batch_iterator = iter(train_dl)
                batch = next(batch_iterator)

            curr_lr = get_lr(config, max_lr, step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, metrics = ddp_trainer(batch)
            
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}, LR = {curr_lr:.6f}")

            (loss * config.loss.loss_scale).backward()

            if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)
            else:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
            if torch.isnan(norm):
                print(f"Warning: NaN gradient detected at step {step}")
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.zero_()

            optimizer.step()
            optimizer.zero_grad()

            # Logging
            batch_tokens = batch.get("attention_mask", torch.ones_like(batch["input_ids"])).sum().item() * config.training.world_size
            batch_flops = flops_per_batch * config.training.world_size
            total_batch_size = batch["input_ids"].size(0) * config.training.world_size
            
            state.total_tokens += batch_tokens
            state.total_flops += batch_flops

            curr_time = time.time()
            step_time = curr_time - prev_time
            prev_time = curr_time

            log_buffer.append({
                "train/loss": loss.item(),
                "train/lr": curr_lr,
                "train/step": step + 1,
                "train/grad_norm": norm.item(),
                "train/epoch": step / len(train_dl),
                "train/total_tokens": state.total_tokens,
                "train/total_flops": state.total_flops,
                "train/tokens_per_sec": batch_tokens / step_time,
                "train/flops_per_sec": batch_flops / step_time,
                "train/samples_per_sec": total_batch_size / step_time,
                "train/it_per_sec": 1 / step_time,
                "train/avg_it_per_sec": (step + 1) / (curr_time - state.start_time),
                **{f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()},
            })

            if ((step + 1) % config.logging.log_freq) == 0:
                metrics = {k: sum(d[k] for d in log_buffer) / len(log_buffer) for k in log_buffer[0]}
                logger.log({k: v for k, v in metrics.items()}, step=step)
                logger.log({"trainer/global_step": step}, step=step)
                log_buffer = []

            # Evaluation
            if ((step + 1) % config.logging.eval_freq) == 0:
                with torch.no_grad():
                    eval_start_time = time.time()
                    model.eval()

                    eval_metrics = {}
                    eval_loss = 0
                    num_eval_samples = 0
                    for i, test_batch in enumerate(tqdm.tqdm(test_dl, desc="Eval", dynamic_ncols=True, total=config.logging.num_eval_batches, disable=not is_main_process)):
                        bs = test_batch["input_ids"].size(0)

                        test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
                        loss, metrics = ddp_trainer(test_batch)

                        for k, v in metrics.items():
                            eval_metrics[k] = eval_metrics.get(k, 0) + (v.item() if isinstance(v, torch.Tensor) else v) * bs

                        eval_loss += loss.item() * bs
                        num_eval_samples += bs

                        if i >= config.logging.num_eval_batches - 1:
                            break

                    eval_elapsed_time = time.time() - eval_start_time
                    logger.log({
                        "eval/loss": eval_loss / num_eval_samples,
                        "eval/time_taken": eval_elapsed_time,
                        **{f"eval/{k}": v / num_eval_samples for k, v in eval_metrics.items()},
                    }, step=step)
                    model.train()

            # Save checkpoint
            state.step += 1
            if ((step + 1) % config.logging.save_freq) == 0:
                dist.barrier()
                output_path = Path(config.logging.save_dir, config.logging.run_name)
                suffix = "latest"
                if (step + 1) == 500000:
                    suffix = "-500k"
                elif (step + 1) == 1000000:
                    suffix = "-1M"
                elif (step + 1) == 250000:
                    suffix = "-250k"
                
                output_path = output_path / suffix
                if is_main_process:
                    save_checkpoint(output_path, trainer, optimizer, state)
                dist.barrier()
                output_path.mkdir(exist_ok=True, parents=True)
                save_rng_state(output_path, global_rank)
                dist.barrier()

            pbar.update(1)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()