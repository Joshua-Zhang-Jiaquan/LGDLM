import datetime
import json
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

# Allow running from repo root or subdirs
sys.path.append("..")
sys.path.append(".")
os.environ["WANDB_MODE"] = "disabled"

# Import MMDiT components
from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
from latentDLM_mmdit.checkpoints_mmdit import (
    save_checkpoint,
    load_checkpoint_for_training,
    TrainingState,
    save_rng_state,
    load_rng_state,
)
from latentDLM_mmdit.modeling_mmdit import get_tokenizer
from latentDLM_mmdit.optimizer import get_optimizer
from latentDLM_mmdit.utils import (
    get_lr,
    parse_dtype,
    calculate_flops_per_batch,
)
from latentDLM_mmdit.data_simple import get_simple_dataloaders

from latentDLM_mmdit.diffusion_process import MaskedDiffusion


class Logger:
    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process

    def init(self, *args, **kwargs):
        if self.is_main_process:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_main_process:
            wandb.log(*args, **kwargs)


def safe_barrier(local_rank: int | None = None) -> None:
    """Call dist.barrier() only when the default process group is initialized.

    Newer PyTorch versions may accept `device_ids` to disambiguate CUDA device.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if local_rank is not None:
            dist.barrier(device_ids=[local_rank])  # type: ignore[arg-type]
        else:
            dist.barrier()
    except TypeError:
        # Older PyTorch does not accept device_ids
        dist.barrier()


@contextmanager
def main_process_first(local_rank: int | None = None):
    """Context manager: rank0 runs the enclosed code first, others wait."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            safe_barrier(local_rank)
        else:
            safe_barrier(local_rank)
            yield
    else:
        yield


# Import our improved continuous diffusion (STABLE VERSION with NaN fixes)
from latentDLM_mmdit.continuous_diffusion import ContinuousDiffusion
from latentDLM_mmdit.improved_trainer_stable import MultimodalDiffusionTrainer
from latentDLM_mmdit.improved_trainer_stable import MultimodalDiffusionTrainer_new

def _init_distributed() -> tuple[int, int, int, bool, bool]:
    """Initialize distributed training if launched with torchrun.

    Returns:
        local_rank, global_rank, world_size, is_main_process, is_distributed
    """
    env_has_ddp = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not env_has_ddp:
        return 0, 0, 1, True, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch detected (torchrun), but CUDA is not available.")

    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Safety check: ensure the GPU device exists before setting it
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {num_gpus} GPU(s) available. "
            f"Please ensure --nproc_per_node <= {num_gpus} and CUDA_VISIBLE_DEVICES is set correctly."
        )
    
    torch.cuda.set_device(local_rank)

    init_kwargs = dict(
        backend="nccl",
        timeout=datetime.timedelta(minutes=30),
        init_method="env://",
    )

    try:
        # Newer PyTorch supports device_id to set PG default device.
        dist.init_process_group(**init_kwargs, device_id=torch.device("cuda", local_rank))  # type: ignore[arg-type]
    except TypeError:
        dist.init_process_group(**init_kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize torch.distributed process group. "
            f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
        ) from e

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = (global_rank == 0)
    is_distributed = dist.is_available() and dist.is_initialized()
    return local_rank, global_rank, world_size, is_main_process, is_distributed


@hydra.main(config_path="configs", config_name="mmdit", version_base="1.1")
def main(config):
    # ---------------- Distributed init ----------------
    local_rank, global_rank, world_size, is_main_process, is_distributed = _init_distributed()

    with open_dict(config):
        config.training.world_size = world_size
        config.training.local_rank = local_rank
        config.training.global_rank = global_rank

    # ---------------- Seeding ----------------
    seed = config.training.seed + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ---------------- CUDA perf knobs ----------------
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # BF16-specific optimizations (if available)
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if hasattr(torch.backends.cudnn, 'allow_bf16_reduced_precision_reduction'):
        torch.backends.cudnn.allow_bf16_reduced_precision_reduction = True
    print("BF16 optimizations enabled")
        
    try:
        torch.backends.cuda.enable_flash_sdp(True)  # PyTorch 2.1+
    except Exception:
        pass

    dtype = parse_dtype(config.training.dtype)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    print(f"Using device={device} and dtype={dtype}")

    # ---------------- Build / load model + trainer ----------------
    if config.training.resume is None:
        tokenizer = get_tokenizer(config)
        vocab_size = len(tokenizer)

        model = MultimodalMMDiT(
            config=config.model,
            vocab_size=vocab_size,
            latent_dim=config.model.get("latent_dim", 768),
            cluster_size=config.model.get("cluster_size", 0),
        ).to(device=device, dtype=dtype)
        
        

        text_noise_schedule = MaskedDiffusion(tokenizer)

        trainer = MultimodalDiffusionTrainer_new(
            model=model,
            tokenizer=tokenizer,
            text_noise_schedule=text_noise_schedule,
            dtype=dtype,
            config=config
        ).to(device=device)

        optimizer = get_optimizer(config, trainer)

        state = TrainingState(
            epoch=0,
            epoch_start_step=0,
            step=0,
        )
    else:
        (
            model,
            text_noise_schedule,
            tokenizer,
            old_config,
            trainer,
            optimizer,
            state,
        ) = load_checkpoint_for_training(config.training.resume, device=device, dtype=dtype)

    # ---------------- Data ----------------
    with main_process_first(local_rank):
        if config.data.get('load_tokens', True):
            # We don't need tokenizer for data loading when using pre-tokenized
            train_dl, test_dl = get_simple_dataloaders(config, tokenizer=None)
        else:
            train_dl, test_dl = get_simple_dataloaders(config, tokenizer)

    max_lr = config.optimizer.lr

    # ---------------- Logging (W&B) ----------------
    logger = Logger(is_main_process)
    # Respect external WANDB_DIR if set; otherwise use config
    os.environ.setdefault("WANDB_DIR", config.logging.get("wandb_dir", "./outputs/"))
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

    non_emb_params = sum(p.numel() for p in model.mmdit.parameters())
    flops_per_batch = calculate_flops_per_batch(
        config, model, len(tokenizer), non_emb_params, method="hoffmann"
    )
    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)

    # ---------------- Optional compilation ----------------
    if config.training.compile_model:
        try:
            opt_trainer = torch.compile(trainer)
        except RuntimeError as e:
            if "Python 3.13" in str(e):
                print("Warning: torch.compile not supported on Python 3.13+, skipping compilation")
                opt_trainer = trainer
            else:
                raise
    else:
        opt_trainer = trainer

    # ---------------- DDP wrap ----------------
    if is_distributed:
        ddp_trainer = DDP(opt_trainer, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    else:
        ddp_trainer = opt_trainer

    if is_main_process:
        non_emb_params_str = (
            f"{non_emb_params / 1e6:.1f}M" if non_emb_params < 500 * 1e6 else f"{non_emb_params / 1e9:.1f}B"
        )
        trainable_params_str = (
            f"{trainable_params / 1e6:.1f}M" if trainable_params < 500 * 1e6 else f"{trainable_params / 1e9:.1f}B"
        )
        print("*** Starting MMDiT with Joint Text-Latent Diffusion ***")
        print(f"* World size: {world_size}")
        print(f"* FLOPs per batch: {flops_per_batch:.3g}")
        print(f"* Per-device batch size: {config.training.train_batch_size}")
        print(f"* Total batch size: {config.training.train_batch_size * world_size}")
        print(f"* Non-embedding parameters: {non_emb_params_str}")
        print(f"* Trainable parameters: {trainable_params_str}")
        print(f"* Model dtype: {next(iter(model.parameters())).dtype}")
        print(f"* Latent dimension: {config.model.get('latent_dim', 768)}")
        print("* Text diffusion: Masked Diffusion")
        print(
            f"* Latent diffusion: Continuous Diffusion "
            f"(beta={config.model.get('latent_beta_min', 0.0001)}-{config.model.get('latent_beta_max', 0.02)})"
        )
        print("*************************")

    # ---------------- Training loop setup ----------------
    if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
        train_dl.sampler.set_epoch(state.epoch)
    batch_iterator = iter(train_dl)

    # Calculate total steps based on epochs
    num_epochs = config.training.num_epochs
    total_batches = len(train_dl)
    total_steps = total_batches * num_epochs

    # Skip batches if resuming
    if state.step - state.epoch_start_step > 0:
        for _ in tqdm.trange(
            state.step - state.epoch_start_step,
            desc="Skipping batches",
            dynamic_ncols=True,
            disable=not is_main_process,
        ):
            next(batch_iterator)


    curr_time = time.time()
    trained_time = 0 if config.training.resume is None else (state.start_time - state.curr_time)
    state.start_time = curr_time - trained_time
    state.curr_time = curr_time
    prev_time = curr_time

    log_buffer = []

    if config.training.resume is not None:
        load_rng_state(config.training.resume, global_rank)

    # Initialize loss logging
    loss_log_file = None
    if is_main_process:
        log_dir = Path(config.logging.save_dir) / config.logging.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        loss_log_file = log_dir / "training_log.jsonl"
        loss_log_file.write_text("")


    # ---------------- Train ----------------
    with tqdm.tqdm(
        total=total_steps,
        initial=state.step,
        desc="Training",
        ncols=100,
        disable=not is_main_process,
        leave=True,
    ) as pbar:
            
        # ---------------- Train ----------------
        # Train until we complete all epochs
        while state.epoch < num_epochs:
            try:
                batch = next(batch_iterator)
            except StopIteration:
                # End of epoch
                state.epoch += 1
                state.epoch_start_step = state.step
                
                # Check if training is complete
                if state.epoch >= num_epochs:
                    break
                    
                # Reset for next epoch
                if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
                    train_dl.sampler.set_epoch(state.epoch)
                batch_iterator = iter(train_dl)
                batch = next(batch_iterator)
            
            # Calculate learning rate
            curr_lr = get_lr(config, max_lr, state.step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Update trainer's current epoch if needed
            if hasattr(ddp_trainer, 'module'):
                ddp_trainer.module.current_epoch = state.epoch
            else:
                ddp_trainer.current_epoch = state.epoch
                
            # Forward pass - pass step for sequential scheduling
            loss, metrics = ddp_trainer(batch, step=state.step)
            
            # Update progress bar (every 10 steps)
            if state.step % 10 == 0 and is_main_process:
                progress_in_epoch = (state.step - state.epoch_start_step) / total_batches
                pbar.set_postfix({
                    "Epoch": f"{state.epoch}/{num_epochs}",
                    "Progress": f"{progress_in_epoch*100:.1f}%",
                    "Loss": f"{loss.item():.4f}",
                    "Text": f"{metrics.get('text_loss', 0.0):.4f}",
                    "Latent": f"{metrics.get('latent_loss', 0.0):.4f}",
                    "Acc": f"{metrics.get('text_accuracy', 0.0):.4f}",
                })
            
            # Log to file (every step)
            if is_main_process and loss_log_file is not None:
                progress_in_epoch = (state.step - state.epoch_start_step)
                log_entry = {
                    "epoch": int(state.epoch),
                    "step": int(state.step),
                    "batch_in_epoch": int(progress_in_epoch),
                    "loss": float(loss.item()),
                    "lr": float(curr_lr),
                }
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        log_entry[key] = float(value)
                    elif hasattr(value, "item"):
                        log_entry[key] = float(value.item())
                with open(loss_log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")



            # ===== LOSS VALIDATION =====
            # Check for invalid loss BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'='*60}")
                print(f"ERROR: Invalid loss detected at step {state.step}")
                print(f"{'='*60}")
                print(f"  Total loss: {loss.item()}")
                print(f"  Text loss: {metrics.get('text_loss', 0.0):.6f}")
                print(f"  Latent loss: {metrics.get('latent_loss', 0.0):.6f}")
                print(f"  Mode: {metrics.get('mode', 'unknown')}")
                print(f"  Learning rate: {curr_lr:.2e}")
                print(f"  Skipping this batch...")
                print(f"{'='*60}\n")

                # Skip this batch entirely
                optimizer.zero_grad(set_to_none=True)
                state.step += 1
                pbar.update(1)
                continue

            # ===== BACKWARD PASS =====
            (loss * config.loss.loss_scale).backward()

            # ===== GRADIENT CLIPPING =====
            # Clip gradients BEFORE checking for NaN
            grad_clip_value = config.optimizer.get('grad_clip_norm', 1.0)
            if grad_clip_value and grad_clip_value > 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    trainer.parameters(),
                    grad_clip_value,
                    error_if_nonfinite=False
                )
            else:
                norm = torch.nn.utils.clip_grad_norm_(
                    trainer.parameters(),
                    1.0,
                    error_if_nonfinite=False
                )

            # ===== NaN/INF GRADIENT CHECK =====
            # Check AFTER clipping
            if torch.isnan(norm) or torch.isinf(norm):
                print(f"\n{'='*60}")
                print(f"ERROR: Invalid gradient norm at step {state.step}")
                print(f"{'='*60}")
                print(f"  Gradient norm: {norm.item() if not torch.isnan(norm) else 'NaN'}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Text loss: {metrics.get('text_loss', 0.0):.6f}")
                print(f"  Latent loss: {metrics.get('latent_loss', 0.0):.6f}")
                print(f"  Learning rate: {curr_lr:.2e}")
                print(f"  Skipping optimizer step and resetting gradients...")
                print(f"{'='*60}\n")

                # Reset gradients and skip optimizer step
                optimizer.zero_grad(set_to_none=True)
                state.step += 1
                pbar.update(1)
                continue

            # ===== OPTIMIZER STEP =====
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Logging throughput
            batch_tokens = (
                batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
                .sum()
                .item()
                * config.training.world_size
            )
            batch_flops = flops_per_batch * config.training.world_size
            total_batch_size = batch["input_ids"].size(0) * config.training.world_size

            state.total_tokens += batch_tokens
            state.total_flops += batch_flops

            curr_time = time.time()
            step_time = curr_time - prev_time
            prev_time = curr_time

            log_buffer.append(
                {
                    "train/loss": float(loss.item()),
                    "train/lr": float(curr_lr),
                    "train/step": int(state.step + 1),  # CHANGED: step -> state.step
                    "train/grad_norm": float(norm.item()),
                    "train/epoch": float(state.epoch + (state.step - state.epoch_start_step) / total_batches),
                    "train/total_tokens": float(state.total_tokens),
                    "train/total_flops": float(state.total_flops),
                    "train/tokens_per_sec": float(batch_tokens / step_time),
                    "train/flops_per_sec": float(batch_flops / step_time),
                    "train/samples_per_sec": float(total_batch_size / step_time),
                    "train/it_per_sec": float(1.0 / step_time),
                    "train/avg_it_per_sec": float((state.step + 1) / (curr_time - state.start_time))  # CHANGED: step -> state.step
                }
            )

            if ((state.step + 1) % config.logging.log_freq) == 0:  # CHANGED: step -> state.step
                avg_metrics = {k: sum(d[k] for d in log_buffer) / len(log_buffer) for k in log_buffer[0]}
                logger.log(avg_metrics, step=state.step)  # CHANGED: step -> state.step
                logger.log({"trainer/global_step": state.step}, step=state.step)  # CHANGED: step -> state.step
                log_buffer = []

            # ---------------- Evaluation ----------------
                    
            if ((state.step + 1) % config.logging.eval_freq) == 0:  # CHANGED: step -> state.step
                with torch.no_grad():
                    eval_start_time = time.time()
                    ddp_trainer.eval()

                    eval_metrics = {}
                    eval_loss = 0.0
                    num_eval_samples = 0

                    for i, test_batch in enumerate(
                        tqdm.tqdm(
                            test_dl,
                            desc="Eval",
                            dynamic_ncols=True,
                            total=config.logging.num_eval_batches,
                            disable=not is_main_process,
                        )
                    ):
                        bs = test_batch["input_ids"].size(0)

                        test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
                        e_loss, e_metrics = ddp_trainer(test_batch)

                        # FIX THIS: Only accumulate numeric metrics
                        for k, v in e_metrics.items():
                            try:
                                # Try to convert to float
                                eval_metrics[k] = eval_metrics.get(k, 0.0) + float(v) * bs
                            except (ValueError, TypeError):
                                # Skip non-numeric metrics
                                pass

                        eval_loss += float(e_loss.item()) * bs
                        num_eval_samples += bs

                        if i >= config.logging.num_eval_batches - 1:
                            break

                    eval_elapsed_time = time.time() - eval_start_time

                    # Re-enable this logging if you want eval logging
                    if is_main_process and num_eval_samples > 0:
                        logger.log(
                            {
                                "eval/loss": eval_loss / num_eval_samples,
                                "eval/time_taken": eval_elapsed_time,
                                **{f"eval/{k}": v / num_eval_samples for k, v in eval_metrics.items()},
                            },
                            step=state.step,  # CHANGED: step -> state.step
                        )

                    ddp_trainer.train()

            # ---------------- Save checkpoint ----------------
            state.step += 1
            if ((state.step) % config.logging.save_freq) == 0:  # CHANGED: step -> state.step
                output_path = Path(config.logging.save_dir, config.logging.run_name)
                suffix = "latest"
                if (state.step) == 500000:
                    suffix = "-500k"
                elif (state.step) == 1000000:
                    suffix = "-1M"
                elif (state.step) == 250000:
                    suffix = "-250k"
                output_path = output_path / suffix

                # Ensure directory exists (rank0), then sync.
                if is_main_process:
                    output_path.mkdir(exist_ok=True, parents=True)
                safe_barrier(local_rank)

                # Save checkpoint on rank0 only.
                if is_main_process:
                    save_checkpoint(output_path, trainer, optimizer, state)

                # Sync to ensure checkpoint is fully written.
                safe_barrier(local_rank)

                # Save RNG state per-rank (requires directory to exist).
                save_rng_state(output_path, global_rank)

                # Final sync (optional).
                safe_barrier(local_rank)

            pbar.update(1)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()