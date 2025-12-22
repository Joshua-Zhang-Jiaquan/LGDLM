# File: hdlm/trainer_latent.py
import torch
import torch.nn as nn
import torch.distributed as dist

from baseline_latent.diffusion_process import sample_t, NoiseSchedule
from baseline_latent.loss import Loss


class LatentConditionedDiffusionTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, loss_fn: Loss, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn
        self.dtype = dtype

        self.device = next(model.parameters()).device

        self.register_buffer("pad_id", torch.tensor(tokenizer.pad_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("mask_id", torch.tensor(tokenizer.mask_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("t0", torch.zeros(1, device=self.device))
        self.register_buffer("t1", torch.ones(1, device=self.device))

    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch, force_transitting=False):
        batch_size = batch["input_ids"].size(0)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            t = sample_t(self.config, batch_size, device=self.device)
            z_t = self.noise_schedule.sample_zt(batch["input_ids"], t)
            
            # Extract latent data if available
            latents = batch.get("latent", None)
            
            # Pass latents to model if available
            if latents is not None:
                logits = self.model(
                    z_t, 
                    t, 
                    latents=latents,  # NEW: pass latents
                    attention_mask=batch["attention_mask"]
                )
            else:
                logits = self.model(
                    z_t, 
                    t, 
                    attention_mask=batch["attention_mask"]
                )
                
            loss, _, metrics = self.loss_fn.forward(
                logits=logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                z_t=z_t,
                t=t,
                reduction=self.config.loss.reduction,
                force_transitting=force_transitting,
            )
        return loss, metrics