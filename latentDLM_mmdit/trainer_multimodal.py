# File: latentDLM_mmdit/trainer_multimodal.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_process import MaskedDiffusion


# Use the improved continuous diffusion
from .continuous_diffusion import ContinuousDiffusion


class MultimodalTrainer(nn.Module):
    """Trainer for multimodal DIT (text + latent generation)."""
    
    def __init__(self, model, tokenizer, text_noise_schedule, loss_fn, dtype):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.text_noise_schedule = text_noise_schedule
        self.latent_diffusion = ContinuousDiffusion()
        self.loss_fn = loss_fn
        self.dtype = dtype
        self.mask_token_id = tokenizer.mask_token_id
    
    def forward(self, batch, force_transitting=False):
        # Extract data
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        latents = batch.get("latent", None)
        
        batch_size = input_ids.shape[0]
        
        # Sample timesteps
        text_t = torch.rand(batch_size, device=input_ids.device)
        text_sigma = text_t.float()
        
        latent_t = self.latent_diffusion.sample_timesteps(batch_size, device=input_ids.device)
        latent_t = latent_t.float()
        
        # Apply text diffusion
        noisy_input_ids = self.text_noise_schedule.sample_zt(input_ids, text_sigma)
        text_target = input_ids
        text_mask = (noisy_input_ids == self.mask_token_id)
        
        # Apply latent diffusion
        if latents is not None:
            latents = latents.to(device=input_ids.device, dtype=self.dtype)
            if latents.dim() == 3:
                latents = latents.squeeze(1)
            
            noise = torch.randn_like(latents)
            noisy_latents, latent_target = self.latent_diffusion.add_noise(latents, latent_t, noise)
        else:
            noisy_latents = None
            latent_target = None
        
        # Forward pass
        outputs = self.model(
            indices=noisy_input_ids,
            sigma=text_sigma,
            latents=noisy_latents,
            latent_timesteps=latent_t,
            attention_mask=attention_mask
        )
        
        # Handle outputs (could be text only or text+latent)
        if isinstance(outputs, tuple):
            if len(outputs) == 2:
                # (text_logits, latent_pred)
                text_logits, latent_pred = outputs
                cluster_logits = None
            elif len(outputs) == 3:
                # (text_logits, latent_pred, cluster_logits)
                text_logits, latent_pred, cluster_logits = outputs
        else:
            # Only text logits
            text_logits = outputs
            latent_pred = None
            cluster_logits = None
        
        # Compute loss
        loss, metrics = self.loss_fn(
            text_logits=text_logits,
            text_target=text_target,
            text_mask=text_mask,
            latent_pred=latent_pred,
            latent_target=latent_target,
            cluster_logits=cluster_logits
        )
        
        return loss, metrics