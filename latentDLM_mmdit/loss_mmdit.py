# File: hdlm/loss_multimodal.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalMDLMLoss(nn.Module):
    """Loss for multimodal masked diffusion (text + continuous latent)."""
    
    def __init__(self, tokenizer, text_noise_schedule, latent_loss_weight=1.0):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_noise_schedule = text_noise_schedule
        self.latent_loss_weight = latent_loss_weight
        self.mask_token_id = tokenizer.mask_token_id
    
    def forward(self, text_logits, text_target, text_mask, latent_pred=None, latent_target=None):
        # Text loss (masked language modeling)
        text_loss = F.cross_entropy(
            text_logits.view(-1, len(self.tokenizer)),
            text_target.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Latent loss (MSE for continuous diffusion)
        latent_loss = 0
        if latent_pred is not None and latent_target is not None:
            latent_loss = F.mse_loss(latent_pred, latent_target)
        
        # Combined loss
        total_loss = text_loss + self.latent_loss_weight * latent_loss
        
        return total_loss, {
            'text_loss': text_loss.item(),
            'latent_loss': latent_loss.item() if latent_loss > 0 else 0,
            'total_loss': total_loss.item()
        }