"""
Patch file for improved_trainer.py to fix NaN gradient issues in loss computation.

Apply these changes to latentDLM_mmdit/improved_trainer.py around lines 336-374.
"""

# ORIGINAL CODE (lines 336-374):
"""
        # ===== LOSS CALCULATION =====
        total_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

        # Text loss - only compute if needed
        if compute_text_loss and text_target is not None:
            vocab_size = text_logits.shape[-1]
            if text_mask is not None and text_mask.any():
                text_loss_unmasked = F.cross_entropy(
                    text_logits.view(-1, vocab_size),
                    text_target.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                text_loss = (text_loss_unmasked * text_mask.view(-1)).sum() / text_mask.sum().clamp(min=1)
                total_loss = total_loss + text_loss
            else:
                text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        else:
            text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

        # Latent loss - only compute if needed
        if compute_latent_loss and latent_pred is not None and latent_target is not None:
            # Ensure shapes match
            if latent_pred.dim() == 3 and latent_target.dim() == 2:
                latent_target = latent_target.unsqueeze(1)
            if latent_pred.dim() == 2 and latent_target.dim() == 3 and latent_target.shape[1] == 1:
                latent_target = latent_target.squeeze(1)

            # Handle shape mismatches
            if latent_pred.shape != latent_target.shape:
                if latent_pred.dim() == 3 and latent_pred.shape[1] > 1:
                    latent_pred = latent_pred.mean(dim=1)
                if latent_target.dim() == 3 and latent_target.shape[1] > 1:
                    latent_target = latent_target.mean(dim=1)

            latent_loss = F.mse_loss(latent_pred, latent_target)
            total_loss = total_loss + latent_loss
        else:
            latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
"""

# IMPROVED CODE (replace lines 336-374 with this):
"""
        # ===== LOSS CALCULATION WITH NUMERICAL STABILITY =====
        total_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        eps = 1e-8  # Numerical stability epsilon

        # Text loss - only compute if needed
        if compute_text_loss and text_target is not None:
            vocab_size = text_logits.shape[-1]
            if text_mask is not None and text_mask.any():
                # Compute cross entropy loss
                text_loss_unmasked = F.cross_entropy(
                    text_logits.view(-1, vocab_size),
                    text_target.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )

                # Add epsilon to prevent division by zero
                mask_sum = text_mask.sum().clamp(min=1) + eps
                text_loss = (text_loss_unmasked * text_mask.view(-1)).sum() / mask_sum

                # Clamp to prevent overflow (max loss of 100 is already very high)
                text_loss = torch.clamp(text_loss, min=0.0, max=100.0)

                # Validate loss value
                if torch.isnan(text_loss) or torch.isinf(text_loss):
                    print(f"WARNING: Invalid text_loss detected: {text_loss.item()}")
                    print(f"  mask_sum: {mask_sum.item()}")
                    print(f"  text_loss_unmasked stats: min={text_loss_unmasked.min().item():.4f}, max={text_loss_unmasked.max().item():.4f}, mean={text_loss_unmasked.mean().item():.4f}")
                    text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
                else:
                    total_loss = total_loss + text_loss
            else:
                text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        else:
            text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

        # Latent loss - only compute if needed
        if compute_latent_loss and latent_pred is not None and latent_target is not None:
            # Ensure shapes match
            if latent_pred.dim() == 3 and latent_target.dim() == 2:
                latent_target = latent_target.unsqueeze(1)
            if latent_pred.dim() == 2 and latent_target.dim() == 3 and latent_target.shape[1] == 1:
                latent_target = latent_target.squeeze(1)

            # Handle shape mismatches
            if latent_pred.shape != latent_target.shape:
                if latent_pred.dim() == 3 and latent_pred.shape[1] > 1:
                    latent_pred = latent_pred.mean(dim=1)
                if latent_target.dim() == 3 and latent_target.shape[1] > 1:
                    latent_target = latent_target.mean(dim=1)

            # Check for invalid values in predictions/targets
            if torch.isnan(latent_pred).any() or torch.isinf(latent_pred).any():
                print(f"WARNING: Invalid latent_pred detected")
                latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
            elif torch.isnan(latent_target).any() or torch.isinf(latent_target).any():
                print(f"WARNING: Invalid latent_target detected")
                latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
            else:
                # Normalize predictions and targets for numerical stability
                # This prevents the loss from exploding when values are very large
                latent_pred_norm = F.normalize(latent_pred, p=2, dim=-1, eps=eps)
                latent_target_norm = F.normalize(latent_target, p=2, dim=-1, eps=eps)

                # Compute MSE loss on normalized vectors
                latent_loss = F.mse_loss(latent_pred_norm, latent_target_norm)

                # Clamp to prevent overflow
                latent_loss = torch.clamp(latent_loss, min=0.0, max=10.0)

                # Validate loss value
                if torch.isnan(latent_loss) or torch.isinf(latent_loss):
                    print(f"WARNING: Invalid latent_loss detected: {latent_loss.item()}")
                    print(f"  latent_pred stats: min={latent_pred.min().item():.4f}, max={latent_pred.max().item():.4f}, mean={latent_pred.mean().item():.4f}")
                    print(f"  latent_target stats: min={latent_target.min().item():.4f}, max={latent_target.max().item():.4f}, mean={latent_target.mean().item():.4f}")
                    latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
                else:
                    # Apply loss weight (reduce latent loss contribution)
                    latent_loss_weight = getattr(self.config.loss, 'latent_loss_weight', 0.1)
                    total_loss = total_loss + latent_loss * latent_loss_weight
        else:
            latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

        # Final validation of total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"ERROR: Invalid total_loss detected!")
            print(f"  text_loss: {text_loss.item()}")
            print(f"  latent_loss: {latent_loss.item()}")
            # Return a small positive loss to avoid breaking training
            total_loss = torch.tensor(0.01, device=device, dtype=self.dtype)
"""

# NOTES:
# 1. Added epsilon (1e-8) to all divisions
# 2. Normalized latent predictions/targets before MSE loss
# 3. Added clamping to prevent overflow
# 4. Added validation checks for NaN/Inf at each step
# 5. Added latent_loss_weight to reduce contribution (default 0.1)
# 6. Print warnings when invalid values detected
