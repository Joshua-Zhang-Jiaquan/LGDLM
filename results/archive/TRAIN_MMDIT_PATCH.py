"""
Patch file for train_mmdit.py to fix NaN gradient issues.

Apply these changes to latentDLM_mmdit/train_mmdit.py around lines 420-437.
"""

# ORIGINAL CODE (lines 420-437):
"""
            (loss * config.loss.loss_scale).backward()

            # Grad clip
            if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
            else:
                norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1e6)

            if torch.isnan(norm):
                print(f"Warning: NaN gradient detected at step {state.step}")
                for param in trainer.parameters():
                    if param.grad is not None:
                        param.grad.data.zero_()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
"""

# IMPROVED CODE (replace lines 420-437 with this):
"""
            # ===== LOSS VALIDATION =====
            # Check for invalid loss BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\\n{'='*60}")
                print(f"ERROR: Invalid loss detected at step {state.step}")
                print(f"{'='*60}")
                print(f"  Total loss: {loss.item()}")
                print(f"  Text loss: {metrics.get('text_loss', 0.0):.6f}")
                print(f"  Latent loss: {metrics.get('latent_loss', 0.0):.6f}")
                print(f"  Mode: {metrics.get('mode', 'unknown')}")
                print(f"  Learning rate: {curr_lr:.2e}")
                print(f"  Skipping this batch...")
                print(f"{'='*60}\\n")

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
                    error_if_nonfinite=False  # Don't raise error, we'll handle it
                )
            else:
                # Default to 1.0, not 1e6 (which is effectively no clipping)
                norm = torch.nn.utils.clip_grad_norm_(
                    trainer.parameters(),
                    1.0,
                    error_if_nonfinite=False
                )

            # ===== NaN/INF GRADIENT CHECK =====
            # Check AFTER clipping
            if torch.isnan(norm) or torch.isinf(norm):
                print(f"\\n{'='*60}")
                print(f"ERROR: Invalid gradient norm at step {state.step}")
                print(f"{'='*60}")
                print(f"  Gradient norm: {norm.item() if not torch.isnan(norm) else 'NaN'}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Text loss: {metrics.get('text_loss', 0.0):.6f}")
                print(f"  Latent loss: {metrics.get('latent_loss', 0.0):.6f}")
                print(f"  Learning rate: {curr_lr:.2e}")

                # Count parameters with NaN gradients
                nan_count = 0
                total_params = 0
                for name, param in trainer.named_parameters():
                    if param.grad is not None:
                        total_params += 1
                        if torch.isnan(param.grad).any():
                            nan_count += 1
                            if nan_count <= 5:  # Print first 5
                                print(f"  NaN in: {name}")

                print(f"  Total params with NaN gradients: {nan_count}/{total_params}")
                print(f"  Skipping optimizer step and resetting gradients...")
                print(f"{'='*60}\\n")

                # Reset gradients and skip optimizer step
                optimizer.zero_grad(set_to_none=True)
                state.step += 1
                pbar.update(1)
                continue

            # ===== OPTIMIZER STEP =====
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
"""

# ADDITIONAL: Add gradient norm monitoring to logging (around line 461)
"""
            log_buffer.append(
                {
                    "train/loss": float(loss.item()),
                    "train/lr": float(curr_lr),
                    "train/step": int(state.step + 1),
                    "train/grad_norm": float(norm.item()),
                    "train/text_loss": float(metrics.get('text_loss', 0.0)),  # ADD THIS
                    "train/latent_loss": float(metrics.get('latent_loss', 0.0)),  # ADD THIS
                    "train/epoch": float(state.epoch + (state.step - state.epoch_start_step) / total_batches),
                    "train/total_tokens": float(state.total_tokens),
                    "train/total_flops": float(state.total_flops),
                    "train/tokens_per_sec": float(batch_tokens / step_time),
                    "train/flops_per_sec": float(batch_flops / step_time),
                    "train/samples_per_sec": float(total_batch_size / step_time),
                    "train/it_per_sec": float(1.0 / step_time),
                    "train/avg_it_per_sec": float((state.step + 1) / (curr_time - state.start_time))
                }
            )
"""
