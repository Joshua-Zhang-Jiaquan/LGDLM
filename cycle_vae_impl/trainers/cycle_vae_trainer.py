import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from latentDLM_mmdit.continuous_diffusion import ContinuousDiffusion


class CycleVAETrainer(nn.Module):
    """Isolated trainer that adds a cycle_text (T2L2T) auxiliary loss.

    This trainer lives outside `latentDLM_mmdit/` and should not require modifying the
    original training scripts.
    """

    def __init__(self, model, tokenizer, text_noise_schedule, dtype: torch.dtype, config):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.text_noise_schedule = text_noise_schedule
        self.dtype = dtype
        self.config = config

        self.mask_token_id = tokenizer.mask_token_id

        self.latent_diffusion = ContinuousDiffusion(
            num_timesteps=config.model.get("latent_timesteps", 1000),
            beta_schedule=config.model.get("latent_beta_schedule", "cosine"),
            parameterization=config.model.get("latent_parameterization", "epsilon"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def _sample_latent_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        mode = str(getattr(self.config.loss, "latent_t_mode", "random")).lower()
        if mode in ("full", "ones", "t1"):
            return torch.ones(batch_size, device=device)
        if mode != "random":
            raise ValueError(f"Unknown loss.latent_t_mode: {mode}")

        t_min = float(getattr(self.config.loss, "cycle_latent_t_min", 0.0))
        t_max = float(getattr(self.config.loss, "cycle_latent_t_max", 0.98))
        if not (0.0 <= t_min < t_max <= 1.0):
            raise ValueError(f"Invalid latent t range: [{t_min}, {t_max}]")

        t = t_min + (t_max - t_min) * torch.rand(batch_size, device=device)
        return torch.clamp(t, max=0.9999)

    def _ensure_latents_2d(self, latents):
        if latents is None or (not torch.is_tensor(latents)):
            raise ValueError("Batch is missing 'latent' tensor")
        if latents.dim() == 3 and latents.shape[1] == 1:
            latents = latents.squeeze(1)
        if latents.dim() != 2:
            raise ValueError(f"Expected latents [B,D] or [B,1,D], got shape {tuple(latents.shape)}")
        return latents

    def _text_ce_loss(self, text_logits: torch.Tensor, text_target: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        vocab_size = text_logits.shape[-1]
        loss_unreduced = F.cross_entropy(
            text_logits.reshape(-1, vocab_size),
            text_target.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        mask = text_mask.reshape(-1).to(dtype=loss_unreduced.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (loss_unreduced * mask).sum() / denom

    def _ramp_weight(self, step, base: float, warmup_steps: int, ramp_steps: int) -> float:
        if step is None:
            return base
        if base <= 0.0:
            return 0.0
        warmup_steps = int(max(0, warmup_steps))
        ramp_steps = int(max(0, ramp_steps))
        s = int(step)
        if s < warmup_steps:
            return 0.0
        if ramp_steps <= 0:
            return base
        p = min(1.0, (s - warmup_steps) / float(ramp_steps))
        return base * p

    def _sample_tokens_from_logits(self, logits: torch.Tensor, sampling: str) -> torch.Tensor:
        # logits: [B,S,V]
        sampling = str(sampling).lower()
        if sampling == "argmax":
            return torch.argmax(logits, dim=-1)
        if sampling == "categorical":
            probs = torch.softmax(logits, dim=-1)
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_flat = torch.multinomial(probs_flat, 1)
            return sampled_flat.view(logits.shape[0], logits.shape[1])
        raise ValueError(f"Unknown cycle_latent_sampling: {sampling}")

    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        if torch.isnan(tensor).any():
            print(f"WARNING: NaN detected in {name}")
            return False
        if torch.isinf(tensor).any():
            print(f"WARNING: INF detected in {name}")
            return False
        return True

    def forward(self, batch: dict, step=None):
        if getattr(self.config.loss, "loss_type", "cycle_vae") != "cycle_vae":
            raise ValueError("CycleVAETrainer requires loss.loss_type=cycle_vae")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        latents = batch.get("latent", None)

        device = input_ids.device
        batch_size = input_ids.shape[0]

        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        latents_gt = self._ensure_latents_2d(latents).to(device=device, dtype=self.dtype)
        
        if not self._validate_tensor(latents_gt, "input_latents"):
            return torch.tensor(0.01, device=device, dtype=self.dtype), {
                "loss": 0.01,
                "error": "invalid_input_latents"
            }
        
        t_lat = self._sample_latent_t(batch_size, device=device).to(dtype=self.dtype)

        # Base T2L latent diffusion loss (random t)
        noise = torch.randn_like(latents_gt)
        z_t, latent_target = self.latent_diffusion.add_noise(latents_gt, t_lat, noise)

        text_timesteps_clean = torch.zeros(batch_size, device=device, dtype=self.dtype)

        outputs = self.model(
            text_tokens=input_ids,
            latents=z_t.unsqueeze(1),
            text_timesteps=text_timesteps_clean,
            latent_timesteps=t_lat,
            attention_mask=attention_mask,
        )

        if isinstance(outputs, tuple) and len(outputs) >= 2:
            text_logits_t2l = outputs[0]
            latent_pred = outputs[1]
        else:
            raise ValueError("Model did not return (text_logits, latent_pred)")

        # Normalize MSE like stable trainer
        eps = 1e-8
        
        if not self._validate_tensor(latent_pred, "latent_pred"):
            base_t2l_latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        else:
            latent_pred_norm = F.normalize(latent_pred, p=2, dim=-1, eps=eps)
            latent_target_norm = F.normalize(latent_target, p=2, dim=-1, eps=eps)
            base_t2l_latent_loss = F.mse_loss(latent_pred_norm, latent_target_norm)
            base_t2l_latent_loss = torch.clamp(base_t2l_latent_loss, min=0.0, max=10.0)
            
            if not self._validate_tensor(base_t2l_latent_loss, "base_t2l_latent_loss"):
                base_t2l_latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        latent_loss_weight = float(getattr(self.config.loss, "latent_loss_weight", 0.1))
        base_t2l_latent_loss_weighted = base_t2l_latent_loss * latent_loss_weight

        # Base L2T text loss (full mask, condition on GT latents)
        full_mask_text = torch.full_like(input_ids, self.mask_token_id)
        text_timesteps_full = torch.ones(batch_size, device=device, dtype=self.dtype)
        latent_timesteps_clean = torch.zeros(batch_size, device=device, dtype=self.dtype)

        outputs_l2t = self.model(
            text_tokens=full_mask_text,
            latents=latents_gt.unsqueeze(1),
            text_timesteps=text_timesteps_full,
            latent_timesteps=latent_timesteps_clean,
            attention_mask=attention_mask,
        )

        if isinstance(outputs_l2t, tuple):
            text_logits_l2t = outputs_l2t[0]
        else:
            text_logits_l2t = outputs_l2t

        text_mask_all = torch.ones_like(input_ids, dtype=torch.bool)
        base_l2t_text_loss = self._text_ce_loss(text_logits_l2t, input_ids, text_mask_all)
        base_l2t_text_loss = torch.clamp(base_l2t_text_loss, min=0.0, max=100.0)

        # Cycle-text (T2L2T): predict x0 from T2L prediction, then L2T conditioned on predicted x0
        param = self.latent_diffusion.parameterization
        if param == "epsilon":
            x0_pred = self.latent_diffusion.predict_x0_from_eps(z_t, t_lat, latent_pred)
        elif param == "x0":
            x0_pred = latent_pred
        elif param == "v_param":
            x0_pred = self.latent_diffusion.predict_x0_from_v(z_t, t_lat, latent_pred)
        else:
            raise ValueError(f"Unknown latent parameterization: {param}")

        if bool(getattr(self.config.loss, "cycle_stop_grad_latent", True)):
            x0_cycle = x0_pred.detach()
        else:
            x0_cycle = x0_pred

        outputs_cycle = self.model(
            text_tokens=full_mask_text,
            latents=x0_cycle.to(dtype=self.dtype).unsqueeze(1),
            text_timesteps=text_timesteps_full,
            latent_timesteps=latent_timesteps_clean,
            attention_mask=attention_mask,
        )

        if isinstance(outputs_cycle, tuple):
            text_logits_cycle = outputs_cycle[0]
        else:
            text_logits_cycle = outputs_cycle

        cycle_text_loss = self._text_ce_loss(text_logits_cycle, input_ids, text_mask_all)
        cycle_text_loss = torch.clamp(cycle_text_loss, min=0.0, max=100.0)
        
        if not self._validate_tensor(cycle_text_loss, "cycle_text_loss"):
            cycle_text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

        cycle_text_weight_base = float(getattr(self.config.loss, "cycle_text_weight", 1.0))
        cycle_text_warmup = int(getattr(self.config.loss, "cycle_text_warmup_steps", 0))
        cycle_text_ramp = int(getattr(self.config.loss, "cycle_text_ramp_steps", 0))
        cycle_text_weight = self._ramp_weight(step, cycle_text_weight_base, cycle_text_warmup, cycle_text_ramp)

        # Cycle-latent (L2T2L): pseudo-label tokens from L2T, then T2L loss conditioned on those tokens
        cycle_latent_weight_base = float(getattr(self.config.loss, "cycle_latent_weight", 0.0))
        cycle_latent_warmup = int(getattr(self.config.loss, "cycle_latent_warmup_steps", 0))
        cycle_latent_ramp = int(getattr(self.config.loss, "cycle_latent_ramp_steps", 0))
        cycle_latent_weight = self._ramp_weight(step, cycle_latent_weight_base, cycle_latent_warmup, cycle_latent_ramp)

        cycle_latent_loss = torch.zeros((), device=device, dtype=self.dtype)
        cycle_latent_sampling = str(getattr(self.config.loss, "cycle_latent_sampling", "argmax"))

        if cycle_latent_weight > 0.0:
            with torch.no_grad():
                pseudo_tokens = self._sample_tokens_from_logits(text_logits_l2t, cycle_latent_sampling)

            t_lat2 = self._sample_latent_t(batch_size, device=device).to(dtype=self.dtype)
            noise2 = torch.randn_like(latents_gt)
            z_t2, latent_target2 = self.latent_diffusion.add_noise(latents_gt, t_lat2, noise2)

            outputs_t2l_cycle = self.model(
                text_tokens=pseudo_tokens,
                latents=z_t2.unsqueeze(1),
                text_timesteps=text_timesteps_clean,
                latent_timesteps=t_lat2,
                attention_mask=attention_mask,
            )
            if isinstance(outputs_t2l_cycle, tuple) and len(outputs_t2l_cycle) >= 2:
                latent_pred2 = outputs_t2l_cycle[1]
            else:
                raise ValueError("Model did not return (text_logits, latent_pred) for cycle_latent")

            if not self._validate_tensor(latent_pred2, "latent_pred2"):
                cycle_latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
            else:
                latent_pred2_norm = F.normalize(latent_pred2, p=2, dim=-1, eps=eps)
                latent_target2_norm = F.normalize(latent_target2, p=2, dim=-1, eps=eps)
                cycle_latent_loss = F.mse_loss(latent_pred2_norm, latent_target2_norm)
                cycle_latent_loss = torch.clamp(cycle_latent_loss, min=0.0, max=10.0)
                
                if not self._validate_tensor(cycle_latent_loss, "cycle_latent_loss"):
                    cycle_latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

        total_loss = (
            base_t2l_latent_loss_weighted
            + base_l2t_text_loss
            + float(cycle_text_weight) * cycle_text_loss
            + float(cycle_latent_weight) * cycle_latent_loss
        )
        
        if not self._validate_tensor(total_loss, "total_loss"):
            print(f"ERROR: Invalid total_loss detected, using fallback")
            total_loss = torch.tensor(0.01, device=device, dtype=self.dtype)

        # Basic metrics
        with torch.no_grad():
            metrics = {
                "loss": float(total_loss.item()),
                "base_t2l_latent_loss": float(base_t2l_latent_loss.item()),
                "base_l2t_text_loss": float(base_l2t_text_loss.item()),
                "cycle_text_loss": float(cycle_text_loss.item()),
                "cycle_text_weight": float(cycle_text_weight),
                "cycle_latent_loss": float(cycle_latent_loss.item()),
                "cycle_latent_weight": float(cycle_latent_weight),
                "cycle_latent_sampling": cycle_latent_sampling,
                "t_lat_mean": float(t_lat.mean().item()),
                "latent_pred_norm": float(latent_pred.norm(dim=-1).mean().item()),
                "x0_pred_norm": float(x0_pred.norm(dim=-1).mean().item()),
            }

        return total_loss, metrics
