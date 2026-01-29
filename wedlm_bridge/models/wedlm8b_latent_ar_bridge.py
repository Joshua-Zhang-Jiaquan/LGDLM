from __future__ import annotations

import torch
import torch.nn as nn

from wedlm_bridge.pretrained_utils import resolve_pretrained_path


class LatentPrefixProjector(nn.Module):
    def __init__(self, latent_dim: int, hidden_size: int, prefix_len: int):
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * self.prefix_len),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 3:
            if latents.shape[1] == 1:
                latents = latents.squeeze(1)
            else:
                latents = latents.mean(dim=1)
        prefix = self.net(latents)
        return prefix.view(latents.shape[0], self.prefix_len, self.hidden_size)


class TextToLatentHead(nn.Module):
    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, latent_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class WeDLM8BLatentARBridge(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()

        try:
            from transformers import AutoModelForCausalLM
        except Exception as e:  # pragma: no cover
            raise ImportError("transformers is required for WeDLM8BLatentARBridge.") from e

        pretrained_name = getattr(config, "pretrained_model_name_or_path", None)
        if not pretrained_name:
            raise ValueError("`model.pretrained_model_name_or_path` must be set.")

        trust_remote_code = bool(getattr(config, "trust_remote_code", True))
        torch_dtype = getattr(config, "pretrained_model_dtype", None)
        if isinstance(torch_dtype, str):
            torch_dtype = {
                "fp32": torch.float32,
                "float32": torch.float32,
                "fp16": torch.float16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
            }.get(torch_dtype.lower())

        model_path, local_files_only = resolve_pretrained_path(config, pretrained_name)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        self.lm.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer

        self.base_model = getattr(self.lm, "model", None) or getattr(self.lm, "transformer", None)
        if self.base_model is None:
            raise RuntimeError("Unsupported HF causal LM: expected `.model` or `.transformer` attribute.")

        hidden_size = getattr(self.lm.config, "hidden_size", None) or getattr(self.lm.config, "n_embd", None)
        if hidden_size is None:
            raise RuntimeError("Could not infer hidden size from HF config.")

        latent_dim = int(getattr(config, "latent_dim", 1024))
        prefix_len = int(getattr(config, "latent_prefix_len", 8))
        self.latent_pooling = str(getattr(config, "latent_pooling", "last")).lower()
        if self.latent_pooling not in {"last", "mean"}:
            raise ValueError("`model.latent_pooling` must be one of: last, mean")

        self.latent_prefix = LatentPrefixProjector(latent_dim=latent_dim, hidden_size=hidden_size, prefix_len=prefix_len)
        self.text_to_latent = TextToLatentHead(hidden_size=hidden_size, latent_dim=latent_dim)

        self.text_head = self.lm.get_output_embeddings()

        if bool(getattr(config, "freeze_backbone", False)):
            for param in self.base_model.parameters():
                param.requires_grad = False
        if bool(getattr(config, "freeze_lm_head", False)) and self.text_head is not None:
            for param in self.text_head.parameters():
                param.requires_grad = False

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.latent_pooling == "mean":
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (hidden * mask).sum(dim=1) / denom

        lengths = attention_mask.long().sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_index = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_index, lengths, :]

    @staticmethod
    def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return position_ids

    def forward(
        self,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        mode: str = "l2t",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        device = text_tokens.device
        if attention_mask is None:
            attention_mask = torch.ones_like(text_tokens, dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)

        if mode == "l2t":
            if latents is None:
                raise ValueError("`latents` must be provided when mode='l2t'.")
            prefix_embeds = self.latent_prefix(latents.to(device=device, dtype=self.lm.get_input_embeddings().weight.dtype))
            text_embeds = self.lm.get_input_embeddings()(text_tokens)

            seq_embeds = torch.cat([prefix_embeds.to(dtype=text_embeds.dtype), text_embeds], dim=1)
            prefix_mask = torch.ones((attention_mask.shape[0], prefix_embeds.shape[1]), device=device, dtype=attention_mask.dtype)
            seq_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            position_ids = self._build_position_ids(seq_mask)
            base_out = self.base_model(
                inputs_embeds=seq_embeds,
                attention_mask=seq_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden = base_out.last_hidden_state

            out_proj = self.text_head or self.lm.get_output_embeddings()
            if out_proj is None and hasattr(self.lm, "lm_head"):
                out_proj = self.lm.lm_head  # type: ignore[attr-defined]
            if out_proj is None:
                raise RuntimeError("No LM head found for text logits.")
            logits = out_proj(hidden)

            prefix_len = prefix_embeds.shape[1]
            text_logits = logits[:, prefix_len:, :]
            return text_logits, None

        if mode == "t2l":
            position_ids = self._build_position_ids(attention_mask)
            base_out = self.base_model(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden = base_out.last_hidden_state
            pooled = self._pool_hidden(hidden, attention_mask)
            latent_pred = self.text_to_latent(pooled)
            return None, latent_pred

        raise ValueError(f"Unknown mode: {mode}. Expected 'l2t' or 't2l'.")

