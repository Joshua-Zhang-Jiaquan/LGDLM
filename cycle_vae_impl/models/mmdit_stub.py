import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDiTStub(nn.Module):
    """Minimal stub for MultimodalMMDiT.

    This exists so `cycle_vae_impl` can run smoke tests even when the external
    `mmdit` dependency is not installed.

    Forward signature matches the real model:
      forward(text_tokens, latents, text_timesteps, latent_timesteps, attention_mask=None)

    Returns:
      (text_logits, latent_pred)
    """

    def __init__(self, vocab_size: int, latent_dim: int, hidden_size: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.max_seq_len = int(max_seq_len)

        self.text_embed = nn.Embedding(self.vocab_size, self.hidden_size)
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_size)

        self.t_text = nn.Linear(1, self.hidden_size)
        self.t_lat = nn.Linear(1, self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=max(1, self.hidden_size // 64),
            dim_feedforward=self.hidden_size * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.text_head = nn.Linear(self.hidden_size, self.vocab_size)
        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.latent_dim),
        )

    def forward(self, text_tokens, latents, text_timesteps, latent_timesteps, attention_mask=None):
        # text_tokens: [B,S]
        x = self.text_embed(text_tokens)

        # time cond
        def _to_1(t, device, dtype):
            if t is None:
                return torch.zeros(text_tokens.shape[0], 1, device=device, dtype=dtype)
            if t.dim() == 1:
                return t.to(device=device, dtype=dtype).unsqueeze(1)
            return t.to(device=device, dtype=dtype).reshape(text_tokens.shape[0], 1)

        ttxt = _to_1(text_timesteps, x.device, x.dtype)
        tlat = _to_1(latent_timesteps, x.device, x.dtype)
        cond = self.t_text(ttxt) + self.t_lat(tlat)
        x = x + cond.unsqueeze(1)

        # latent conditioning: latents expected [B,1,D] or None
        if latents is not None:
            if latents.dim() == 3 and latents.shape[1] == 1:
                z = latents[:, 0]
            elif latents.dim() == 2:
                z = latents
            else:
                z = latents.mean(dim=1)
            z_emb = self.latent_proj(z).unsqueeze(1)
            x = x + z_emb

        if attention_mask is not None:
            # transformer expects True for tokens to attend; we provide src_key_padding_mask with True for PAD.
            key_padding = ~attention_mask.bool()
        else:
            key_padding = None

        h = self.encoder(x, src_key_padding_mask=key_padding)
        text_logits = self.text_head(h)

        pooled = h.mean(dim=1)
        latent_pred = self.latent_head(pooled)
        return text_logits, latent_pred
