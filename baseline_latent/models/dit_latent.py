# File: hdlm/models/dit_latent.py
import math
import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
  import flash_attn
  import flash_attn.layers.rotary
  has_flash_attn = True
except ImportError:
  torch.backends.cuda.enable_flash_sdp(enabled=True)
  has_flash_attn = False

from .dit import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
    apply_rotary_pos_emb,
    Rotary,
    LayerNorm,
    TimestepEmbedder,
    EmbeddingLayer,
    DDitFinalLayer,
    DDiTBlockWithMask
)

class LatentConditionedDDiTBlock(DDiTBlockWithMask):
    """DDiT block with additional latent conditioning."""
    
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__(dim, n_heads, cond_dim, mlp_ratio, dropout)
        
        # Additional modulation for latent conditioning
        self.latent_adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.latent_adaLN_modulation.weight.data.zero_()
        self.latent_adaLN_modulation.bias.data.zero_()
    
    def forward(self, x, rotary_cos_sin, c, latent_cond=None, attention_mask=None, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # Combine timestep conditioning with latent conditioning
        if latent_cond is not None:
            latent_shift, latent_scale = latent_cond.chunk(2, dim=-1)
            # Get base modulation
            shift_msa_base, scale_msa_base, gate_msa_base, shift_mlp_base, scale_mlp_base, gate_mlp_base = \
                self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            
            # Get latent modulation
            latent_mod = self.latent_adaLN_modulation(c)[:, None]
            latent_shift_msa, latent_scale_msa, latent_gate_msa, latent_shift_mlp, latent_scale_mlp, latent_gate_mlp = \
                latent_mod.chunk(6, dim=2)
            
            # Combine: base + latent_scale * latent_modulation
            shift_msa = shift_msa_base + latent_scale[:, None] * latent_shift_msa
            scale_msa = scale_msa_base + latent_scale[:, None] * latent_scale_msa
            gate_msa = gate_msa_base + latent_scale[:, None] * latent_gate_msa
            
            shift_mlp = shift_mlp_base + latent_scale[:, None] * latent_shift_mlp
            scale_mlp = scale_mlp_base + latent_scale[:, None] * latent_scale_mlp
            gate_mlp = gate_mlp_base + latent_scale[:, None] * latent_gate_mlp
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp,
             scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # Attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv,
                        'b s (three h d) -> b s three h d',
                        three=3,
                        h=self.n_heads)
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(
            qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        
        if has_flash_attn and attention_mask is None:
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            if seqlens is None:
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seq_len, step=seq_len,
                    dtype=torch.int32, device=qkv.device)
            else:
                cu_seqlens = seqlens.cumsum(-1)
            x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
        else:
            q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
            
            if attention_mask is not None:
                attn_mask = attention_mask.bool()
                attn_mask = attn_mask.unsqueeze(1) & attn_mask.unsqueeze(2)
                attn_mask = attn_mask.unsqueeze(1)
                float_mask = torch.zeros(attn_mask.shape, device=q.device, dtype=q.dtype)
                float_mask = float_mask.masked_fill(~attn_mask, -1e9)
                
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
            else:
                x = F.scaled_dot_product_attention(q, k, v)
            
            x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x),
                                      None,
                                      gate_msa,
                                      x_skip,
                                      self.dropout)

        # MLP operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(
                self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)
        return x


class DITWithLatentConditioning(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, latent_dim: int = 768, cluster_size: int = 100):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.cluster_size = cluster_size
        self.rounded_vocab_size = vocab_size + cluster_size + (128 - (vocab_size + cluster_size) % 128) % 128

        # Text embedding
        self.vocab_embed = EmbeddingLayer(config.hidden_size, self.rounded_vocab_size)
        
        # Timestep embedding
        self.sigma_map = TimestepEmbedder(config.cond_dim)
        
        # Latent conditioning embedding (project latent to conditioning space)
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, config.cond_dim * 4),
            nn.SiLU(),
            nn.Linear(config.cond_dim * 4, config.cond_dim * 2)  # outputs (scale, shift)
        )
        
        self.rotary_emb = Rotary(
            config.hidden_size // config.n_heads,
            max_seq_len=config.max_seq_len,
        )

        # Use latent-conditioned blocks
        blocks = []
        for _ in range(config.n_blocks):
            blocks.append(LatentConditionedDDiTBlock(
                config.hidden_size,
                config.n_heads,
                config.cond_dim,
                dropout=config.dropout))
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.hidden_size,
            self.rounded_vocab_size,
            config.cond_dim)
        
        if cluster_size > 0:
            self.output_layer_clusters = DDitFinalLayer(
                config.hidden_size,
                self.rounded_vocab_size,
                config.cond_dim)
        else:
            self.output_layer_clusters = None
        
        self.register_buffer("logit_bias", torch.full((1, 1, 1), 0.0))

    def forward(self, indices, sigma, latents=None, attention_mask=None):
        # Convert sigma to float32 for timestep embedding
        sigma_float = sigma.float() if sigma.dtype != torch.float32 else sigma

        # Standard forward
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma_float))

        # Convert c back to x's dtype
        c = c.to(dtype=x.dtype)

        # Add latent conditioning
        latent_cond = None
        if latents is not None:
            if latents.dim() == 3:
                latents = latents.squeeze(1)
            latent_cond = self.latent_encoder(latents)

        rotary_cos_sin = self.rotary_emb(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](
                x, rotary_cos_sin, c, 
                latent_cond=latent_cond,
                attention_mask=attention_mask, 
                seqlens=None)

        x1 = self.output_layer(x, c)

        if self.cluster_size > 0:
            x2 = self.output_layer_clusters(x, c)
            return x1, x2

        return x1