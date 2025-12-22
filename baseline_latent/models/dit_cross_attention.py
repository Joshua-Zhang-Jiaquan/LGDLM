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
class CrossAttentionDDiTBlock(nn.Module):
    """DDiT block with cross-attention to latents."""
    
    def __init__(self, dim, n_heads, cond_dim, latent_dim=768, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.cond_dim = cond_dim
        
        # Self-attention (same as DDiTBlock)
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        
        # CROSS-ATTENTION to latents
        self.cross_attn_norm = LayerNorm(dim)
        self.cross_attn_q = nn.Linear(dim, dim, bias=False)  # Query from text
        self.cross_attn_kv = nn.Linear(latent_dim, 2 * dim, bias=False)  # Key/Value from latents
        self.cross_attn_out = nn.Linear(dim, dim, bias=False)
        self.cross_dropout = nn.Dropout(dropout)
        
        # MLP
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        
        # AdaLN modulation (timestep only)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
    
    def forward(self, x, rotary_cos_sin, c, latents=None, latent_mask=None, attention_mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Get AdaLN modulation
        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        
        # ===== SELF-ATTENTION (text → text) =====
        x_skip = x
        x_norm = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        
        qkv = self.attn_qkv(x_norm)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        
        # Self-attention
        if has_flash_attn and attention_mask is None:
            qkv = rearrange(qkv, 'b s ... -> (b s) ...')
            cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len,
                                      dtype=torch.int32, device=qkv.device)
            attn_out = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, seq_len, 0., causal=False)
            attn_out = rearrange(attn_out, '(b s) h d -> b s (h d)', b=batch_size)
        else:
            q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
            
            if attention_mask is not None:
                attn_mask = attention_mask.bool()
                attn_mask = attn_mask.unsqueeze(1) & attn_mask.unsqueeze(2)
                attn_mask = attn_mask.unsqueeze(1)
                float_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                float_mask = float_mask.masked_fill(~attn_mask, -1e9)
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
            else:
                attn_out = F.scaled_dot_product_attention(q, k, v)
            
            attn_out = rearrange(attn_out, 'b h s d -> b s (h d)', b=batch_size)
        
        x = x_skip + gate_msa * self.dropout1(self.attn_out(attn_out))
        
        # ===== CROSS-ATTENTION (text → latents) =====
        if latents is not None:
            x_cross_skip = x
            
            # Normalize
            x_cross_norm = modulate_fused(self.cross_attn_norm(x), shift_msa, scale_msa)
            
            # Query from text, Key/Value from latents
            q_cross = self.cross_attn_q(x_cross_norm)  # [batch, seq_len, dim]
            q_cross = rearrange(q_cross, 'b s (h d) -> b h s d', h=self.n_heads)
            
            # Project latents to key/value
            kv_cross = self.cross_attn_kv(latents)  # [batch, latent_seq_len, 2*dim]
            k_cross, v_cross = kv_cross.chunk(2, dim=-1)
            k_cross = rearrange(k_cross, 'b l (h d) -> b h l d', h=self.n_heads)
            v_cross = rearrange(v_cross, 'b l (h d) -> b h l d', h=self.n_heads)
            
            # Cross-attention
            if latent_mask is not None:
                latent_mask_expanded = latent_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, latent_seq]
                float_latent_mask = torch.zeros_like(latent_mask_expanded, dtype=q_cross.dtype)
                float_latent_mask = float_latent_mask.masked_fill(~latent_mask_expanded, -1e9)
                cross_attn_out = F.scaled_dot_product_attention(
                    q_cross, k_cross, v_cross, attn_mask=float_latent_mask)
            else:
                cross_attn_out = F.scaled_dot_product_attention(q_cross, k_cross, v_cross)
            
            cross_attn_out = rearrange(cross_attn_out, 'b h s d -> b s (h d)')
            x = x_cross_skip + gate_msa * self.cross_dropout(self.cross_attn_out(cross_attn_out))
        
        # ===== MLP =====
        x_mlp_skip = x
        x = modulate_fused(self.norm2(x), shift_mlp, scale_mlp)
        x = self.mlp(x)
        x = x_mlp_skip + gate_mlp * self.dropout2(x)
        
        return x
class DITWithCrossAttention(nn.Module, huggingface_hub.PyTorchModelHubMixin):
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
        
        # Latent projection (for cross-attention keys/values)
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        self.rotary_emb = Rotary(
            config.hidden_size // config.n_heads,
            max_seq_len=config.max_seq_len,
        )

        # Use cross-attention blocks (every other block)
        blocks = []
        for i in range(config.n_blocks):
            # Alternate between regular and cross-attention blocks
            if i % 2 == 0:  # Even blocks: cross-attention
                blocks.append(CrossAttentionDDiTBlock(
                    config.hidden_size,
                    config.n_heads,
                    config.cond_dim,
                    latent_dim=config.hidden_size,  # Latents projected to hidden_size
                    dropout=config.dropout))
            else:  # Odd blocks: regular DDiT (no cross-attention)
                from .dit import DDiTBlockWithMask
                blocks.append(DDiTBlockWithMask(
                    config.hidden_size,
                    config.n_heads,
                    config.cond_dim,
                    dropout=config.dropout))
        
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.hidden_size,
            self.rounded_vocab_size,
            config.cond_dim)
        
        # Optional: Add latent prediction head
        self.latent_prediction_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, latent_dim)
        )
        
        if cluster_size > 0:
            self.output_layer_clusters = DDitFinalLayer(
                config.hidden_size,
                self.rounded_vocab_size,
                config.cond_dim)
        else:
            self.output_layer_clusters = None
        
        self.register_buffer("logit_bias", torch.full((1, 1, 1), 0.0))
    
    def forward(self, indices, sigma, latents=None, latent_mask=None, attention_mask=None, return_latent_pred=False):
        # Standard forward
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma.float())).to(x.dtype)
        
        # Project latents if provided
        projected_latents = None
        if latents is not None:
            # latents: [batch, latent_seq_len, latent_dim]
            if latents.dim() == 3 and latents.shape[1] == 1:
                # Expand single latent to sequence
                projected_latents = self.latent_projection(latents)
            else:
                projected_latents = self.latent_projection(latents)
        
        rotary_cos_sin = self.rotary_emb(x)
        
        # Store latent predictions
        latent_preds = []
        
        for i, block in enumerate(self.blocks):
            if hasattr(block, 'cross_attn_q') and projected_latents is not None:
                # Cross-attention block
                x = block(
                    x, rotary_cos_sin, c, 
                    latents=projected_latents,
                    latent_mask=latent_mask,
                    attention_mask=attention_mask)
            else:
                # Regular DDiT block
                x = block(x, rotary_cos_sin, c, attention_mask=attention_mask)
            
            # Collect intermediate representations for latent prediction
            if return_latent_pred:
                latent_preds.append(x.mean(dim=1))  # [batch, hidden_size]
        
        # Text output
        x1 = self.output_layer(x, c)
        
        # Latent prediction (optional)
        latent_pred = None
        if return_latent_pred and latent_preds:
            # Combine predictions from all layers
            combined = torch.stack(latent_preds, dim=1).mean(dim=1)  # [batch, hidden_size]
            latent_pred = self.latent_prediction_head(combined)  # [batch, latent_dim]
        
        if self.cluster_size > 0:
            x2 = self.output_layer_clusters(x, c)
            if return_latent_pred:
                return x1, x2, latent_pred
            return x1, x2
        
        if return_latent_pred:
            return x1, latent_pred
        return x1