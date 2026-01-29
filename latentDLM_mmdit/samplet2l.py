# File: latentDLM_mmdit/sample_t2l_fixed.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import sys
import os
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train_mmdit import ContinuousDiffusion
from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
from latentDLM_mmdit.modeling_mmdit import get_tokenizer

class FixedT2LSampler:
    def __init__(self, checkpoint_path, config_path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif 'config' in checkpoint and checkpoint['config'] is not None:
            config = checkpoint['config']
            print("Using config from checkpoint")
        else:
            config = {
                'model': {
                    'hidden_size': 1024,
                    'n_blocks': 24,
                    'n_heads': 24,
                    'cond_dim': 1024,
                    'max_seq_len': 4096,  # Changed from 1024 to 4096
                    'dropout': 0.1,
                    'num_residual_streams': 2,
                    'qk_rmsnorm': True,
                    'use_multimodal': True,
                    'latent_dim': 32,  # Changed from 1024 to 32
                }
            }
            print("Using default config")
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.tokenizer_vocab_size = len(self.tokenizer)
        print(f"Tokenizer vocab size: {self.tokenizer_vocab_size}")
        
        # 关键：使用检查点的词汇量，而不是配置中的
        # 首先检查检查点中是否有vocab_size信息
        if 'model_state_dict' in checkpoint:
            # 从text_head.weight的形状推断词汇量
            for key in checkpoint['model_state_dict']:
                if 'text_head.weight' in key:
                    vocab_size_from_checkpoint = checkpoint['model_state_dict'][key].shape[0]
                    print(f"Inferred vocab size from checkpoint: {vocab_size_from_checkpoint}")
                    # 更新config中的vocab_size
                    config['model']['vocab_size'] = vocab_size_from_checkpoint
                    break
        
        model_vocab_size = config['model'].get('vocab_size', 30522)  # Default to 30522
        print(f"Model vocab size: {model_vocab_size}")
        
        if model_vocab_size != self.tokenizer_vocab_size:
            print(f"Warning: Model vocab size ({model_vocab_size}) != Tokenizer vocab size ({self.tokenizer_vocab_size})")
            print(f"Difference: {model_vocab_size - self.tokenizer_vocab_size} tokens")
        
        # Create model - 使用模型配置的词汇量
        latent_dim = config['model'].get('latent_dim', 32)  # Default to 32 based on error
        print(f"Creating model with latent_dim={latent_dim}, vocab_size={model_vocab_size}")
        
        # 创建模型时使用检查点的参数
        self.model = MultimodalMMDiT(
            config=config['model'],
            vocab_size=model_vocab_size,  # 使用检查点的词汇量
            latent_dim=latent_dim,
            cluster_size=0
        ).to(device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            # 尝试加载状态字典
            try:
                self.model.load_state_dict(new_state_dict, strict=True)
                print("Model loaded successfully with strict=True")
            except RuntimeError as e:
                print(f"Strict loading failed: {e}")
                print("Trying non-strict loading...")
                missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
                print(f"Model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
                if missing:
                    print(f"Missing keys: {missing[:10]}")
                if unexpected:
                    print(f"Unexpected keys: {unexpected[:10]}")
        
        self.model.eval()
        self.model_vocab_size = model_vocab_size
        self.latent_dim = latent_dim
        
        # 检查模型权重是否有 NaN
        nan_count = 0
        inf_count = 0
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"WARNING: NaN found in parameter: {name}")
                nan_count += 1
            if torch.isinf(param).any():
                print(f"WARNING: Inf found in parameter: {name}")
                inf_count += 1
        if nan_count > 0 or inf_count > 0:
            print(f"Total: {nan_count} params with NaN, {inf_count} params with Inf")
        else:
            print("Model weights check passed: no NaN or Inf found")
        
        # 打印模型信息
        print(f"\nModel initialized with:")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Max seq len: {config['model'].get('max_seq_len', 4096)}")
        print(f"  Vocab size: {self.model_vocab_size}")
        
    @torch.no_grad()
    def generate(self, text_tokens, attention_mask=None, steps=100, guidance_scale=1.0):
        """Generate latents from text tokens using diffusion"""
        batch_size = text_tokens.shape[0]
        text_tokens = text_tokens.to(self.device)
        
        if attention_mask is None:
            attention_mask = (text_tokens != self.tokenizer.pad_token_id).long().to(self.device)
        
        # Initialize latents with noise - shape: [batch_size, 1, latent_dim]
        latents = torch.randn(batch_size, 1, self.latent_dim, device=self.device)
        
        # Timesteps for diffusion process
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)[:-1]
        
        # Text is at timestep 0 (fully known)
        text_timesteps = torch.zeros(batch_size, device=self.device)
        
        # 初始检查
        print(f"Initial latents: mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
        
        for i in tqdm(range(steps), desc="Generating latents"):
            t = timesteps[i].expand(batch_size)
            
            # 检查输入是否有 NaN
            if torch.isnan(latents).any():
                print(f"WARNING: Input latents contain NaN at step {i}, replacing with zeros")
                latents = torch.nan_to_num(latents, nan=0.0)
            
            # Forward pass - get latent predictions
            _, latent_pred = self.model(
                text_tokens=text_tokens,
                latents=latents,
                text_timesteps=text_timesteps,
                latent_timesteps=t,
                attention_mask=attention_mask,
            )
            
            # 调试：检查形状和 NaN
            if i == 0:
                print(f"\nlatent_pred shape: {latent_pred.shape}")
                print(f"Expected: [batch={batch_size}, seq=1, latent_dim={self.latent_dim}]")
            
            # NaN 检测
            if torch.isnan(latent_pred).any():
                print(f"WARNING: NaN detected in latent_pred at step {i}")
                print(f"  latent_pred stats: min={latent_pred.min()}, max={latent_pred.max()}")
                print(f"  latents stats: min={latents.min()}, max={latents.max()}")
                print(f"  timestep t={t[0].item():.4f}")
                # 尝试用零替换 NaN 以继续调试
                latent_pred = torch.nan_to_num(latent_pred, nan=0.0)
            
            # Apply guidance scale (CFG) if > 1.0
            if guidance_scale > 1.0:
                # Get unconditional prediction (conditioned on empty text)
                empty_text = torch.full_like(text_tokens, self.tokenizer.pad_token_id)
                empty_mask = torch.zeros_like(attention_mask)
                _, latent_pred_uncond = self.model(
                    text_tokens=empty_text,
                    latents=latents,
                    text_timesteps=text_timesteps,
                    latent_timesteps=t,
                    attention_mask=empty_mask,
                )
                
                # Classifier-free guidance
                latent_pred = latent_pred_uncond + guidance_scale * (latent_pred - latent_pred_uncond)
            
            # DDPM update rule
            if i < steps - 1:
                # Get next timestep
                next_t = timesteps[i + 1]
                
                # Simple noise schedule (cosine or linear)
                # Using linear schedule for simplicity
                alpha = 1.0 - t
                alpha_next = 1.0 - next_t
                
                # Add noise for next step
                noise = torch.randn_like(latent_pred)
                
                # DDPM update: 使用更稳定的 noise scale 计算
                # 避免 alpha=0 时的除零问题
                eps = 1e-8
                alpha_clamped = torch.clamp(alpha, min=eps)
                
                # 计算 beta (噪声比例)
                beta = 1.0 - alpha_next / alpha_clamped
                beta = torch.clamp(beta, 0.0, 1.0)
                noise_scale = torch.sqrt(beta * alpha_next)
                noise_scale = torch.clamp(noise_scale, 0.0, 1.0)
                
                latents = latent_pred + noise_scale.unsqueeze(1).unsqueeze(2) * noise
            else:
                # Final step: use prediction directly
                latents = latent_pred
            
            # NaN 检测（更新后）
            if torch.isnan(latents).any():
                print(f"WARNING: NaN detected in latents after update at step {i}")
                latents = torch.nan_to_num(latents, nan=0.0)
            
            # 显示进度
            if i % 10 == 0:
                latents_norm = latents.norm(dim=-1).mean().item()
                pred_norm = latent_pred.norm(dim=-1).mean().item()
                print(f"Step {i+1}/{steps}: latents_norm={latents_norm:.3f}, pred_norm={pred_norm:.3f}")
        
        return latents.squeeze(1)  # Remove sequence dimension: [batch_size, latent_dim]
    
    def encode_text(self, texts, max_length=128):
        """Encode text to tokens"""
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)
    
    def load_texts(self, txt_file, num_samples=None):
        """Load texts from file"""
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if num_samples and len(lines) > num_samples:
            import random
            lines = random.sample(lines, num_samples)
        elif num_samples:
            lines = lines[:num_samples]
        
        return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--text_file", required=True, help="File containing text prompts (one per line)")
    parser.add_argument("--text_prompt", default=None, help="Single text prompt (overrides text_file)")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG guidance scale")
    parser.add_argument("--output_dir", default="./t2l_fixed_output")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum text length")
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = FixedT2LSampler(args.checkpoint, args.config)
    
    # Load or create texts
    if args.text_prompt:
        texts = [args.text_prompt] * args.num_samples
    else:
        texts = sampler.load_texts(args.text_file, args.num_samples)
    
    print(f"Loaded {len(texts)} texts")
    for i, text in enumerate(texts[:min(3, len(texts))]):  # Show first 3
        print(f"Text {i+1}: {text}")
    
    # Encode texts
    all_tokens = []
    all_masks = []
    
    for i in range(0, len(texts), args.batch_size):
        batch_texts = texts[i:i+args.batch_size]
        tokens, masks = sampler.encode_text(batch_texts, max_length=args.max_length)
        all_tokens.append(tokens)
        all_masks.append(masks)
    
    text_tokens = torch.cat(all_tokens, dim=0)
    attention_masks = torch.cat(all_masks, dim=0)
    
    print(f"\nText tokens shape: {text_tokens.shape}")
    
    # Generate latents
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_latents = []
    
    for i in range(0, text_tokens.shape[0], args.batch_size):
        batch_tokens = text_tokens[i:i+args.batch_size]
        batch_masks = attention_masks[i:i+args.batch_size]
        
        batch_idx = i // args.batch_size + 1
        total_batches = (text_tokens.shape[0] + args.batch_size - 1) // args.batch_size
        print(f"\nGenerating batch {batch_idx}/{total_batches}")
        
        # Decode and show text
        decoded_text = sampler.tokenizer.decode(
            batch_tokens[0], 
            skip_special_tokens=True
        ).strip()
        print(f"Input text: {decoded_text}")
        
        latents = sampler.generate(
            batch_tokens,
            attention_mask=batch_masks,
            steps=args.steps,
            guidance_scale=args.guidance_scale
        )
        
        all_latents.append(latents.cpu())
        
        # Save individual latent
        for j in range(latents.shape[0]):
            idx = i + j
            latent_np = latents[j].cpu().numpy()
            np.save(output_dir / f"latent_{idx+1:03d}.npy", latent_np)
            
            # Print stats
            print(f"\nSample {idx + 1}:")
            print(f"  Latent shape: {latent_np.shape}")
            print(f"  Mean: {latent_np.mean():.4f}, Std: {latent_np.std():.4f}")
            print(f"  Min: {latent_np.min():.4f}, Max: {latent_np.max():.4f}")
    
    # Combine all latents
    if all_latents:
        all_latents = torch.cat(all_latents, dim=0)
    
    # Save metadata
    with open(output_dir / "results.json", "w", encoding='utf-8') as f:
        json.dump({
            'texts': texts,
            'num_samples': len(texts),
            'latent_dim': sampler.latent_dim,
            'parameters': vars(args)
        }, f, ensure_ascii=False, indent=2)
    
    # Save latents as .pt file
    torch.save(all_latents, output_dir / "all_latents.pt")
    torch.save(text_tokens, output_dir / "text_tokens.pt")
    
    print(f"\nSaved {len(texts)} latents to {output_dir}")
    print(f"\nLatent statistics:")
    print(f"  Shape: {all_latents.shape}")
    print(f"  Global mean: {all_latents.mean():.4f}, std: {all_latents.std():.4f}")
    
    # Save text file with prompts
    with open(output_dir / "prompts.txt", "w", encoding='utf-8') as f:
        for i, text in enumerate(texts):
            f.write(f"Sample {i+1}: {text}\n")

if __name__ == "__main__":
    main()