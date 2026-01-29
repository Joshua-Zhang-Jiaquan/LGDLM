# # File: latentDLM_mmdit/sample_unconditional_fixed.py
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# import argparse
# from pathlib import Path
# import json
# import sys
# import os
# import numpy as np
# import yaml

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# from train_mmdit import ContinuousDiffusion
# from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
# from latentDLM_mmdit.modeling_mmdit import get_tokenizer

# class FixedUnconditionalSampler:
#     def __init__(self, checkpoint_path, config_path=None, device=None):
#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.device = device
#         print(f"Loading checkpoint from: {checkpoint_path}")
        
#         # Load checkpoint
#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
#         # Load config
#         if config_path and os.path.exists(config_path):
#             with open(config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#         elif 'config' in checkpoint and checkpoint['config'] is not None:
#             config = checkpoint['config']
#             print("Using config from checkpoint")
#         else:
#             config = {
#                 'model': {
#                     'hidden_size': 1024,
#                     'n_blocks': 24,
#                     'n_heads': 24,
#                     'cond_dim': 1024,
#                     'max_seq_len': 4096,  # Changed from 1024 to 4096
#                     'dropout': 0.1,
#                     'num_residual_streams': 2,
#                     'qk_rmsnorm': True,
#                     'use_multimodal': True,
#                     'latent_dim': 32,  # Changed from 1024 to 32
#                 }
#             }
#             print("Using default config")
        
#         # Get tokenizer (still needed for text generation)
#         self.tokenizer = get_tokenizer(config)
#         self.mask_token_id = self.tokenizer.mask_token_id
#         self.tokenizer_vocab_size = len(self.tokenizer)
#         print(f"Tokenizer vocab size: {self.tokenizer_vocab_size}")
        
#         # 关键：使用检查点的词汇量，而不是配置中的
#         # 首先检查检查点中是否有vocab_size信息
#         if 'model_state_dict' in checkpoint:
#             # 从text_head.weight的形状推断词汇量
#             for key in checkpoint['model_state_dict']:
#                 if 'text_head.weight' in key:
#                     vocab_size_from_checkpoint = checkpoint['model_state_dict'][key].shape[0]
#                     print(f"Inferred vocab size from checkpoint: {vocab_size_from_checkpoint}")
#                     # 更新config中的vocab_size
#                     config['model']['vocab_size'] = vocab_size_from_checkpoint
#                     break
        
#         model_vocab_size = config['model'].get('vocab_size', 30522)  # Default to 30522
#         print(f"Model vocab size: {model_vocab_size}")
        
#         if model_vocab_size != self.tokenizer_vocab_size:
#             print(f"Warning: Model vocab size ({model_vocab_size}) != Tokenizer vocab size ({self.tokenizer_vocab_size})")
#             print(f"Difference: {model_vocab_size - self.tokenizer_vocab_size} tokens")
        
#         # Create model - 使用模型配置的词汇量
#         latent_dim = config['model'].get('latent_dim', 32)  # Default to 32 based on error
#         print(f"Creating model with latent_dim={latent_dim}, vocab_size={model_vocab_size}")
        
#         # 创建模型时使用检查点的参数
#         self.model = MultimodalMMDiT(
#             config=config['model'],
#             vocab_size=model_vocab_size,  # 使用检查点的词汇量
#             latent_dim=latent_dim,
#             cluster_size=0
#         ).to(device)
        
#         # Load weights
#         if 'model_state_dict' in checkpoint:
#             state_dict = checkpoint['model_state_dict']
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 if k.startswith('model.'):
#                     new_state_dict[k[6:]] = v
#                 else:
#                     new_state_dict[k] = v
            
#             # 尝试加载状态字典
#             try:
#                 self.model.load_state_dict(new_state_dict, strict=True)
#                 print("Model loaded successfully with strict=True")
#             except RuntimeError as e:
#                 print(f"Strict loading failed: {e}")
#                 print("Trying non-strict loading...")
#                 missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
#                 print(f"Model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
#                 if missing:
#                     print(f"Missing keys: {missing[:10]}")
#                 if unexpected:
#                     print(f"Unexpected keys: {unexpected[:10]}")
        
#         self.model.eval()
#         self.model_vocab_size = model_vocab_size
#         self.latent_dim = latent_dim
        
#         # 打印模型信息
#         print(f"\nModel initialized with:")
#         print(f"  Latent dim: {self.latent_dim}")
#         print(f"  Max seq len: {config['model'].get('max_seq_len', 4096)}")
#         print(f"  Vocab size: {self.model_vocab_size}")
    
#     @torch.no_grad()
#     def generate_latents(self, num_samples=1, steps=100):
#         """Generate latents from pure noise (unconditional)"""
#         batch_size = num_samples
        
#         # Initialize latents with full noise
#         latents = torch.randn(batch_size, 1, self.latent_dim, device=self.device)
        
#         # Create empty text tokens (all padding)
#         empty_text = torch.full((batch_size, 1), self.tokenizer.pad_token_id, 
#                                device=self.device, dtype=torch.long)
#         attention_mask = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)
        
#         # Timesteps for diffusion process
#         timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)[:-1]
        
#         # Text is at timestep 0 (fully known)
#         text_timesteps = torch.zeros(batch_size, device=self.device)
        
#         for i in tqdm(range(steps), desc="Generating latents (unconditional)"):
#             t = timesteps[i].expand(batch_size)
            
#             # Forward pass - get latent predictions (unconditional)
#             _, latent_pred = self.model(
#                 text_tokens=empty_text,
#                 latents=latents,
#                 text_timesteps=text_timesteps,
#                 latent_timesteps=t,
#                 attention_mask=attention_mask,
#             )
            
#             # DDPM update rule
#             if i < steps - 1:
#                 # Get next timestep
#                 next_t = timesteps[i + 1]
                
#                 # Simple linear noise schedule
#                 alpha = 1.0 - t
#                 alpha_next = 1.0 - next_t
                
#                 # Add noise for next step
#                 noise = torch.randn_like(latent_pred)
                
#                 # DDPM update: x_{t-1} = pred + sqrt(1 - alpha_bar_next) * noise
#                 noise_scale = torch.sqrt(alpha_next - alpha * (alpha_next / alpha))
#                 noise_scale = torch.clamp(noise_scale, 0.0, 1.0)
                
#                 latents = latent_pred + noise_scale.unsqueeze(1).unsqueeze(2) * noise
#             else:
#                 # Final step: use prediction directly
#                 latents = latent_pred
            
#             # 显示进度
#             if i % 10 == 0:
#                 latents_norm = latents.norm(dim=-1).mean().item()
#                 pred_norm = latent_pred.norm(dim=-1).mean().item()
#                 print(f"Step {i+1}/{steps}: latents_norm={latents_norm:.3f}, pred_norm={pred_norm:.3f}")
        
#         return latents.squeeze(1)  # Remove sequence dimension: [batch_size, latent_dim]
    
#     @torch.no_grad()
#     def generate_text(self, latents=None, seq_len=128, steps=20, temperature=1.0):
#         """Generate text from latents (can use generated latents or input latents)"""
#         if latents is None:
#             # Generate random latents if none provided
#             batch_size = 1
#             latents = torch.randn(batch_size, self.latent_dim, device=self.device)
#         else:
#             batch_size = latents.shape[0]
#             latents = latents.to(self.device)
        
#         # 检查潜在向量的维度
#         if latents.shape[1] != self.latent_dim:
#             print(f"Warning: Input latents have dimension {latents.shape[1]}, but model expects {self.latent_dim}")
#             if latents.shape[1] > self.latent_dim:
#                 print(f"Truncating from {latents.shape[1]} to {self.latent_dim}")
#                 latents = latents[:, :self.latent_dim]
#             else:
#                 print(f"Padding from {latents.shape[1]} to {self.latent_dim}")
#                 padding = torch.zeros(batch_size, self.latent_dim - latents.shape[1], device=self.device)
#                 latents = torch.cat([latents, padding], dim=1)
        
#         # Initialize text with all masks
#         text_tokens = torch.full((batch_size, seq_len), self.mask_token_id,
#                                 device=self.device, dtype=torch.long)
        
#         text_timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)[:-1]
#         latent_timesteps = torch.zeros(batch_size, device=self.device)
        
#         for i in tqdm(range(steps), desc="Generating text"):
#             text_t = text_timesteps[i].expand(batch_size)
            
#             # Forward pass
#             text_logits, _ = self.model(
#                 text_tokens=text_tokens,
#                 latents=latents.unsqueeze(1),
#                 text_timesteps=text_t,
#                 latent_timesteps=latent_timesteps,
#                 attention_mask=None,
#             )
            
#             # 确保形状正确 [batch, seq, vocab]
#             if text_logits.dim() == 3 and text_logits.shape[-1] != self.model_vocab_size:
#                 print(f"Warning: Last dim {text_logits.shape[-1]} != model vocab {self.model_vocab_size}")
#                 # 尝试调整
#                 if text_logits.shape[1] == self.model_vocab_size:
#                     text_logits = text_logits.transpose(1, 2)
#                     print(f"Transposed to: {text_logits.shape}")
            
#             # 截断到tokenizer的词汇量（如果模型词汇量更大）
#             if text_logits.shape[-1] > self.tokenizer_vocab_size:
#                 text_logits = text_logits[..., :self.tokenizer_vocab_size]
            
#             # 应用温度
#             if temperature != 1.0:
#                 text_logits = text_logits / temperature
            
#             # 采样 - 使用截断后的词汇量
#             probs = F.softmax(text_logits, dim=-1)
#             probs_flat = probs.reshape(-1, probs.shape[-1])
#             sampled_flat = torch.multinomial(probs_flat, 1)
#             sampled = sampled_flat.view(batch_size, seq_len)
            
#             # 更新掩码位置
#             mask = (text_tokens == self.mask_token_id)
#             if mask.any():
#                 text_tokens[mask] = sampled[mask]
            
#             # 显示进度
#             if i % 5 == 0:
#                 mask_ratio = mask.float().mean().item()
#                 print(f"Step {i+1}/{steps}: mask_ratio={mask_ratio:.3f}")
        
#         return text_tokens, latents
    
#     def decode(self, tokens):
#         """Decode tokens to text - 处理超出词汇表的token"""
#         texts = []
#         for t in tokens.cpu().numpy():
#             valid = []
#             for tok in t:
#                 # 检查token是否在tokenizer的词汇表中
#                 if tok >= self.tokenizer_vocab_size:
#                     continue
#                 if tok in [self.tokenizer.pad_token_id, 
#                           getattr(self.tokenizer, 'cls_token_id', -1),
#                           getattr(self.tokenizer, 'sep_token_id', -1),
#                           self.mask_token_id]:
#                     continue
#                 valid.append(tok)
            
#             if valid:
#                 text = self.tokenizer.decode(valid, skip_special_tokens=True).strip()
#                 texts.append(text)
#             else:
#                 texts.append("")
#         return texts

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint", required=True)
#     parser.add_argument("--config", default=None)
#     parser.add_argument("--num_latents", type=int, default=3, help="Number of latents to generate")
#     parser.add_argument("--text_steps", type=int, default=20, help="Steps for text generation")
#     parser.add_argument("--latent_steps", type=int, default=100, help="Steps for latent generation")
#     parser.add_argument("--temperature", type=float, default=1.0)
#     parser.add_argument("--seq_len", type=int, default=128, help="Generated text sequence length")
#     parser.add_argument("--output_dir", default="./unconditional_fixed_output")
    
#     args = parser.parse_args()
    
#     # Create sampler
#     sampler = FixedUnconditionalSampler(args.checkpoint, args.config)
    
#     # Generate latents unconditionally
#     print(f"\nGenerating {args.num_latents} latents unconditionally...")
#     latents = sampler.generate_latents(num_samples=args.num_latents, steps=args.latent_steps)
#     print(f"Generated latents shape: {latents.shape}")
    
#     # Generate text from the generated latents
#     print(f"\nGenerating text from the generated latents...")
#     tokens, final_latents = sampler.generate_text(
#         latents=latents,
#         seq_len=args.seq_len,
#         steps=args.text_steps,
#         temperature=args.temperature
#     )
    
#     # Decode tokens to text
#     texts = sampler.decode(tokens)
    
#     # Save results
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(exist_ok=True, parents=True)
    
#     # Save latents as .npy files
#     for i in range(latents.shape[0]):
#         latent_np = latents[i].cpu().numpy()
#         np.save(output_dir / f"generated_latent_{i+1:03d}.npy", latent_np)
    
#     # Save metadata
#     with open(output_dir / "results.json", "w", encoding='utf-8') as f:
#         json.dump({
#             'texts': texts,
#             'num_samples': len(texts),
#             'latent_dim': sampler.latent_dim,
#             'parameters': vars(args)
#         }, f, ensure_ascii=False, indent=2)
    
#     # Save text file with generated texts
#     with open(output_dir / "generated_texts.txt", "w", encoding='utf-8') as f:
#         for i, text in enumerate(texts):
#             f.write(f"Sample {i+1}:\n")
#             f.write(f"Text: {text}\n")
#             f.write(f"Latent stats - Mean: {latents[i].mean():.4f}, Std: {latents[i].std():.4f}\n")
#             f.write("-" * 80 + "\n")
    
#     # Save all data as .pt files
#     torch.save(latents, output_dir / "generated_latents.pt")
#     torch.save(tokens, output_dir / "generated_tokens.pt")
    
#     print(f"\nSaved {len(texts)} samples to {output_dir}")
#     print(f"\nGenerated latents and texts:")
#     for i, (text, latent) in enumerate(zip(texts, latents)):
#         print(f"\nSample {i+1}:")
#         print(f"  Text: {text}")
#         print(f"  Latent - Mean: {latent.mean():.4f}, Std: {latent.std():.4f}")
#         print(f"  Latent - Min: {latent.min():.4f}, Max: {latent.max():.4f}")

# if __name__ == "__main__":
#     main()



# File: latentDLM_mmdit/sample_unconditional_fixed.py
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

class FixedUnconditionalSampler:
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
                    'max_seq_len': 4096,
                    'dropout': 0.1,
                    'num_residual_streams': 2,
                    'qk_rmsnorm': True,
                    'use_multimodal': True,
                    'latent_dim': 32,
                }
            }
            print("Using default config")
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.tokenizer_vocab_size = len(self.tokenizer)
        print(f"Tokenizer vocab size: {self.tokenizer_vocab_size}")
        
        # Get vocab size from checkpoint
        vocab_size = 30522  # Default BERT vocab size
        if 'model_state_dict' in checkpoint:
            for key in checkpoint['model_state_dict']:
                if 'text_head.weight' in key:
                    vocab_size = checkpoint['model_state_dict'][key].shape[0]
                    break
        
        config['model']['vocab_size'] = vocab_size
        print(f"Model vocab size: {vocab_size}")
        
        # Get latent dimension from checkpoint
        latent_dim = 32
        if 'model_state_dict' in checkpoint:
            for key in checkpoint['model_state_dict']:
                if 'latent_head.3.weight' in key or 'latent_head.3.bias' in key:
                    if 'weight' in key:
                        latent_dim = checkpoint['model_state_dict'][key].shape[0]
                    else:
                        latent_dim = checkpoint['model_state_dict'][key].shape[0]
                    break
        
        config['model']['latent_dim'] = latent_dim
        print(f"Latent dim from checkpoint: {latent_dim}")
        
        # Create model
        self.model = MultimodalMMDiT(
            config=config['model'],
            vocab_size=vocab_size,
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
            
            # Remove noise schedule keys (they're not in the model)
            keys_to_remove = []
            for k in new_state_dict.keys():
                if 'noise_schedule' in k:
                    keys_to_remove.append(k)
            
            for k in keys_to_remove:
                del new_state_dict[k]
                print(f"Removed unexpected key: {k}")
            
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"Missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(f"Unexpected keys (first 5): {unexpected[:5]}")
        
        self.model.eval()
        self.model_vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        print(f"\nModel initialized with:")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Max seq len: {config['model'].get('max_seq_len', 4096)}")
        print(f"  Vocab size: {self.model_vocab_size}")
    
    @torch.no_grad()
    def generate_latents(self, num_samples=1, steps=100):
        """Generate latents from pure noise (unconditional)"""
        batch_size = num_samples
        
        # Initialize latents with full noise
        latents = torch.randn(batch_size, 1, self.latent_dim, device=self.device)
        
        # Create empty text tokens (all padding)
        empty_text = torch.full((batch_size, 1), self.tokenizer.pad_token_id, 
                               device=self.device, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)
        
        # Simple DDIM sampling schedule
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)[:-1]
        
        # Text is at timestep 0 (fully known)
        text_timesteps = torch.zeros(batch_size, device=self.device)
        
        for i in tqdm(range(steps), desc="Generating latents"):
            t = timesteps[i].expand(batch_size)
            
            # Forward pass
            _, latent_pred = self.model(
                text_tokens=empty_text,
                latents=latents,
                text_timesteps=text_timesteps,
                latent_timesteps=t,
                attention_mask=attention_mask,
            )
            
            # Debug: check for NaN
            if torch.isnan(latent_pred).any():
                print(f"Warning: NaN detected in latent_pred at step {i}")
                latent_pred = torch.nan_to_num(latent_pred, nan=0.0)
            
            # Simple DDIM update: x_{t-1} = x_t + (pred - x_t) * dt
            if i < steps - 1:
                next_t = timesteps[i + 1]
                dt = t - next_t  # Positive step
                
                # Update: move towards prediction
                latents = latents + (latent_pred - latents) * dt.unsqueeze(1).unsqueeze(2)
                
                # Add small amount of noise if not last step
                if i < steps - 2:
                    noise = torch.randn_like(latents)
                    latents = latents + 0.01 * noise
            else:
                # Final step: use prediction directly
                latents = latent_pred
            
            # Check for NaN
            if torch.isnan(latents).any():
                print(f"Warning: NaN in latents at step {i}, resetting to prediction")
                latents = latent_pred.clone()
            
            # Display progress
            if i % 10 == 0:
                latents_norm = latents.norm(dim=-1).mean().item()
                pred_norm = latent_pred.norm(dim=-1).mean().item()
                if not np.isnan(latents_norm) and not np.isnan(pred_norm):
                    print(f"Step {i+1}/{steps}: latents_norm={latents_norm:.3f}, pred_norm={pred_norm:.3f}")
                else:
                    print(f"Step {i+1}/{steps}: latents_norm=nan, pred_norm=nan")
        
        # Final NaN check
        if torch.isnan(latents).any():
            print("Warning: Final latents contain NaN, using zeros")
            latents = torch.zeros_like(latents)
        
        return latents.squeeze(1)
    
    @torch.no_grad()
    def generate_text(self, latents, seq_len=128, steps=20, temperature=1.0):
        """Generate text from latents"""
        batch_size = latents.shape[0]
        latents = latents.to(self.device)
        
        # Check latent dimension
        if latents.shape[1] != self.latent_dim:
            print(f"Adjusting latent dimension from {latents.shape[1]} to {self.latent_dim}")
            if latents.shape[1] > self.latent_dim:
                latents = latents[:, :self.latent_dim]
            else:
                padding = torch.zeros(batch_size, self.latent_dim - latents.shape[1], device=self.device)
                latents = torch.cat([latents, padding], dim=1)
        
        # Initialize text with all masks
        text_tokens = torch.full((batch_size, seq_len), self.mask_token_id,
                                device=self.device, dtype=torch.long)
        
        text_timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)[:-1]
        latent_timesteps = torch.zeros(batch_size, device=self.device)
        
        for i in tqdm(range(steps), desc="Generating text"):
            text_t = text_timesteps[i].expand(batch_size)
            
            # Forward pass
            text_logits, _ = self.model(
                text_tokens=text_tokens,
                latents=latents.unsqueeze(1),
                text_timesteps=text_t,
                latent_timesteps=latent_timesteps,
                attention_mask=None,
            )
            
            # Check shape [batch, seq, vocab]
            if text_logits.dim() == 3:
                if text_logits.shape[-1] != self.model_vocab_size:
                    # Try to fix shape
                    if text_logits.shape[1] == self.model_vocab_size:
                        text_logits = text_logits.transpose(1, 2)
                
                # Truncate to tokenizer vocab
                if text_logits.shape[-1] > self.tokenizer_vocab_size:
                    text_logits = text_logits[..., :self.tokenizer_vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                text_logits = text_logits / temperature
            
            # Check for NaN or extreme values
            if torch.isnan(text_logits).any() or torch.isinf(text_logits).any():
                print(f"Warning: Invalid values in text_logits at step {i}")
                text_logits = torch.nan_to_num(text_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Sample
            try:
                probs = F.softmax(text_logits, dim=-1)
                
                # Check if probs are valid
                if torch.isnan(probs).any() or (probs < 0).any():
                    print(f"Warning: Invalid probabilities at step {i}, using uniform")
                    probs = torch.ones_like(text_logits) / text_logits.shape[-1]
                
                probs_flat = probs.reshape(-1, probs.shape[-1])
                sampled_flat = torch.multinomial(probs_flat, 1)
                sampled = sampled_flat.view(batch_size, seq_len)
                
            except Exception as e:
                print(f"Error in sampling at step {i}: {e}")
                # Fallback: sample from uniform distribution
                sampled = torch.randint(0, self.tokenizer_vocab_size, (batch_size, seq_len), device=self.device)
            
            # Update mask positions
            mask = (text_tokens == self.mask_token_id)
            if mask.any():
                text_tokens[mask] = sampled[mask]
            
            # Display progress
            if i % 5 == 0:
                mask_ratio = mask.float().mean().item()
                print(f"Step {i+1}/{steps}: mask_ratio={mask_ratio:.3f}")
        
        return text_tokens
    
    def decode(self, tokens):
        """Decode tokens to text"""
        texts = []
        for t in tokens.cpu().numpy():
            valid = []
            for tok in t:
                if tok >= self.tokenizer_vocab_size:
                    continue
                if tok in [self.tokenizer.pad_token_id, 
                          getattr(self.tokenizer, 'cls_token_id', -1),
                          getattr(self.tokenizer, 'sep_token_id', -1),
                          self.mask_token_id]:
                    continue
                valid.append(tok)
            
            if valid:
                text = self.tokenizer.decode(valid, skip_special_tokens=True).strip()
                texts.append(text)
            else:
                texts.append("")
        return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--num_latents", type=int, default=3)
    parser.add_argument("--text_steps", type=int, default=20)
    parser.add_argument("--latent_steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=64, help="Shorter seq_len for better stability")
    parser.add_argument("--output_dir", default="./unconditional_results")
    
    args = parser.parse_args()
    
    # Enable CUDA error checking
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Create sampler
    sampler = FixedUnconditionalSampler(args.checkpoint, args.config)
    
    print(f"\nGenerating {args.num_latents} latents unconditionally...")
    
    # Try to generate latents
    try:
        latents = sampler.generate_latents(num_samples=args.num_latents, steps=args.latent_steps)
        print(f"Generated latents shape: {latents.shape}")
        
        # Check for NaN in latents
        if torch.isnan(latents).any():
            print("Warning: Generated latents contain NaN, using random latents instead")
            latents = torch.randn(args.num_latents, sampler.latent_dim, device=sampler.device)
        
        print(f"Latent stats - Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
        print(f"Min: {latents.min():.4f}, Max: {latents.max():.4f}")
        
    except Exception as e:
        print(f"Error generating latents: {e}")
        print("Using random latents instead")
        latents = torch.randn(args.num_latents, sampler.latent_dim, device=sampler.device)
    
    print(f"\nGenerating text from the generated latents...")
    
    # Generate text
    try:
        tokens = sampler.generate_text(
            latents=latents,
            seq_len=args.seq_len,
            steps=args.text_steps,
            temperature=args.temperature
        )
        
        texts = sampler.decode(tokens)
        
    except Exception as e:
        print(f"Error generating text: {e}")
        texts = [f"Error: {str(e)}" for _ in range(args.num_latents)]
        tokens = None
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save latents
    for i in range(latents.shape[0]):
        latent_np = latents[i].cpu().numpy()
        np.save(output_dir / f"generated_latent_{i+1:03d}.npy", latent_np)
    
    # Save metadata
    with open(output_dir / "results.json", "w", encoding='utf-8') as f:
        json.dump({
            'texts': texts,
            'num_samples': len(texts),
            'latent_dim': sampler.latent_dim,
            'parameters': vars(args)
        }, f, ensure_ascii=False, indent=2)
    
    # Save generated texts
    with open(output_dir / "generated_texts.txt", "w", encoding='utf-8') as f:
        for i, text in enumerate(texts):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Text: {text}\n")
            if i < latents.shape[0]:
                latent_mean = latents[i].mean().item()
                latent_std = latents[i].std().item()
                f.write(f"Latent stats - Mean: {latent_mean:.4f}, Std: {latent_std:.4f}\n")
            f.write("-" * 80 + "\n")
    
    # Save data
    torch.save(latents.cpu(), output_dir / "generated_latents.pt")
    if tokens is not None:
        torch.save(tokens.cpu(), output_dir / "generated_tokens.pt")
    
    print(f"\nSaved {len(texts)} samples to {output_dir}")
    print(f"\nGenerated latents and texts:")
    for i, text in enumerate(texts):
        print(f"\nSample {i+1}:")
        print(f"  Text: {text}")
        if i < latents.shape[0]:
            latent = latents[i]
            print(f"  Latent - Mean: {latent.mean():.4f}, Std: {latent.std():.4f}")
            print(f"  Latent - Min: {latent.min():.4f}, Max: {latent.max():.4f}")

if __name__ == "__main__":
    main()