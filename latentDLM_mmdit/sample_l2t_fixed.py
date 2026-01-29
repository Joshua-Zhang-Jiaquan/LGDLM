# File: latentDLM_mmdit/sample_l2t_fixed.py
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

class FixedL2TSampler:
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
        
        # 打印模型信息
        print(f"\nModel initialized with:")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Max seq len: {config['model'].get('max_seq_len', 1024)}")
        print(f"  Vocab size: {self.model_vocab_size}")
    
    @torch.no_grad()
    def generate(self, latents, seq_len=128, steps=30, temperature=1.0):
        """Generate text from latents using confidence-based progressive decoding"""
        batch_size = latents.shape[0]
        latents = latents.to(self.device)
        
        # 检查潜在向量的维度
        if latents.shape[1] != self.latent_dim:
            print(f"Warning: Input latents have dimension {latents.shape[1]}, but model expects {self.latent_dim}")
            if latents.shape[1] > self.latent_dim:
                print(f"Truncating from {latents.shape[1]} to {self.latent_dim}")
                latents = latents[:, :self.latent_dim]
            else:
                print(f"Padding from {latents.shape[1]} to {self.latent_dim}")
                padding = torch.zeros(batch_size, self.latent_dim - latents.shape[1], device=self.device)
                latents = torch.cat([latents, padding], dim=1)
        
        # Initialize text with all masks
        text_tokens = torch.full((batch_size, seq_len), self.mask_token_id,
                                device=self.device, dtype=torch.long)
        
        text_timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)[:-1]
        latent_timesteps = torch.zeros(batch_size, device=self.device)
        
        # 计算每一步应该解码的 token 数量（使用余弦调度）
        # 早期步骤解码较少，后期解码较多
        total_tokens = seq_len
        
        for i in tqdm(range(steps), desc="Generating"):
            # 获取当前 mask 位置
            mask = (text_tokens == self.mask_token_id)
            
            if not mask.any():
                print(f"All tokens decoded at step {i+1}")
                break  # 所有位置都已解码
            
            text_t = text_timesteps[i].expand(batch_size)
            
            # Forward pass
            text_logits, _ = self.model(
                text_tokens=text_tokens,
                latents=latents.unsqueeze(1),
                text_timesteps=text_t,
                latent_timesteps=latent_timesteps,
                attention_mask=None,
            )
            
            # 调试：检查形状（只在第一步打印）
            if i == 0:
                print(f"\ntext_logits shape: {text_logits.shape}")
                print(f"Expected: [batch={batch_size}, seq={seq_len}, vocab={self.model_vocab_size}]")
            
            # 确保形状正确 [batch, seq, vocab]
            if text_logits.dim() == 3 and text_logits.shape[-1] != self.model_vocab_size:
                if i == 0:
                    print(f"Warning: Last dim {text_logits.shape[-1]} != model vocab {self.model_vocab_size}")
                # 尝试调整
                if text_logits.shape[1] == self.model_vocab_size:
                    text_logits = text_logits.transpose(1, 2)
                    if i == 0:
                        print(f"Transposed to: {text_logits.shape}")
            
            # 截断到tokenizer的词汇量（如果模型词汇量更大）
            if text_logits.shape[-1] > self.tokenizer_vocab_size:
                if i == 0:
                    print(f"Truncating vocab from {text_logits.shape[-1]} to {self.tokenizer_vocab_size}")
                text_logits = text_logits[..., :self.tokenizer_vocab_size]
            
            # 应用温度
            if temperature != 1.0:
                text_logits = text_logits / temperature
            
            # 计算采样概率
            probs = F.softmax(text_logits, dim=-1)
            
            # 采样所有位置
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_flat = torch.multinomial(probs_flat, 1)
            sampled = sampled_flat.view(batch_size, seq_len)
            
            # 计算每个位置的置信度（最大概率）
            confidence = probs.max(dim=-1).values  # [batch, seq]
            
            # 只在 mask 位置计算置信度，其他位置设为 -inf（已解码的不再改变）
            confidence = torch.where(mask, confidence, torch.tensor(float('-inf'), device=self.device))
            
            # 计算这一步应该解码的 token 数量（渐进式）
            # 使用余弦调度：总共需要在 steps 步内解码完所有 token
            # 当前进度
            progress = (i + 1) / steps
            # 累计应该解码的比例（余弦调度，前期慢后期快）
            cumulative_ratio = 1 - np.cos(progress * np.pi / 2)
            # 目标已解码数量
            target_decoded = int(total_tokens * cumulative_ratio)
            
            # 对于每个样本，选择置信度最高的位置解码
            for b in range(batch_size):
                current_mask_count = mask[b].sum().item()
                if current_mask_count == 0:
                    continue
                
                # 当前已解码数量
                current_decoded = total_tokens - current_mask_count
                # 需要新解码的数量
                num_to_decode = max(1, target_decoded - current_decoded)
                num_to_decode = min(num_to_decode, current_mask_count)
                
                # 获取这个样本的置信度
                conf_b = confidence[b]  # [seq]
                
                # 选择置信度最高的 num_to_decode 个位置
                _, top_indices = conf_b.topk(num_to_decode)
                
                # 只更新选中的位置
                text_tokens[b, top_indices] = sampled[b, top_indices]
            
            # 显示进度
            if i % max(1, steps // 20) == 0 or i == steps - 1:
                remaining_mask = (text_tokens == self.mask_token_id).float().mean().item()
                print(f"Step {i+1}/{steps}: remaining_mask_ratio={remaining_mask:.3f}, target_decoded_ratio={cumulative_ratio:.3f}")
        
        return text_tokens
    
    def decode(self, tokens):
        """Decode tokens to text - 处理超出词汇表的token"""
        texts = []
        for t in tokens.cpu().numpy():
            valid = []
            for tok in t:
                # 检查token是否在tokenizer的词汇表中
                if tok >= self.tokenizer_vocab_size:
                    # print(f"Warning: Token {tok} is out of vocabulary (vocab_size={self.tokenizer_vocab_size})")
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
    
    def load_latents(self, npy_dir, num_samples=None):
        """Load .npy files"""
        import glob
        files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        
        if num_samples and len(files) > num_samples:
            import random
            files = random.sample(files, num_samples)
        elif num_samples:
            files = files[:num_samples]
        
        latents = []
        for f in tqdm(files, desc="Loading latents"):
            data = np.load(f)
            # 检查潜在向量的维度
            if data.shape[0] != self.latent_dim:
                print(f"Warning: Latent file {f} has dimension {data.shape[0]}, expected {self.latent_dim}")
                if data.shape[0] > self.latent_dim:
                    data = data[:self.latent_dim]
                else:
                    # 填充到正确的维度
                    padding = np.zeros(self.latent_dim - data.shape[0])
                    data = np.concatenate([data, padding])
            
            latents.append(torch.from_numpy(data).float())
        
        if latents:
            latents_tensor = torch.stack(latents, dim=0)
            print(f"Loaded latents shape: {latents_tensor.shape}")
            print(f"Latent dimension: {self.latent_dim}")
            return latents_tensor
        
        # 如果没有找到文件，创建随机潜在向量
        print(f"No .npy files found in {npy_dir}, creating random latents")
        return torch.randn(num_samples or 3, self.latent_dim)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--npy_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=128, help="Generated text sequence length")
    parser.add_argument("--output_dir", default="./l2t_fixed_output")
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = FixedL2TSampler(args.checkpoint, args.config)
    
    # Load latents
    latents = sampler.load_latents(args.npy_dir, args.num_samples)
    print(f"Loaded {latents.shape[0]} latents with dimension {latents.shape[1]}")
    
    # Generate
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_texts = []
    all_tokens = []
    
    for i in range(0, latents.shape[0], args.batch_size):
        batch = latents[i:i+args.batch_size]
        print(f"\nGenerating batch {i//args.batch_size + 1}/{(latents.shape[0] + args.batch_size - 1)//args.batch_size}")
        print(f"Latent shape: {batch.shape}")
        
        tokens = sampler.generate(batch, seq_len=args.seq_len, steps=args.steps, temperature=args.temperature)
        texts = sampler.decode(tokens)
        
        all_texts.extend(texts)
        all_tokens.append(tokens.cpu())
        
        for j, text in enumerate(texts):
            idx = i + j
            print(f"\nSample {idx + 1}:")
            print(f"Text: {text}")
    
    # Save
    tokens_tensor = None
    if len(all_tokens) > 0:
        tokens_tensor = torch.cat(all_tokens, dim=0)
    
    with open(output_dir / "results.json", "w", encoding='utf-8') as f:
        json.dump({
            'texts': all_texts,
            'num_samples': len(all_texts),
            'latent_dim': sampler.latent_dim,
            'parameters': vars(args)
        }, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "texts.txt", "w", encoding='utf-8') as f:
        for i, text in enumerate(all_texts):
            f.write(f"Sample {i+1}:\n{text}\n\n")
    
    torch.save(latents, output_dir / "latents.pt")
    if tokens_tensor is not None:
        torch.save(tokens_tensor, output_dir / "tokens.pt")
    
    print(f"\nSaved {len(all_texts)} samples to {output_dir}")
    print(f"\nGenerated texts:")
    for i, text in enumerate(all_texts):
        print(f"{i+1}. {text}")

if __name__ == "__main__":
    main()