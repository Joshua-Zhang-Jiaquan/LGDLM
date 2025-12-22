#!/usr/bin/env python
"""MMDiT training with MASKED diffusion objective for language modeling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple
import json
import time
import math

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch.distributions as D

from transformers import AutoModel, AutoTokenizer, BertTokenizer

from mmdit.mmdit_generalized_pytorch import MMDiT

# -----------------------------------------------------------------------------
# Dataset with tokenized text


# -----------------------------------------------------------------------------
# Dataset with tokenized text

@dataclass
class CocoExample:
    image: Tensor
    caption: str
    caption_tokens: Tensor  # Token IDs
    caption_mask: Tensor    # Attention mask
    image_id: int


class CocoCaptionDataset(Dataset[CocoExample]):
    def __init__(
        self, 
        root: Path, 
        split: str, 
        transform: transforms.Compose, 
        tokenizer: BertTokenizer,
        max_length: int = 128,
        max_samples: int | None = None
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load prepared data
        data_file = root / f"{split}_data.json"
        if data_file.exists():
            print(f"Loading data from {data_file}")
            with open(data_file, 'r') as f:
                self.samples = json.load(f)
        else:
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> CocoExample:
        sample = self.samples[index]
        image_path = Path(sample['image_path'])
        caption = sample.get('caption', 'A photo')
        
        # Tokenize caption
        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
        if not image_path.exists():
            image = Image.new("RGB", (256, 256), color="gray")
        else:
            image = Image.open(image_path).convert("RGB")
        
        return CocoExample(
            image=self.transform(image),
            caption=caption,
            caption_tokens=tokenized["input_ids"][0],
            caption_mask=tokenized["attention_mask"][0].bool(),  # Convert to bool here
            image_id=sample.get('image_id', index)
        )


def collate_coco_examples(batch):
    """Custom collate function for CocoExample dataclass."""
    images = torch.stack([item.image for item in batch])
    captions = [item.caption for item in batch]
    caption_tokens = torch.stack([item.caption_tokens for item in batch])
    caption_mask = torch.stack([item.caption_mask for item in batch])
    image_ids = [item.image_id for item in batch]
    
    # Convert mask to boolean
    caption_mask = caption_mask.bool()
    
    return {
        'image': images,
        'caption': captions,
        'caption_tokens': caption_tokens,
        'caption_mask': caption_mask,  # Now boolean
        'image_id': image_ids
    }


# -----------------------------------------------------------------------------
# Discrete Diffusion Utilities for Text


def discrete_diffusion_noise(
    tokens: Tensor, 
    timesteps: Tensor, 
    mask_token_id: int,
    schedule: str = "cosine"
) -> Tuple[Tensor, Tensor]:
    """
    Apply discrete diffusion noise (masking) to tokens.
    
    Args:
        tokens: [batch, seq_len] token IDs
        timesteps: [batch] continuous in [0, 1]
        mask_token_id: ID for [MASK] token
        schedule: noise schedule type
    
    Returns:
        noisy_tokens: tokens with some replaced by mask_token_id
        mask: binary mask indicating which tokens were masked
    """
    batch_size, seq_len = tokens.shape
    
    # Convert continuous timestep to masking probability
    if schedule == "cosine":
        # Cosine schedule (like improved DDPM)
        alpha = torch.cos(timesteps * math.pi / 2).clamp(min=1e-4)
    elif schedule == "linear":
        # Linear schedule
        alpha = 1 - timesteps
    else:
        # Sigmoid schedule
        alpha = 1 / (1 + torch.exp(-10 * (0.5 - timesteps)))
    
    alpha = alpha.view(-1, 1)  # [batch, 1]
    
    # Sample masks
    mask_prob = 1 - alpha
    mask = torch.bernoulli(mask_prob.expand_as(tokens)).bool()
    
    # Apply masking
    noisy_tokens = tokens.clone()
    noisy_tokens[mask] = mask_token_id
    
    return noisy_tokens, mask

def get_timestep_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    """Sinusoidal timestep embeddings (like in original Transformer)."""
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb


def get_discrete_timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Timestep embedding for discrete diffusion."""
    # Same as before but ensures timesteps are in [0, 1]
    timesteps = timesteps.clamp(0, 1)
    return get_timestep_embedding(timesteps * 1000, dim)


# -----------------------------------------------------------------------------
# Text Token Encoder (learnable embedding)


class TextTokenEncoder(nn.Module):
    """Learnable token embeddings for discrete text tokens."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int = 128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        
        # Initialize
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, token_ids: Tensor) -> Tensor:
        """Convert token IDs to embeddings with positional encoding."""
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(token_ids)  # [batch, seq_len, dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)  # [1, seq_len, dim]
        
        # Combine
        embeddings = token_embeds + position_embeds
        return embeddings



class ImagePatchEncoder(nn.Module):
    """Learnable patch embedding with positional encoding."""

    def __init__(self, *, patch_size: int = 16, dim: int = 512, in_channels: int = 3, image_size: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

    def forward(self, images: Tensor) -> Tensor:
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, dim)
        x = x + self.pos_embed  # Add positional encoding
        return x


class BertTextEncoder(nn.Module):
    """Frozen BERT encoder."""

    def __init__(self, *, model_name: str = "bert-base-uncased", max_length: int = 77):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, captions: Iterable[str], device: torch.device) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(
            list(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = self.model(**tokens)
        return outputs.last_hidden_state, tokens["attention_mask"].bool()





# -----------------------------------------------------------------------------
# Discrete Diffusion MMDiT Model


class MaskedDiffusionMMDiT(nn.Module):
    """MMDiT adapted for masked diffusion language modeling."""
    
    def __init__(
        self,
        vocab_size: int,
        text_embed_dim: int,
        image_embed_dim: int,
        cond_dim: int,
        depth: int = 12,
        num_residual_streams: int = 4,
        max_length: int = 128,
        qk_rmsnorm: bool = True,
    ):
        super().__init__()

        self.cond_dim = cond_dim
        
        # Text encoder (learnable)
        self.text_encoder = TextTokenEncoder(vocab_size, text_embed_dim, max_length)
        
        # MMDiT backbone
        self.mmdit = MMDiT(
            depth=depth,
            dim_modalities=(text_embed_dim, image_embed_dim),
            dim_cond=cond_dim,
            qk_rmsnorm=qk_rmsnorm,
            num_residual_streams=num_residual_streams,
        )
        
        # Output head for text (predicts logits over vocabulary)
        self.text_head = nn.Linear(text_embed_dim, vocab_size)
        
        # Initialize output head
        nn.init.normal_(self.text_head.weight, std=0.02)
        nn.init.zeros_(self.text_head.bias)
    
    def forward(
        self,
        text_tokens: Tensor,      # [batch, seq_len] token IDs
        image_tokens: Tensor,     # [batch, num_patches, image_dim]
        text_mask: Tensor,        # [batch, seq_len] attention mask
        timesteps: Tensor,        # [batch] diffusion timesteps
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for masked diffusion.
        
        Returns:
            text_logits: [batch, seq_len, vocab_size] logits for token prediction
            image_pred: [batch, num_patches, image_dim] predicted image tokens
        """
        # Get timestep conditioning
        time_cond = get_discrete_timestep_embedding(timesteps, self.cond_dim)
        
        # Encode text tokens
        text_embeddings = self.text_encoder(text_tokens)  # [batch, seq_len, text_dim]
        
        # Pass through MMDiT
        text_out, image_out = self.mmdit(
            modality_tokens=(text_embeddings, image_tokens),
            modality_masks=(text_mask.bool(), None),  # Ensure boolean mask
            time_cond=time_cond,
        )
        
        # Predict token logits
        text_logits = self.text_head(text_out)
        
        return text_logits, image_out



def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Training Loop with Discrete Diffusion


def train_masked_diffusion() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("MMDiT MASKED DIFFUSION Training")
    print("=" * 80)
    print(f"Using MASKED diffusion objective (predict tokens, not noise)")
    print(f"MMDiT depth: {args.mmdit_depth} layers")
    print(f"Batch size: {args.batch_size}")
    print(f"Dropout: {args.dropout}")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    mask_token_id = tokenizer.mask_token_id
    vocab_size = len(tokenizer)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Mask token ID: {mask_token_id}")
    
    def make_dataloader_with_tokenizer(split: str):
        image_transform = transforms.Compose([
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        dataset = CocoCaptionDataset(
            root=args.data_root,
            split=split,
            transform=image_transform,
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_samples=args.max_samples if split == "train" else min(1000, args.max_samples or 1000),
        )
        
        return DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=(split == "train"), 
            num_workers=4, 
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_coco_examples  # <-- ADD THIS LINE
        )
        
    train_loader = make_dataloader_with_tokenizer("train")
    
    # Try validation
    try:
        val_loader = make_dataloader_with_tokenizer("validation")
        has_validation = True
        print("Validation split loaded")
    except:
        val_loader = None
        has_validation = False
        print("No validation split found")
    
    # Initialize models
    print("\nInitializing models...")
    
    # Image encoder (same as before)
    image_encoder = ImagePatchEncoder(
        patch_size=args.patch_size, 
        dim=args.dim_image,
        image_size=args.image_size
    ).to(device)
    
    # Masked diffusion MMDiT
    model = MaskedDiffusionMMDiT(
        vocab_size=vocab_size,
        text_embed_dim=args.dim_text,
        image_embed_dim=args.dim_image,
        cond_dim=args.dim_cond,
        depth=args.mmdit_depth,
        num_residual_streams=args.num_residual_streams,
        max_length=args.max_length,
        qk_rmsnorm=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in image_encoder.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    
    # Optimizer
    params = list(model.parameters()) + list(image_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(1000, total_steps // 10)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"\nStarting training for {args.epochs} epochs ({total_steps} steps)...")
    print(f"Warmup steps: {warmup_steps}")
    print("=" * 80)
    
    step_iter = 0
    best_val_loss = float('inf')
    start_time = time.time()
    
    model.train()
    image_encoder.train()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        for batch in train_loader:
            images = batch['image'].to(device)
            text_tokens = batch['caption_tokens'].to(device)
            text_mask = batch['caption_mask'].to(device).bool()
            
            # Encode images
            image_tokens = image_encoder(images)
            
            # Sample random timesteps for diffusion
            batch_size = images.shape[0]
            timesteps = torch.rand(batch_size, device=device)  # Uniform [0, 1]
            
            # Apply discrete diffusion noise (masking) to text
            noisy_tokens, mask = discrete_diffusion_noise(
                tokens=text_tokens,
                timesteps=timesteps,
                mask_token_id=mask_token_id,
                schedule=args.noise_schedule
            )
            
            # Forward through model
            text_logits, image_pred = model(
                text_tokens=noisy_tokens,
                image_tokens=image_tokens,
                text_mask=text_mask.bool(),  # Ensure boolean mask
                timesteps=timesteps,
            )
            
            # Compute losses
            
            # 1. Text loss: cross-entropy only on masked tokens
            # Create target ignoring unmasked tokens
            text_target = text_tokens.clone()
            text_target[~mask] = -100  # Ignore index for cross-entropy
            
            text_loss = F.cross_entropy(
                text_logits.view(-1, vocab_size),
                text_target.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            
            # 2. Image loss: MSE (predict original image tokens)
            # Add some noise to images for regularization
            image_noise = torch.randn_like(image_tokens) * 0.1
            image_target = image_tokens + image_noise
            image_loss = F.mse_loss(image_pred, image_target)
            
            # Total loss
            loss = text_loss + args.image_loss_weight * image_loss
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Logging
            if step_iter % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                steps_per_sec = (step_iter + 1) / elapsed
                eta_hours = (total_steps - step_iter) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
                
                # Compute accuracy on masked tokens
                with torch.no_grad():
                    pred_tokens = text_logits.argmax(dim=-1)
                    correct = (pred_tokens[mask] == text_tokens[mask]).sum().item()
                    total_masked = mask.sum().item()
                    accuracy = correct / total_masked if total_masked > 0 else 0
                
                print(
                    f"[{step_iter:06d}/{total_steps:06d}] "
                    f"loss={loss.item():.4f} "
                    f"(text={text_loss.item():.4f}, img={image_loss.item():.4f}) "
                    f"acc={accuracy:.3f} "
                    f"masked={total_masked} "
                    f"lr={lr:.2e} | "
                    f"{steps_per_sec:.2f} it/s | "
                    f"ETA: {eta_hours:.1f}h"
                )
            
            # Validation
            if has_validation and step_iter % args.val_every == 0 and step_iter > 0:
                val_loss = validate_masked(
                    model, image_encoder, val_loader, device, mask_token_id, args
                )
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = args.output_dir / "mmdit_masked_best.pt"
                    torch.save({
                        'step': step_iter,
                        'model_state_dict': model.state_dict(),
                        'image_encoder_state_dict': image_encoder.state_dict(),
                        'val_loss': best_val_loss,
                    }, best_path)
                    print(f"✓ New best model saved! (val_loss={best_val_loss:.4f})\n")
            
            # Save checkpoint
            if step_iter % args.save_every == 0 and step_iter > 0:
                checkpoint_path = args.output_dir / f"mmdit_masked_step_{step_iter:06d}.pt"
                torch.save({
                    'step': step_iter,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'image_encoder_state_dict': image_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'tokenizer': tokenizer,
                }, checkpoint_path)
                print(f"\n✓ Checkpoint saved: {checkpoint_path}\n")
            
            step_iter += 1
    
    # Save final
    final_path = args.output_dir / "mmdit_masked_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'image_encoder_state_dict': image_encoder.state_dict(),
        'tokenizer': tokenizer,
    }, final_path)
    
    print("\n" + "=" * 80)
    print(f"Training complete! Time: {(time.time() - start_time)/3600:.2f}h")
    print(f"Final model: {final_path}")
    if has_validation:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


@torch.no_grad()
def validate_masked(model, image_encoder, val_loader, device, mask_token_id, args):
    """Validation for masked diffusion."""
    model.eval()
    image_encoder.eval()
    
    total_loss = 0
    total_text_loss = 0
    total_image_loss = 0
    total_accuracy = 0
    total_masked = 0
    num_batches = 0
    
    for batch in val_loader:
        images = batch['image'].to(device)
        text_tokens = batch['caption_tokens'].to(device)
        text_mask = batch['caption_mask'].to(device)
        
        # Encode images
        image_tokens = image_encoder(images)
        
        # Sample timesteps
        batch_size = images.shape[0]
        timesteps = torch.rand(batch_size, device=device)
        
        # Apply discrete diffusion noise
        noisy_tokens, mask = discrete_diffusion_noise(
            tokens=text_tokens,
            timesteps=timesteps,
            mask_token_id=mask_token_id,
            schedule=args.noise_schedule
        )
        
        # Forward
        text_logits, image_pred = model(
            text_tokens=noisy_tokens,
            image_tokens=image_tokens,
            text_mask=text_mask.bool(),  # Ensure boolean mask
            timesteps=timesteps,
        )
        
        # Text loss
        text_target = text_tokens.clone()
        text_target[~mask] = -100
        text_loss = F.cross_entropy(
            text_logits.view(-1, len(text_logits.shape[-1])),
            text_target.view(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        # Image loss
        image_loss = F.mse_loss(image_pred, image_tokens)
        
        # Accuracy
        pred_tokens = text_logits.argmax(dim=-1)
        correct = (pred_tokens[mask] == text_tokens[mask]).sum().item()
        total_masked_batch = mask.sum().item()
        
        # Accumulate
        loss = text_loss + args.image_loss_weight * image_loss
        total_loss += loss.item()
        total_text_loss += text_loss.item()
        total_image_loss += image_loss.item()
        total_accuracy += correct
        total_masked += total_masked_batch
        num_batches += 1
        
        if num_batches >= 50:
            break
    
    model.train()
    image_encoder.train()
    
    avg_loss = total_loss / num_batches
    avg_text_loss = total_text_loss / num_batches
    avg_image_loss = total_image_loss / num_batches
    accuracy = total_accuracy / total_masked if total_masked > 0 else 0
    
    print(f"\n>>> VAL: loss={avg_loss:.4f} (text={avg_text_loss:.4f}, img={avg_image_loss:.4f}) acc={accuracy:.3f}")
    
    return avg_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MMDiT with MASKED diffusion objective.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128, help="Max text sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--dim-cond", type=int, default=256)
    parser.add_argument("--dim-image", type=int, default=512)
    parser.add_argument("--dim-text", type=int, default=768)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mmdit-depth", type=int, default=12)
    parser.add_argument("--num-residual-streams", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_masked"))
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--noise-schedule", type=str, default="cosine", choices=["cosine", "linear", "sigmoid"])
    parser.add_argument("--image-loss-weight", type=float, default=1.0, help="Weight for image loss")
    
    return parser.parse_args()


if __name__ == "__main__":
    train_masked_diffusion()