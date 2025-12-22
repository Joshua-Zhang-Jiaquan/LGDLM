#!/usr/bin/env python
"""MMDiT training with proper diffusion objective (not just reconstruction)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import time
import math

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from mmdit.mmdit_generalized_pytorch import MMDiT

# -----------------------------------------------------------------------------
# Dataset (same as before)





#!/usr/bin/env python
"""MMDiT training with proper diffusion objective (not just reconstruction)."""

import os
# Set HuggingFace cache directory BEFORE importing transformers
os.environ['HF_HOME'] = '/home/yl892/rds/hpc-work/mmdit/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/home/yl892/rds/hpc-work/mmdit/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/home/yl892/rds/hpc-work/mmdit/huggingface_cache/datasets'

# Make sure the directory exists
os.makedirs('/home/yl892/rds/hpc-work/mmdit/huggingface_cache', exist_ok=True)
os.makedirs('/home/yl892/rds/hpc-work/mmdit/huggingface_cache/datasets', exist_ok=True)

# Now import everything else
from __future__ import annotations
import argparse
# ... rest of your imports ...



@dataclass
class CocoExample:
    image: Tensor
    caption: str
    image_id: int


class CocoCaptionDataset(Dataset[CocoExample]):
    def __init__(self, root: Path, split: str, transform: transforms.Compose, max_samples: int | None = None):
        self.root = root
        self.split = split
        self.transform = transform
        
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
        
        if not image_path.exists():
            image = Image.new("RGB", (256, 256), color="gray")
        else:
            image = Image.open(image_path).convert("RGB")
        
        return CocoExample(
            image=self.transform(image),
            caption=sample.get('caption', 'A photo'),
            image_id=sample.get('image_id', index)
        )


# -----------------------------------------------------------------------------
# Encoders


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
# Diffusion noise schedule


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


def add_noise(x: Tensor, noise: Tensor, timesteps: Tensor, noise_schedule: str = "cosine") -> Tensor:
    """Add noise to image tokens based on timestep."""
    # Simple linear schedule for now
    alpha = 1 - timesteps / 1000.0  # timesteps in [0, 1000]
    alpha = alpha.view(-1, 1, 1)
    return alpha * x + (1 - alpha).sqrt() * noise


# -----------------------------------------------------------------------------
# Training utilities


# Add this near your other utility functions
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def collate_coco_examples(batch):
    images = torch.stack([item.image for item in batch])
    captions = [item.caption for item in batch]
    image_ids = [item.image_id for item in batch]
    return {'image': images, 'caption': captions, 'image_id': image_ids}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MMDiT with diffusion objective.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for regularization")
    return parser.parse_args()


def make_dataloader(args: argparse.Namespace, split: str) -> DataLoader:
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
        max_samples=args.max_samples if split == "train" else min(1000, args.max_samples or 1000),
    )

    return DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(split == "train"), 
        num_workers=4, 
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_coco_examples
    )


@torch.no_grad()
def validate(model, image_encoder, text_encoder, val_loader, device, dim_cond):
    """Validation loop."""
    model.eval()
    image_encoder.eval()
    
    total_loss = 0
    total_loss_text = 0
    total_loss_image = 0
    num_batches = 0
    
    for batch in val_loader:
        images = batch['image'].to(device)
        captions = batch['caption']
        
        text_tokens_clean, text_mask = text_encoder(captions, device=device)
        image_tokens_clean = image_encoder(images)
        
        # Sample random timesteps
        batch_size = images.shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,), device=device).float()
        
        # Add noise to BOTH
        text_noise = torch.randn_like(text_tokens_clean)
        image_noise = torch.randn_like(image_tokens_clean)
        
        text_tokens_noisy = add_noise(text_tokens_clean, text_noise, timesteps)
        image_tokens_noisy = add_noise(image_tokens_clean, image_noise, timesteps)
        
        # Get timestep embeddings
        time_cond = get_timestep_embedding(timesteps, dim_cond)
        
        # Forward
        text_out, image_out = model(
            modality_tokens=(text_tokens_noisy, image_tokens_noisy),
            modality_masks=(text_mask, None),
            time_cond=time_cond,
        )
        
        # Predict noise for BOTH (with masking for text)
        text_loss_unmasked = F.mse_loss(text_out, text_noise, reduction='none')
        text_loss_masked = (text_loss_unmasked * text_mask.unsqueeze(-1)).sum()
        text_loss = text_loss_masked / text_mask.sum()
        
        loss_text = text_loss.item()
        loss_image = F.mse_loss(image_out, image_noise).item()
        loss = loss_text + loss_image
        
        total_loss += loss
        total_loss_text += loss_text
        total_loss_image += loss_image
        num_batches += 1
        
        if num_batches >= 50:  # Limit validation batches
            break
    
    model.train()
    image_encoder.train()
    
    return {
        'loss': total_loss / num_batches,
        'loss_text': total_loss_text / num_batches,
        'loss_image': total_loss_image / num_batches,
    }

def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MMDiT Diffusion Training")
    print("=" * 80)
    print(f"Using PROPER diffusion objective (predicting noise)")
    print(f"MMDiT depth: {args.mmdit_depth} layers")
    print(f"Batch size: {args.batch_size}")
    print(f"Dropout: {args.dropout}")
    print("=" * 80)

    print("\nInitializing dataloaders...")
    train_loader = make_dataloader(args, "train")
    
    # Try to load validation split
    try:
        val_loader = make_dataloader(args, "validation")
        has_validation = True
        print("Validation split loaded")
    except:
        val_loader = None
        has_validation = False
        print("No validation split found")
    
    print("\nInitializing encoders...")
    text_encoder = BertTextEncoder(max_length=77).to(device)
    image_encoder = ImagePatchEncoder(
        patch_size=args.patch_size, 
        dim=args.dim_image,
        image_size=args.image_size
    ).to(device)

    print(f"\nInitializing MMDiT with {args.mmdit_depth} layers...")
    mmdit = MMDiT(
        depth=args.mmdit_depth,
        dim_modalities=(args.dim_text, args.dim_image),
        dim_cond=args.dim_cond,
        qk_rmsnorm=True,
        num_residual_streams=args.num_residual_streams,
    ).to(device)

    total_params = sum(p.numel() for p in mmdit.parameters()) + sum(p.numel() for p in image_encoder.parameters())
    print(f"Total trainable parameters: {total_params:,}")

    params = list(mmdit.parameters()) + list(image_encoder.parameters())
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
    
    mmdit.train()
    image_encoder.train()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # FIXED version - add this in your training loop:
        for batch in train_loader:
            images = batch['image'].to(device)
            captions = batch['caption']

            # Encode text (frozen) and images (learnable)
            with torch.no_grad():
                text_tokens_clean, text_mask = text_encoder(captions, device=device)
            image_tokens_clean = image_encoder(images)

            # Sample random timesteps for diffusion
            batch_size = images.shape[0]
            timesteps = torch.randint(0, 1000, (batch_size,), device=device).float()
            
            # Add noise to BOTH text and images
            text_noise = torch.randn_like(text_tokens_clean)
            image_noise = torch.randn_like(image_tokens_clean)
            
            # Apply noise with the timestep schedule
            text_tokens_noisy = add_noise(text_tokens_clean, text_noise, timesteps)
            image_tokens_noisy = add_noise(image_tokens_clean, image_noise, timesteps)
            
            # Get timestep embeddings as conditioning
            time_cond = get_timestep_embedding(timesteps, args.dim_cond)

            # Forward through MMDiT with BOTH noisy inputs
            text_out, image_out = mmdit(
                modality_tokens=(text_tokens_noisy, image_tokens_noisy),
                modality_masks=(text_mask, None),
                time_cond=time_cond,
            )

            # Loss: predict noise for BOTH modalities
            # Note: Apply mask to text loss to ignore padding tokens
            text_loss_unmasked = F.mse_loss(text_out, text_noise, reduction='none')
            # Apply attention mask (only compute loss on real tokens)
            text_loss_masked = (text_loss_unmasked * text_mask.unsqueeze(-1)).sum()
            text_loss = text_loss_masked / text_mask.sum()
            
            loss_text = text_loss  # Predict text noise
            loss_image = F.mse_loss(image_out, image_noise)  # Predict image noise
            loss = loss_text + loss_image

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
                
                print(
                    f"[{step_iter:06d}/{total_steps:06d}] "
                    f"loss={loss.item():.4f} "
                    f"(text={loss_text.item():.4f}, noise={loss_image.item():.4f}) "
                    f"lr={lr:.2e} | "
                    f"{steps_per_sec:.2f} it/s | "
                    f"ETA: {eta_hours:.1f}h"
                )

            # Validation
            if has_validation and step_iter % args.val_every == 0 and step_iter > 0:
                val_metrics = validate(mmdit, image_encoder, text_encoder, val_loader, device, args.dim_cond)
                print(f"\n>>> VAL: loss={val_metrics['loss']:.4f} (text={val_metrics['loss_text']:.4f}, noise={val_metrics['loss_image']:.4f})")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_path = args.output_dir / "mmdit_best.pt"
                    torch.save({
                        'step': step_iter,
                        'model_state_dict': mmdit.state_dict(),
                        'image_encoder_state_dict': image_encoder.state_dict(),
                        'val_loss': best_val_loss,
                    }, best_path)
                    print(f"✓ New best model saved! (val_loss={best_val_loss:.4f})\n")
                else:
                    print()

            # Save checkpoint
            if step_iter % args.save_every == 0 and step_iter > 0:
                checkpoint_path = args.output_dir / f"mmdit_step_{step_iter:06d}.pt"
                torch.save({
                    'step': step_iter,
                    'epoch': epoch,
                    'model_state_dict': mmdit.state_dict(),
                    'image_encoder_state_dict': image_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"\n✓ Checkpoint saved: {checkpoint_path}\n")

            step_iter += 1

    # Save final
    final_path = args.output_dir / "mmdit_final.pt"
    torch.save({
        'model_state_dict': mmdit.state_dict(),
        'image_encoder_state_dict': image_encoder.state_dict(),
    }, final_path)
    
    print("\n" + "=" * 80)
    print(f"Training complete! Time: {(time.time() - start_time)/3600:.2f}h")
    print(f"Final model: {final_path}")
    if has_validation:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    train()
