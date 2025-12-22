# # File: latentDLM_mmdit/data_simple.py
# import json
# from pathlib import Path
# import torch
# import numpy as np
# from datasets import Dataset
# import random


# File: latentDLM_mmdit/data_simple.py (CORRECTED)
import json
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset  # Import PyTorch's base Dataset
import random


class SimpleLatentDataset(Dataset):  # Inherit ONLY from torch.utils.data.Dataset
    """Simple dataset that loads text and latent pairs from JSON."""
    
    def __init__(self, json_path, tokenizer, max_length=512, max_samples=None):
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.samples = json.load(f)  # Changed variable name from 'data' to 'samples'
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {json_path}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]  # Use the new name 'samples' here
        text = item.get('text', '')
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
        }
        
        # Load latent if available
        if 'latent_path' in item:
            latent_path = Path(self.json_path.parent) / item['latent_path']
            if latent_path.exists():
                try:
                    latent = np.load(latent_path)
                    # Ensure latent is a tensor of shape [1, latent_dim]
                    latent_tensor = torch.from_numpy(latent).float()
                    if latent_tensor.dim() == 1:
                        latent_tensor = latent_tensor.unsqueeze(0)
                    result['latent'] = latent_tensor
                except Exception as e:
                    print(f"Warning: Could not load latent from {latent_path}: {e}")
                    # Optionally, create a zero latent as fallback
                    # latent_dim = config.model.get('latent_dim', 768)
                    # result['latent'] = torch.zeros(1, latent_dim)
        
        return result


def get_simple_dataloaders(config, tokenizer):
    """Simple dataloader that works with JSON files and latents."""
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # Create datasets
    train_json = Path(config.data.data_files.train)
    val_json = Path(config.data.data_files.validation) if hasattr(config.data, 'data_files') and hasattr(config.data.data_files, 'validation') else None
    
    train_ds = SimpleLatentDataset(
        train_json,
        tokenizer,
        max_length=config.model.max_seq_len,
        max_samples=config.data.get('max_samples', None)
    )
    
    if val_json and val_json.exists():
        test_ds = SimpleLatentDataset(
            val_json,
            tokenizer,
            max_length=config.model.max_seq_len,
            max_samples=config.data.get('max_val_samples', 1000)
        )
    else:
        # Create validation split from training
        print("No validation JSON found, splitting training data")
        total_size = len(train_ds)
        val_size = min(1000, total_size // 10)
        indices = list(range(total_size))
        random.shuffle(indices)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        from torch.utils.data import Subset
        test_ds = Subset(train_ds, val_indices)
        train_ds = Subset(train_ds, train_indices)
    
    # Collate function
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Handle latents
        if 'latent' in batch[0]:
            latents = [item['latent'] for item in batch]
            # Pad if variable length
            max_len = max(lat.shape[0] for lat in latents)
            latent_dim = latents[0].shape[-1]
            padded_latents = torch.zeros(len(latents), max_len, latent_dim)
            for i, lat in enumerate(latents):
                padded_latents[i, :lat.shape[0]] = lat
            result['latent'] = padded_latents
        
        return result
    
    # Create distributed sampler if needed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    
    # Create dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.train_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    test_dl = DataLoader(
        test_ds,
        batch_size=config.training.eval_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )
    
    return train_dl, test_dl















# # File: latentDLM_mmdit/data_simple.py
# import json
# from pathlib import Path
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import random
# from datasets import load_dataset
# import hashlib


# class SimpleLatentDataset(Dataset):
#     """Dataset that loads FULL OpenWebText + latents where available."""
    
#     def __init__(self, config, tokenizer, split="train"):
#         self.config = config
#         self.tokenizer = tokenizer
#         self.split = split
#         self.max_length = config.model.max_seq_len
        
#         # Load FULL dataset (all ~8M samples)
#         print(f"Loading {split} dataset: {config.data.dataset_name}")
        
#         # Load main text dataset
#         if config.data.dataset_name == "openwebtext":
#             if split == "train":
#                 # Load training split (full dataset)
#                 self.text_dataset = load_dataset(
#                     "openwebtext",
#                     split="train",
#                     trust_remote_code=True,
#                     cache_dir=config.data.cache_dir,
#                     streaming=False
#                 )
#                 print(f"Loaded ALL OpenWebText training samples: {len(self.text_dataset)}")
#             else:
#                 # Validation split
#                 test_size = config.data.get('test_size', 5000)
#                 self.text_dataset = load_dataset(
#                     "openwebtext",
#                     split=f"train[-{test_size}:]",
#                     trust_remote_code=True,
#                     cache_dir=config.data.cache_dir,
#                     streaming=False
#                 )
#         else:
#             # Other datasets
#             self.text_dataset = load_dataset(
#                 config.data.dataset_name,
#                 split=split,
#                 cache_dir=config.data.cache_dir,
#                 streaming=False
#             )
        
#         # Load latent mappings (your 20k processed samples)
#         self.latent_index = {}
#         if hasattr(config.data, 'latent_data_root'):
#             latent_root = Path(config.data.latent_data_root)
            
#             # Load index file
#             index_file = latent_root / "train_data.json"
#             if index_file.exists():
#                 with open(index_file, 'r', encoding='utf-8') as f:
#                     latent_data = json.load(f)
                
#                 # Create better matching: hash full text
#                 for item in latent_data:
#                     text = item.get('text', '').strip()
#                     if text and 'latent_path' in item:
#                         # Hash the full text for accurate matching
#                         text_hash = hashlib.md5(text.encode()).hexdigest()
#                         self.latent_index[text_hash] = {
#                             'path': latent_root / item['latent_path'],
#                             'text': text,
#                             'id': item.get('id', 0)
#                         }
                
#                 print(f"Loaded {len(self.latent_index)} latent mappings")
#             else:
#                 print(f"Warning: No latent index found at {index_file}")
        
#         self.has_latents = len(self.latent_index) > 0
#         self.total_size = len(self.text_dataset)
        
#         # Track statistics
#         self.stats = {
#             'total_samples': self.total_size,
#             'latent_samples': len(self.latent_index),
#             'latent_coverage': len(self.latent_index) / self.total_size if self.total_size > 0 else 0
#         }
        
#         print(f"Total {split} samples: {self.total_size:,}")
#         print(f"With latents: {self.stats['latent_samples']:,} ({self.stats['latent_coverage']*100:.2f}%)")
    
#     def __len__(self):
#         return self.total_size
    
#     def __getitem__(self, idx):
#         # Get text from full dataset
#         item = self.text_dataset[idx]
#         text = item.get('text', '').strip()
        
#         # Tokenize text
#         tokenized = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
        
#         result = {
#             'input_ids': tokenized['input_ids'].squeeze(0),
#             'attention_mask': tokenized['attention_mask'].squeeze(0),
#             'has_latent': torch.tensor(0, dtype=torch.bool)  # Default: no latent
#         }
        
#         # Try to find matching latent
#         if self.has_latents and text:
#             # Hash text for matching
#             text_hash = hashlib.md5(text.encode()).hexdigest()
            
#             if text_hash in self.latent_index:
#                 latent_info = self.latent_index[text_hash]
#                 latent_path = latent_info['path']
                
#                 if latent_path.exists():
#                     try:
#                         latent = np.load(latent_path)
#                         latent_tensor = torch.from_numpy(latent).float()
                        
#                         # Ensure correct shape: [seq_len, latent_dim]
#                         if latent_tensor.dim() == 1:
#                             latent_tensor = latent_tensor.unsqueeze(0)  # [1, dim]
                        
#                         result['latent'] = latent_tensor
#                         result['has_latent'] = torch.tensor(1, dtype=torch.bool)
                        
#                     except Exception as e:
#                         # Silently skip if latent loading fails
#                         pass
        
#         return result


# def collate_latent_fn(batch, config):
#     """Collate function that handles variable latent presence."""
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     attention_mask = torch.stack([item['attention_mask'] for item in batch])
#     has_latent = torch.stack([item['has_latent'] for item in batch])
    
#     result = {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'has_latent': has_latent,
#     }
    
#     # Check if any sample has latents
#     any_has_latent = has_latent.any().item()
    
#     if any_has_latent:
#         latent_dim = config.model.get('latent_dim', 768)
        
#         # Collect latents (use zeros for samples without latents)
#         latents = []
#         for i, item in enumerate(batch):
#             if 'latent' in item:
#                 latents.append(item['latent'])
#             else:
#                 # Zero latent for samples without latents
#                 latents.append(torch.zeros(1, latent_dim))
        
#         # Pad to same length
#         max_len = max(lat.shape[0] for lat in latents)
#         padded_latents = torch.zeros(len(latents), max_len, latent_dim)
        
#         for i, lat in enumerate(latents):
#             padded_latents[i, :lat.shape[0]] = lat
        
#         result['latent'] = padded_latents
#         result['latent_mask'] = (padded_latents != 0).any(dim=-1)  # Mask for actual latents
    
#     return result


# def get_simple_dataloaders(config, tokenizer):
#     """Main dataloader function - loads FULL dataset."""
    
#     print("=" * 60)
#     print("Creating dataloaders with FULL dataset + latent conditioning")
#     print("=" * 60)
    
#     # Create datasets
#     train_ds = SimpleLatentDataset(config, tokenizer, split="train")
    
#     # Create validation dataset
#     if hasattr(config.data, 'dataset_name'):
#         if config.data.dataset_name == "openwebtext":
#             # Use last portion of training data for validation
#             test_size = config.data.get('test_size', 5000)
#             print(f"Using last {test_size} samples for validation")
            
#             # Create validation subset
#             from torch.utils.data import Subset
#             val_indices = list(range(len(train_ds) - test_size, len(train_ds)))
#             test_ds = Subset(train_ds, val_indices)
            
#             # Trim training dataset
#             train_indices = list(range(len(train_ds) - test_size))
#             train_ds = Subset(train_ds, train_indices)
#         else:
#             # Other datasets may have separate validation split
#             test_ds = SimpleLatentDataset(config, tokenizer, split="validation")
#     else:
#         # Fallback: random split
#         print("Creating random validation split")
#         total_size = len(train_ds)
#         val_size = min(5000, total_size // 10)
#         indices = list(range(total_size))
#         random.shuffle(indices)
#         val_indices = indices[:val_size]
#         train_indices = indices[val_size:]
        
#         from torch.utils.data import Subset
#         test_ds = Subset(train_ds, val_indices)
#         train_ds = Subset(train_ds, train_indices)
    
#     print(f"Training samples: {len(train_ds):,}")
#     print(f"Validation samples: {len(test_ds):,}")
    
#     # Create partial collate function with config
#     from functools import partial
#     collate_fn = partial(collate_latent_fn, config=config)
    
#     # Create distributed sampler if needed
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         train_sampler = DistributedSampler(train_ds, shuffle=True)
#         test_sampler = DistributedSampler(test_ds, shuffle=False)
#     else:
#         train_sampler = None
#         test_sampler = None
    
#     # Create dataloaders
#     train_dl = DataLoader(
#         train_ds,
#         batch_size=config.training.train_batch_size,
#         sampler=train_sampler,
#         shuffle=(train_sampler is None),
#         collate_fn=collate_fn,
#         num_workers=config.data.get('num_workers', 4),
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=True
#     )
    
#     test_dl = DataLoader(
#         test_ds,
#         batch_size=config.training.eval_batch_size,
#         sampler=test_sampler,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=config.data.get('num_workers', 2),
#         pin_memory=True,
#         drop_last=False
#     )
    
#     return train_dl, test_dl