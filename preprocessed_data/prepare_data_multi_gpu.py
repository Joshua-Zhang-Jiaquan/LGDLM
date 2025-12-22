#!/usr/bin/env python
"""Prepare text and latent data for MMDiT text-to-latent training using PyTorch Distributed."""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from datasets import load_dataset
import hashlib
import random
import time
import os
import sys

# -----------------------------------------------------------------------------
# Distributed Setup
# -----------------------------------------------------------------------------

def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        torch.cuda.set_device(local_rank)
        
        print(f"Initialized process group: rank {rank}/{world_size}, local_rank {local_rank}")
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0

# -----------------------------------------------------------------------------
# LM1B Detokenizer
# -----------------------------------------------------------------------------

def lm1b_detokenizer(x):
    """Detokenize LM1B text."""
    if not isinstance(x, str):
        return x
    
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x

# -----------------------------------------------------------------------------
# Latent Extraction Models
# -----------------------------------------------------------------------------

class T5Encoder(torch.nn.Module):
    """T5 Encoder for distributed extraction."""
    
    def __init__(self, model_name: str = "google/t5-v1_1-large"):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        
        # Use 8-bit if available
        try:
            import bitsandbytes as bnb
            self.encoder = self.encoder.to(torch.bfloat16)
            print("Using bfloat16 precision")
        except:
            pass
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to latent vectors."""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.encoder.device)
        
        outputs = self.encoder(**inputs)
        # Mean pooling over sequence
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

class SONAREncoder(torch.nn.Module):
    """SONAR Encoder for distributed extraction."""
    
    def __init__(self, model_name: str = "facebook/SONAR"):
        super().__init__()
        try:
            # Try to import SONAR pipeline
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            print("Using official SONAR pipeline")
            self.use_official = True
            self.pipeline = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=torch.device("cuda"),
                dtype=torch.float16,
            )
        except ImportError:
            # Fallback to transformers
            print("Using transformers SONAR (official SONAR not available)")
            from transformers import AutoModel, AutoTokenizer
            self.use_official = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                "facebook/SONAR",
                trust_remote_code=True
            )
            self.encoder = AutoModel.from_pretrained(
                "facebook/SONAR",
                trust_remote_code=True
            )
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using SONAR."""
        if self.use_official:
            # Use official SONAR pipeline
            embeddings = self.pipeline.predict(
                texts,
                source_lang="eng_Latn",
                batch_size=len(texts)
            )
            if isinstance(embeddings, torch.Tensor):
                return embeddings
            else:
                return torch.tensor(embeddings, device="cuda")
        else:
            # Use transformers fallback
            embeddings_list = []
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.encoder.device)
                
                outputs = self.encoder(**inputs)
                # Mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings_list.append(embedding)
            
            return torch.cat(embeddings_list, dim=0)

class E5Encoder(torch.nn.Module):
    """E5 Encoder for distributed extraction."""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading E5 model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using E5."""
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embeddings

class QwenEncoder(torch.nn.Module):
    """Qwen Encoder for distributed extraction."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Extract embeddings from Qwen."""
        embeddings_list = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            outputs = self.model(**inputs, output_hidden_states=True)
            # Get last hidden state and mean pool
            last_hidden = outputs.hidden_states[-1]
            embedding = last_hidden.mean(dim=1)
            embeddings_list.append(embedding)
        
        return torch.cat(embeddings_list, dim=0)

class BGEM3Encoder(torch.nn.Module):
    """BGE M3 Encoder wrapper using FlagEmbedding's BGEM3FlagModel."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        super().__init__()
        try:
            from FlagEmbedding import BGEM3FlagModel
            print(f"Loading BGE model: {model_name} via FlagEmbedding")
            # BGEM3FlagModel handles internal device/precision settings.
            self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        except Exception as e:
            print("Could not initialize BGEM3FlagModel:", e)
            self.model = None

        # Default device: current CUDA device if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to dense vectors using the BGEM3FlagModel.

        Returns a torch.Tensor on the current device.
        """
        if self.model is None:
            raise RuntimeError("BGEM3FlagModel not available (failed to import/initialize)")

        # Use a reasonable per-call batch size; BGEM3 supports long sequences.
        try:
            encoded = self.model.encode(
                texts,
                batch_size=min(12, max(1, len(texts))),
                max_length=8192,
            )
            dense = encoded.get('dense_vecs', None) or encoded.get('dense', None) or encoded
        except TypeError:
            # Some wrappers may accept different args
            encoded = self.model.encode(texts)
            dense = encoded.get('dense_vecs', None) or encoded

        arr = np.asarray(dense)
        tensor = torch.from_numpy(arr).to(self.device)
        return tensor

# -----------------------------------------------------------------------------
# Model Factory
# -----------------------------------------------------------------------------

def create_model(model_type: str, rank: int, local_rank: int):
    """Create the appropriate model for given type."""
    model = None
    
    if model_type == "t5":
        model = T5Encoder("google/t5-v1_1-large")
    elif model_type == "sonar":
        model = SONAREncoder("facebook/SONAR")
    elif model_type == "e5":
        model = E5Encoder("intfloat/multilingual-e5-large")
    elif model_type == "bge":
        model = BGEM3Encoder("BAAI/bge-m3")
    elif model_type == "qwen":
        model = QwenEncoder("Qwen/Qwen3-Embedding-0.6B")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move to GPU
    model = model.cuda(local_rank)
    
    # Wrap with DDP if multi-GPU
    if dist.is_initialized() and dist.get_world_size() > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    print(f"[Rank {rank}] Created {model_type} model on GPU {local_rank}")
    return model

# -----------------------------------------------------------------------------
# Distributed Dataset Processor
# -----------------------------------------------------------------------------

class DistributedDatasetProcessor:
    """Process dataset using distributed GPU setup."""
    
    def __init__(
        self,
        dataset_name: str,
        output_dir: Path,
        model_type: str,
        rank: int,
        world_size: int,
        local_rank: int,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
        resume: bool = True
    ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.model_type = model_type
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.resume = resume
        
        # Load dataset
        print(f"[Rank {rank}] Loading dataset {dataset_name}")
        if dataset_name == "openwebtext":
            dataset = load_dataset("openwebtext", trust_remote_code=True, streaming=False)
            self.dataset = dataset["train"]
            self.text_key = "text"
        elif dataset_name == "lm1b":
            dataset = load_dataset("lm1b", trust_remote_code=True, streaming=False)
            self.dataset = dataset["train"]
            self.text_key = "text"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.total_samples = len(self.dataset)
        if max_samples:
            self.total_samples = min(max_samples, self.total_samples)
        
        # Setup output directories
        self.text_dir = output_dir / "texts" / "train"
        self.latent_dir = output_dir / "latents" / "train"
        
        if rank == 0:  # Only rank 0 creates directories
            self.text_dir.mkdir(parents=True, exist_ok=True)
            self.latent_dir.mkdir(parents=True, exist_ok=True)
        
        # Sync directory creation
        if world_size > 1:
            dist.barrier()
        
        # Load progress
        self.processed_indices = self._load_progress()
        
        # Initialize model
        self.model = create_model(model_type, rank, local_rank)
    
    def _load_progress(self) -> set:
        """Load processed indices from checkpoint."""
        checkpoint_file = self.output_dir / f"progress_rank_{self.rank}.json"
        
        if self.resume and checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    progress = json.load(f)
                    indices = set(progress.get("processed_indices", []))
                    print(f"[Rank {self.rank}] Loaded {len(indices)} processed indices")
                    return indices
            except:
                print(f"[Rank {self.rank}] Could not load checkpoint")
        
        return set()
    
    def _save_progress(self, processed_indices: set):
        """Save progress to checkpoint."""
        checkpoint_file = self.output_dir / f"progress_rank_{self.rank}.json"
        
        progress = {
            "processed_indices": list(processed_indices),
            "last_index": max(processed_indices) if processed_indices else -1,
            "total_processed": len(processed_indices),
            "rank": self.rank,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def process(self) -> List[Dict[str, Any]]:
        """Process dataset in distributed manner."""
        all_samples = []
        
        # Each rank processes its own slice of data
        samples_per_rank = self.total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if self.rank < self.world_size - 1 else self.total_samples
        
        print(f"[Rank {self.rank}] Processing indices {start_idx:,} to {end_idx:,} "
              f"({end_idx - start_idx:,} samples)")
        
        # Skip already processed indices
        indices_to_process = []
        for idx in range(start_idx, end_idx):
            if idx not in self.processed_indices:
                indices_to_process.append(idx)
        
        print(f"[Rank {self.rank}] {len(indices_to_process)} samples to process "
              f"(skipped {end_idx - start_idx - len(indices_to_process)} already processed)")
        
        # Process in batches
        for batch_start in tqdm(
            range(0, len(indices_to_process), self.batch_size),
            desc=f"Rank {self.rank}",
            disable=self.rank != 0  # Only rank 0 shows progress bar
        ):
            batch_end = min(batch_start + self.batch_size, len(indices_to_process))
            batch_indices = indices_to_process[batch_start:batch_end]
            
            # Get batch texts
            batch_texts = []
            valid_indices = []
            
            for idx in batch_indices:
                try:
                    item = self.dataset[idx]
                    text = item.get(self.text_key, '').strip()
                    
                    if not text or not isinstance(text, str):
                        continue
                    
                    if self.dataset_name == "lm1b":
                        text = lm1b_detokenizer(text)
                    
                    text = text.strip()
                    if len(text) < 20:
                        continue
                    
                    batch_texts.append(text)
                    valid_indices.append(idx)
                    
                except Exception as e:
                    print(f"[Rank {self.rank}] Error at index {idx}: {e}")
                    continue
            
            if not batch_texts:
                continue
            
            # Encode texts
            try:
                with torch.no_grad():
                    latents = self.model(batch_texts)
                
                # Save results
                batch_samples = []
                for idx, text, latent in zip(valid_indices, batch_texts, latents):
                    text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
                    filename = f"{self.dataset_name}_{idx:08d}_{text_hash}"
                    
                    # Save text file
                    text_path = self.text_dir / f"{filename}.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    
                    # Save latent file
                    latent_path = self.latent_dir / f"{filename}.npy"
                    np.save(latent_path, latent.cpu().numpy())
                    
                    batch_samples.append({
                        "text": text,
                        "text_path": str(text_path.relative_to(self.output_dir)),
                        "latent_path": str(latent_path.relative_to(self.output_dir)),
                        "dataset": self.dataset_name,
                        "id": idx
                    })
                
                all_samples.extend(batch_samples)
                self.processed_indices.update(valid_indices)
                
                # Save checkpoint every 1000 samples
                if len(all_samples) % 1000 == 0:
                    self._save_progress(self.processed_indices)
                    if self.rank == 0:
                        print(f"[Rank {self.rank}] Checkpoint saved: {len(all_samples)} samples")
                        
            except Exception as e:
                print(f"[Rank {self.rank}] Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final progress save
        self._save_progress(self.processed_indices)
        
        return all_samples

# -----------------------------------------------------------------------------
# Main Distributed Processing Function
# -----------------------------------------------------------------------------

def main_distributed(args):
    """Main function for distributed processing."""
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Processing with {world_size} GPUs")
        print(f"Latent model: {args.latent_model}")
        print(f"Output directory: {args.output_dir}")
        print(f"{'='*60}")
    
    output_dir = Path(args.output_dir)
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sync directory creation
    if world_size > 1:
        dist.barrier()
    
    all_samples = []
    start_time = time.time()
    
    # Process each dataset
    for dataset_name in args.datasets:
        if rank == 0:
            print(f"\nProcessing dataset: {dataset_name}")
        
        processor = DistributedDatasetProcessor(
            dataset_name=dataset_name,
            output_dir=output_dir,
            model_type=args.latent_model,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            resume=args.resume
        )
        
        samples = processor.process()
        all_samples.extend(samples)
        
        if rank == 0:
            print(f"Rank {rank} completed: {len(samples)} samples")
    
    # Gather all samples to rank 0
    if world_size > 1:
        # Convert samples to string for gathering
        samples_json = json.dumps(all_samples, ensure_ascii=False)
        
        # Gather all JSON strings to rank 0
        gathered_samples = [None] * world_size
        dist.gather_object(samples_json, gathered_samples if rank == 0 else None, dst=0)
        
        if rank == 0:
            # Combine all samples
            combined_samples = []
            for sample_json in gathered_samples:
                if sample_json:
                    combined_samples.extend(json.loads(sample_json))
            
            all_samples = combined_samples
    else:
        combined_samples = all_samples
    
    # Rank 0 saves final dataset
    if rank == 0:
        # Remove duplicates
        unique_samples = {}
        for sample in combined_samples:
            unique_samples[sample["id"]] = sample
        all_samples = list(unique_samples.values())
        
        print(f"\nTotal samples processed: {len(all_samples)}")
        
        # Create validation split if requested
        if args.create_validation and all_samples:
            print("Creating validation split...")
            
            val_text_dir = output_dir / "texts" / "validation"
            val_latent_dir = output_dir / "latents" / "validation"
            val_text_dir.mkdir(parents=True, exist_ok=True)
            val_latent_dir.mkdir(parents=True, exist_ok=True)
            
            random.seed(42)
            random.shuffle(all_samples)
            split_idx = int(len(all_samples) * (1 - args.val_ratio))
            
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]
            
            # Move validation files
            for sample in tqdm(val_samples, desc="Moving validation files"):
                old_text_path = output_dir / sample["text_path"]
                old_latent_path = output_dir / sample["latent_path"]
                
                if old_text_path.exists() and old_latent_path.exists():
                    new_text_path = val_text_dir / old_text_path.name
                    new_latent_path = val_latent_dir / old_latent_path.name
                    
                    shutil.move(str(old_text_path), str(new_text_path))
                    shutil.move(str(old_latent_path), str(new_latent_path))
                    
                    sample["text_path"] = str(new_text_path.relative_to(output_dir))
                    sample["latent_path"] = str(new_latent_path.relative_to(output_dir))
            
            # Save JSON files
            with open(output_dir / "train_data.json", 'w', encoding='utf-8') as f:
                json.dump(train_samples, f, indent=2, ensure_ascii=False)
            
            with open(output_dir / "validation_data.json", 'w', encoding='utf-8') as f:
                json.dump(val_samples, f, indent=2, ensure_ascii=False)
            
            print(f"Split: {len(train_samples)} train, {len(val_samples)} validation")
        else:
            # Save all as training data
            with open(output_dir / "train_data.json", 'w', encoding='utf-8') as f:
                json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        elapsed_time = time.time() - start_time
        stats = {
            "total_samples": len(all_samples),
            "datasets": args.datasets,
            "latent_model": args.latent_model,
            "num_gpus": world_size,
            "batch_size": args.batch_size,
            "processing_time_seconds": round(elapsed_time, 2),
            "processing_time_hours": round(elapsed_time / 3600, 2),
            "samples_per_second": round(len(all_samples) / elapsed_time, 2),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("DATA PREPARATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_samples)}")
        print(f"Processing time: {elapsed_time/3600:.2f} hours")
        print(f"Speed: {len(all_samples)/elapsed_time:.2f} samples/second")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        
        # Clean up progress files
        for rank_file in output_dir.glob("progress_rank_*.json"):
            rank_file.unlink()
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data for MMDiT text-to-latent training using distributed processing."
    )
    
    parser.add_argument("--output-dir", type=str, default="./data_root",
                       help="Output directory for processed data")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["openwebtext"],
                       choices=["openwebtext", "lm1b"],
                       help="Datasets to process")
    parser.add_argument("--latent-model", type=str, default="t5",
                       choices=["t5", "sonar", "e5", "qwen", "bge"],
                       help="Model to extract latent vectors")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size per GPU")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset")
    parser.add_argument("--resume", action="store_true", default=True,
                       help="Resume from checkpoint")
    parser.add_argument("--create-validation", action="store_true",
                       help="Create validation split")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                       help="Ratio of data for validation")
    
    args = parser.parse_args()
    
    # Run distributed processing
    main_distributed(args)

