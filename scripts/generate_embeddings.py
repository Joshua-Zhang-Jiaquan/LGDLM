#!/usr/bin/env python
"""Generate latent embeddings from text using E5/Qwen/T5/SONAR models with torchrun support."""

import argparse
import json
from pathlib import Path
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import hashlib
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Optional, Dict, Any

# ========== CRITICAL: Set this BEFORE any imports ==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
# Model Classes
# -----------------------------------------------------------------------------

class E5Encoder(torch.nn.Module):
    """E5 Encoder for embedding extraction."""
    
    def __init__(self, model_path: str = "intfloat/multilingual-e5-large"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        
        device = f"cuda:{torch.cuda.current_device()}"
        
        try:
            self.model = SentenceTransformer(
                model_path, 
                device=device,
                trust_remote_code=True,
                tokenizer_kwargs={'fix_mistral_regex': True}
            )
        except Exception as e:
            print(f"Warning: Could not set fix_mistral_regex: {e}")
            self.model = SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=True
            )
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    @torch.no_grad()
    def forward(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts using E5."""
        if not texts:
            return torch.empty((0, self.embedding_dim), 
                             device=f"cuda:{torch.cuda.current_device()}")
        
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            device=f"cuda:{torch.cuda.current_device()}",
            show_progress_bar=False
        )
        return embeddings

class QwenEmbeddingEncoder(torch.nn.Module):
    """Qwen/Qwen3-Embedding-8B Encoder."""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-Embedding-8B", 
                 embedding_dim: int = 1024):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        
        model_kwargs = {
            'dtype': torch.bfloat16
        }
        
        # Load the full model
        self.model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={'padding_side': 'left'}
        )
        
        self.full_dim = self.model.get_sentence_embedding_dimension()
        self.embedding_dim = min(embedding_dim, self.full_dim)
        
        self.use_prompt = True
        
    @torch.no_grad()
    def forward(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """Encode texts and truncate to desired dimension."""
        if not texts:
            return torch.empty((0, self.embedding_dim), 
                             device=f"cuda:{torch.cuda.current_device()}")
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                prompt_name="query" if self.use_prompt else None,
                device=f"cuda:{torch.cuda.current_device()}",
                show_progress_bar=False
            )
            
            # TRUNCATE to desired dimension
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, :self.embedding_dim]
            
            return embeddings
        except Exception as e:
            print(f"Error encoding with Qwen: {e}")
            device_id = torch.cuda.current_device()
            return torch.zeros((len(texts), self.embedding_dim), 
                             device=f"cuda:{device_id}")

        
        
        
class T5Encoder(torch.nn.Module):
    """T5 Encoder for embedding extraction."""
    
    def __init__(self, model_name: str = "google/t5-v1_1-large"):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer
        
        # Add explicit path handling for T5
        try:
            # Try to load with explicit configuration
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name, 
                use_fast=False  # Use slow tokenizer for T5
            )
        except Exception as e:
            print(f"Warning: Could not load T5 tokenizer normally: {e}")
            # Try alternative loading
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                use_fast=False,
                legacy=True
            )
        
        # Load the encoder model
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        
        # Move to GPU
        device = f"cuda:{torch.cuda.current_device()}"
        self.encoder = self.encoder.to(device)
        
        self.embedding_dim = self.encoder.config.d_model
    
    @torch.no_grad()
    def forward(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts to latent vectors."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # T5 expects "summarize: " prefix for some versions, but encoder doesn't need it
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.encoder.device)
            
            outputs = self.encoder(**inputs)
            
            # Use mean pooling of last hidden states
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Optional: normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings)
        
        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            device = self.encoder.device
            return torch.empty((0, self.embedding_dim), device=device)

class SONAREncoder(torch.nn.Module):
    """SONAR Encoder for embedding extraction."""
    
    def __init__(self, model_path: str = "facebook/SONAR"):
        super().__init__()
        
        device = f"cuda:{torch.cuda.current_device()}"
        
        try:
            # Try to use the official SONAR library
            print(f"Loading SONAR model on {device}...")
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            
            self.pipeline = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=torch.device(device),
                dtype=torch.float16,
            )
            
            self.use_pipeline = True
            print("SONAR pipeline initialized successfully")
            
            # Test embedding dimension
            test_text = ["Test sentence"]
            test_emb = self.pipeline.predict(test_text, source_lang="eng_Latn", batch_size=1)
            if isinstance(test_emb, torch.Tensor):
                self.embedding_dim = test_emb.shape[1]
            else:
                self.embedding_dim = 1024  # Default SONAR dimension
                
        except Exception as e:
            print(f"Warning: Could not load SONAR pipeline: {e}")
            
            try:
                # Fallback to transformers
                print("Trying SONAR via transformers...")
                from transformers import AutoModel, AutoTokenizer
                
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                ).to(device)
                
                self.use_pipeline = False
                self.embedding_dim = self.model.config.hidden_size
                print(f"SONAR loaded via transformers with embedding dim: {self.embedding_dim}")
                
            except Exception as e2:
                print(f"Error loading SONAR: {e2}")
                print("Falling back to E5 as SONAR alternative...")
                
                # Ultimate fallback: use E5
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    "intfloat/multilingual-e5-large",
                    device=device
                )
                self.use_pipeline = False
                self.embedding_dim = 1024
                self.fallback_e5 = True
    
    @torch.no_grad()
    def forward(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts using SONAR."""
        if not texts:
            return torch.empty((0, self.embedding_dim), 
                             device=f"cuda:{torch.cuda.current_device()}")
        
        device = f"cuda:{torch.cuda.current_device()}"
        
        try:
            if hasattr(self, 'use_pipeline') and self.use_pipeline:
                # Use official SONAR pipeline
                embeddings = self.pipeline.predict(
                    texts,
                    source_lang="eng_Latn",
                    batch_size=min(batch_size, len(texts))
                )
                
                if not isinstance(embeddings, torch.Tensor):
                    embeddings = torch.tensor(embeddings, device=device)
                    
            elif hasattr(self, 'fallback_e5') and self.fallback_e5:
                # Use E5 fallback
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=min(batch_size, len(texts)),
                    device=device,
                    show_progress_bar=False
                )
                
            else:
                # Use transformers model
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).to(device)
                    
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    embeddings_batch = outputs.last_hidden_state.mean(dim=1)
                    embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                    
                    all_embeddings.append(embeddings_batch)
                
                if all_embeddings:
                    embeddings = torch.cat(all_embeddings, dim=0)
                else:
                    embeddings = torch.empty((0, self.embedding_dim), device=device)
            
            return embeddings
            
        except Exception as e:
            print(f"Error in SONAR forward: {e}")
            # Return zeros as fallback
            return torch.zeros((len(texts), self.embedding_dim), device=device)

# -----------------------------------------------------------------------------
# Simple Text Chunking
# -----------------------------------------------------------------------------

def split_text_fixed_length(text: str, max_chars: int = 1024, min_chars: int = 50, overlap: int = 64) -> List[str]:
    """Simple fixed-length text chunking."""
    if not text or len(text) < min_chars:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk = text[start:end]
        
        # Try to end at sentence boundary
        if end < text_length and max_chars > 100:
            boundary_chars = '.!?。！？\n\n'
            for i in range(min(100, len(chunk))):
                idx = len(chunk) - 1 - i
                if idx >= 0 and chunk[idx] in boundary_chars:
                    chunk = chunk[:idx+1]
                    break
        
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        
        if end == text_length:
            break
        start = end - overlap
    
    return chunks

# -----------------------------------------------------------------------------
# Dataset Loading
# -----------------------------------------------------------------------------

def load_dataset_simple(dataset_name: str, source_dir: Optional[str] = None, max_samples: Optional[int] = None):
    """Load dataset with minimal complexity."""
    if source_dir and os.path.exists(source_dir):
        # Try different formats
        source_path = Path(source_dir)
        
        # Check for arrow files
        arrow_files = list(source_path.glob("*.arrow"))
        if arrow_files:
            dataset = load_dataset("arrow", data_files=[str(f) for f in arrow_files], streaming=False)
            return dataset["train"]
        
        # Check for parquet files
        parquet_files = list(source_path.glob("*.parquet"))
        if parquet_files:
            dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], streaming=False)
            return dataset["train"]
        
        # Check for JSON/JSONL files
        json_files = list(source_path.glob("*.json*"))
        if json_files:
            dataset = load_dataset("json", data_files=[str(f) for f in json_files], streaming=False)
            return dataset["train"]
    
    # Download from HuggingFace
    if dataset_name == "openwebtext":
        try:
            dataset = load_dataset("Skylion007/openwebtext", streaming=False, trust_remote_code=True)
            return dataset["train"]
        except:
            dataset = load_dataset("openwebtext", trust_remote_code=True, streaming=False)
            return dataset["train"]
    elif dataset_name == "lm1b":
        dataset = load_dataset("lm1b", trust_remote_code=True, streaming=False)
        return dataset["train"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# -----------------------------------------------------------------------------
# Distributed Processor
# -----------------------------------------------------------------------------

class DistributedEmbeddingGenerator:
    """Generate embeddings with distributed processing."""
    
    def __init__(self, args, rank, world_size, local_rank):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        
        # Setup output directories
        self.output_dir = Path(args.output_dir)
        self.text_dir = self.output_dir / "texts" / "train"
        self.latent_dir = self.output_dir / "latents" / "train"
        
        if rank == 0:
            self.text_dir.mkdir(parents=True, exist_ok=True)
            self.latent_dir.mkdir(parents=True, exist_ok=True)
        
        # Sync directory creation
        if world_size > 1:
            dist.barrier()
        
        # Load dataset
        if rank == 0:
            print(f"\nLoading dataset: {args.dataset}")
        
        self.dataset = load_dataset_simple(args.dataset, args.source_dir, args.max_samples)
        
        if args.max_samples and hasattr(self.dataset, '__len__'):
            self.dataset = self.dataset.select(range(min(args.max_samples, len(self.dataset))))
        
        self.total_docs = len(self.dataset)
        
        # Each rank processes its own slice
        docs_per_rank = self.total_docs // world_size
        self.start_idx = rank * docs_per_rank
        self.end_idx = self.start_idx + docs_per_rank if rank < world_size - 1 else self.total_docs
        
        # Initialize model
        if args.model == "e5":
            model_path = args.model_path if args.model_path else "intfloat/multilingual-e5-large"
            self.model = E5Encoder(model_path)
            self.model_batch_size = args.batch_size
        elif args.model == "qwen":
            model_path = args.model_path if args.model_path else "Qwen/Qwen3-Embedding-8B"
            self.model = QwenEmbeddingEncoder(model_path, 
                                           embedding_dim=args.embedding_dim)
            self.model_batch_size = args.batch_size  # Qwen needs smaller batches
        elif args.model == "t5":
            model_path = args.model_path if args.model_path else "google/t5-v1_1-large"
            self.model = T5Encoder(model_path)
            self.model_batch_size = args.batch_size
        elif args.model == "sonar":
            model_path = args.model_path if args.model_path else "facebook/SONAR"
            self.model = SONAREncoder(model_path)
            self.model_batch_size = args.batch_size
        
        self.model = self.model.cuda(local_rank)
        self.model.eval()
        
        # Wrap with DDP if multi-GPU
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank
            )
        
        if rank == 0:
            print(f"Model initialized with embedding dimension: {self.model.module.embedding_dim if hasattr(self.model, 'module') else self.model.embedding_dim}")
            print(f"Batch size: {self.model_batch_size}")
            print(f"Each rank processing {self.end_idx - self.start_idx:,} documents")
    
    def process(self):
        """Process assigned documents with proper batching across chunks."""
        all_samples = []
        all_chunks = []
        chunk_metadata = []  # Track which document/chunk each belongs to
        total_chunks = 0
        start_time = time.time()
        
        # Only show progress bar on rank 0
        disable_progress = self.rank != 0
        doc_range = range(self.start_idx, self.end_idx)
        
        # First pass: collect all chunks
        for doc_idx in tqdm(doc_range, desc=f"Rank {self.rank} collecting", disable=disable_progress):
            try:
                item = self.dataset[doc_idx]
                text = item.get('text', '').strip()
                
                if not text or len(text) < self.args.min_chars:
                    continue
                
                # Split into chunks
                chunks = split_text_fixed_length(
                    text, 
                    max_chars=self.args.max_chars, 
                    min_chars=self.args.min_chars,
                    overlap=self.args.overlap
                )
                
                if not chunks:
                    continue
                
                # Store chunks and metadata
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        "doc_idx": doc_idx,
                        "chunk_idx": i,
                        "chunk": chunk
                    })
                    
            except Exception as e:
                print(f"Rank {self.rank}: Error processing document {doc_idx}: {e}")
                continue
        
        # Second pass: process all chunks in batches
        if all_chunks:
            print(f"Rank {self.rank}: Processing {len(all_chunks)} chunks in batches of {self.model_batch_size}")
            
            # Process in batches
            for batch_start in tqdm(range(0, len(all_chunks), self.model_batch_size), 
                                desc=f"Rank {self.rank} embedding", disable=disable_progress):
                batch_end = min(batch_start + self.model_batch_size, len(all_chunks))
                batch_chunks = all_chunks[batch_start:batch_end]
                batch_metadata = chunk_metadata[batch_start:batch_end]
                
                # Generate embeddings for this batch
                with torch.no_grad():
                    if self.args.model == "e5":
                        # E5 handles batching internally
                        embeddings = self.model(batch_chunks, batch_size=self.model_batch_size)
                    elif self.args.model == "qwen":
                        # Qwen handles batching internally
                        embeddings = self.model(batch_chunks, batch_size=min(8, self.model_batch_size))
                    elif self.args.model == "t5":
                        # T5 needs explicit batching
                        embeddings = self.model(batch_chunks, batch_size=self.model_batch_size)
                    elif self.args.model == "sonar":
                        # SONAR handles batching
                        embeddings = self.model(batch_chunks, batch_size=self.model_batch_size)
                    else:
                        # Default
                        embeddings = self.model(batch_chunks, batch_size=self.model_batch_size)
                
                # Save each chunk in the batch
                for i, (emb, meta) in enumerate(zip(embeddings, batch_metadata)):
                    doc_idx = meta["doc_idx"]
                    chunk_idx = meta["chunk_idx"]
                    chunk = meta["chunk"]
                    
                    # Create unique filename with rank info
                    text_hash = hashlib.md5(chunk.encode()).hexdigest()[:16]
                    filename = f"{self.args.dataset}_{doc_idx:08d}_{chunk_idx:03d}_{text_hash}"
                    
                    # Save text file
                    text_path = self.text_dir / f"{filename}.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(chunk)
                    
                    # Save latent file
                    latent_path = self.latent_dir / f"{filename}.npy"
                    # np.save(latent_path, emb.cpu().numpy())
                    np.save(latent_path, emb.float().cpu().numpy())
                    
                    # Store metadata
                    sample = {
                        "text": chunk,
                        "text_path": str(text_path.relative_to(self.output_dir)),
                        "latent_path": str(latent_path.relative_to(self.output_dir)),
                        "dataset": self.args.dataset,
                        "doc_id": doc_idx,
                        "chunk_id": chunk_idx,
                        "chunk_length": len(chunk),
                        "model": self.args.model,
                        "embedding_dim": self.model.module.embedding_dim if hasattr(self.model, 'module') else self.model.embedding_dim,
                        "rank": self.rank
                    }
                    
                    all_samples.append(sample)
                    total_chunks += 1
        
        # Print statistics for this rank
        elapsed = time.time() - start_time
        if total_chunks > 0:
            chunks_per_sec = total_chunks / elapsed
            print(f"Rank {self.rank}: Processed {self.end_idx - self.start_idx:,} docs, {total_chunks:,} chunks "
                f"({chunks_per_sec:.1f} chunks/sec)")
        
        return all_samples

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate text embeddings using E5/Qwen/T5/SONAR models with torchrun support."
    )
    
    # Required parameters
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--dataset", type=str, default="openwebtext",
                       choices=["openwebtext", "lm1b"],
                       help="Dataset to process")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="e5",
                       choices=["e5", "qwen", "t5", "sonar"],
                       help="Model to extract embeddings")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model (optional)")
    parser.add_argument("--embedding-dim", type=int, default=1024,
                       help="Embedding dimension for Qwen model")
    
    # Processing parameters
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding generation (affects model inference)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset")
    parser.add_argument("--max-chars", type=int, default=1024,
                       help="Maximum characters per text chunk")
    parser.add_argument("--min-chars", type=int, default=50,
                       help="Minimum characters per text chunk")
    parser.add_argument("--overlap", type=int, default=64,
                       help="Overlap between chunks in characters")
    
    # Dataset loading
    parser.add_argument("--source-dir", type=str, default=None,
                       help="Local directory to load dataset from")
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Generating {args.model} embeddings with {world_size} GPU(s)")
        print(f"Output directory: {args.output_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"Max chars per chunk: {args.max_chars}")
        print(f"{'='*60}")
    
    # Initialize generator
    generator = DistributedEmbeddingGenerator(args, rank, world_size, local_rank)
    
    # Process data
    all_samples = generator.process()
    
    # Gather all samples to rank 0
    if world_size > 1:
        # Convert samples to JSON string for gathering
        samples_json = json.dumps(all_samples, ensure_ascii=False)
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
    
    # Rank 0 saves final results
    if rank == 0:
        # Save metadata
        metadata_path = generator.output_dir / "train_data.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(combined_samples, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        total_chars = sum(s['chunk_length'] for s in combined_samples)
        avg_chars = total_chars / len(combined_samples) if combined_samples else 0
        
        stats = {
            "total_chunks": len(combined_samples),
            "dataset": args.dataset,
            "model": args.model,
            "embedding_dim": generator.model.module.embedding_dim if hasattr(generator.model, 'module') else generator.model.embedding_dim,
            "batch_size": generator.model_batch_size,
            "num_gpus": world_size,
            "max_chars_per_chunk": args.max_chars,
            "min_chars_per_chunk": args.min_chars,
            "avg_chars_per_chunk": round(avg_chars, 1),
            "total_characters": total_chars,
            "processing_time_seconds": round(total_time, 2),
            "chunks_per_second": round(len(combined_samples) / total_time, 2) if total_time > 0 else 0,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(generator.output_dir / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("EMBEDDING GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total chunks: {len(combined_samples)}")
        print(f"Average chunk length: {avg_chars:.1f} characters")
        print(f"Processing time: {total_time/3600:.2f} hours")
        print(f"Speed: {len(combined_samples)/total_time:.2f} chunks/second" if total_time > 0 else "Speed: N/A")
        print(f"Output directory: {generator.output_dir}")
        print(f"  - Texts: {generator.text_dir}")
        print(f"  - Latents: {generator.latent_dir}")
        print(f"  - Metadata: {metadata_path}")
        print(f"{'='*60}")
        
        print(f"\nNext step: Run tokenization with:")
        print(f"python tokenize_parallel.py --json_path {metadata_path} --output_dir {generator.output_dir}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()