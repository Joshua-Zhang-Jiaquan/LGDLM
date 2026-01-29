#!/usr/bin/env python
"""ULTRA-OPTIMIZED PARALLEL tokenization for 1B+ TXT files - NO JSON."""

import json
from pathlib import Path
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import argparse

# ========== CRITICAL: Set this BEFORE any imports ==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize_worker(args):
    """Worker function for parallel tokenization - optimized"""
    chunk, tokenizer_path, max_length = args
    
    # Import tokenizer INSIDE worker (critical!)
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    results = []
    for item in chunk:
        text = item['text']
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'
        )

        results.append({
            'input_ids': tokenized['input_ids'].astype(np.int16)[0],  # Use int16 to save space
            'attention_mask': tokenized['attention_mask'].astype(np.bool_)[0],
            'source_file': item['source_file'],
            'sample_id': item['sample_id']
        })

    return results

def stream_txt_files(data_dir, max_samples=None):
    """Generator to stream TXT files without loading all at once"""
    data_dir = Path(data_dir)
    txt_files = sorted(list(data_dir.glob("*.txt")))
    
    if not txt_files:
        txt_files = sorted(list(data_dir.rglob("*.txt")))
    
    print(f"🔍 Found {len(txt_files):,} .txt files")
    
    sample_id = 0
    files_processed = 0
    
    for txt_file in txt_files:
        try:
            # Read text file
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            
            if text:  # Skip empty files
                yield {
                    'text': text,
                    'source_file': str(txt_file),
                    'sample_id': sample_id
                }
                sample_id += 1
                files_processed += 1
                
                if max_samples and sample_id >= max_samples:
                    return
                        
        except Exception as e:
            print(f"⚠️ Warning: Could not load {txt_file}: {e}")
            continue
    
    print(f"📊 Total files processed: {files_processed:,}")
    print(f"📊 Total samples found: {sample_id:,}")

def save_tokens_batch(token_dir, batch_results, batch_num):
    """Save a batch of tokenized results efficiently"""
    batch_dir = token_dir / f"batch_{batch_num // 1000:04d}"
    batch_dir.mkdir(exist_ok=True)
    
    for result in batch_results:
        # Save as .npy for maximum speed and space efficiency
        token_path = batch_dir / f"tokens_{result['sample_id']:012d}.npy"
        
        # Save as a dictionary in .npy format
        np.save(
            token_path,
            {
                'input_ids': result['input_ids'],
                'attention_mask': result['attention_mask'],
                'source': result['source_file'],
                'sample_id': result['sample_id']
            },
            allow_pickle=True
        )
    
    return len(batch_results)

def pre_tokenize_txt_parallel(data_dir, tokenizer_path, max_length=512, 
                             output_dir=None, num_workers=127, 
                             chunk_size=1000, max_samples=None):
    """PARALLEL pre-tokenization of TXT files using all CPU cores - NO JSON."""
    
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir.parent / f"tokenized_{int(time.time())}"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create token directory with batch subdirectories
    token_dir = output_dir / "tokens"
    token_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("🚀 PARALLEL TOKENIZATION OF TXT FILES - NO JSON")
    print("=" * 80)
    print(f"📁 Source directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"⚙️  Tokenizer: {tokenizer_path}")
    print(f"📏 Max length: {max_length}")
    print(f"👷 Workers: {num_workers}")
    print(f"📦 Chunk size: {chunk_size:,}")
    print(f"🔢 Max samples: {max_samples or 'ALL'}")
    print("=" * 80)
    
    # Stream data from TXT files
    print("\n📂 Streaming data from TXT files...")
    data_stream = stream_txt_files(data_dir, max_samples)
    
    # Prepare data for parallel processing
    print("\n🗂️ Preparing data chunks for parallel processing...")
    chunks = []
    current_chunk = []
    total_samples = 0
    
    for item in data_stream:
        current_chunk.append(item)
        total_samples += 1
        
        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
    
    # Add the last chunk if any
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"📊 Created {len(chunks):,} chunks for parallel processing")
    print(f"📊 Total samples to process: {total_samples:,}")
    
    # Prepare arguments for workers
    worker_args = [(chunk, tokenizer_path, max_length) for chunk in chunks]
    
    # PARALLEL PROCESSING
    print(f"\n⚡ Starting parallel tokenization with {num_workers} workers...")
    print("💡 Monitor CPU usage with: top -b -n 1 | grep python")
    
    start_time = time.time()
    batch_num = 0
    stats_file = output_dir / "processing_stats.txt"
    
    # Use multiprocessing Pool
    with mp.Pool(
        processes=num_workers,
        maxtasksperchild=200  # Refresh workers to avoid memory leaks
    ) as pool:
        try:
            # Process chunks in parallel with progress bar
            with tqdm(total=len(chunks), desc="Tokenizing chunks") as pbar:
                for result in pool.imap_unordered(tokenize_worker, worker_args, chunksize=1):
                    # Save results immediately
                    saved_count = save_tokens_batch(token_dir, result, batch_num)
                    batch_num += 1
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Calculate and display stats
                    elapsed = time.time() - start_time
                    processed_samples = min(batch_num * chunk_size, total_samples)
                    rate = processed_samples / elapsed if elapsed > 0 else 0
                    
                    pbar.set_postfix({
                        "samples": f"{processed_samples:,}",
                        "speed": f"{rate:,.0f}/s",
                        "batches": f"{batch_num:,}"
                    })
                    
                    # Save stats every 100 chunks
                    if batch_num % 100 == 0:
                        with open(stats_file, 'w') as f:
                            f.write(f"Total samples processed: {processed_samples:,}\n")
                            f.write(f"Processing rate: {rate:,.0f} samples/sec\n")
                            f.write(f"Elapsed time: {elapsed:.1f}s ({elapsed/3600:.1f}h)\n")
                            f.write(f"Batches completed: {batch_num:,}\n")
                            f.write(f"Last update: {time.ctime()}\n")
        
        except KeyboardInterrupt:
            print("\n⚠️ INTERRUPTED by user!")
            print(f"💾 Saving partial results...")
            pool.terminate()
            pool.join()
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            pool.terminate()
            pool.join()
            raise
    
    # Final statistics
    total_time = time.time() - start_time
    final_rate = total_samples / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 80)
    print("✅ PARALLEL TOKENIZATION COMPLETE!")
    print("=" * 80)
    print(f"📊 Total samples processed: {total_samples:,}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"⚡ Average rate: {final_rate:,.0f} samples/sec")
    print(f"👷 CPU cores used: {num_workers}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🗂️  Token files: {token_dir}/batch_*/tokens_*.npy")
    print("=" * 80)
    
    # Save final minimal info (not JSON)
    with open(output_dir / "COMPLETION_INFO.txt", 'w') as f:
        f.write(f"PARALLEL TOKENIZATION COMPLETED SUCCESSFULLY\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Total samples: {total_samples:,}\n")
        f.write(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)\n")
        f.write(f"Average rate: {final_rate:,.0f} samples/sec\n")
        f.write(f"CPU cores used: {num_workers}\n")
        f.write(f"Chunk size: {chunk_size}\n")
        f.write(f"Tokenizer: {tokenizer_path}\n")
        f.write(f"Max length: {max_length}\n")
        f.write(f"Source directory: {data_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
    
    return output_dir

if __name__ == "__main__":
    # THIS IS CRITICAL FOR MULTIPROCESSING
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Parallel tokenization of TXT files for 1B+ samples - NO JSON")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory with TXT files")
    parser.add_argument("--tokenizer", type=str, required=True,
                       help="Tokenizer path")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: data_dir/../tokenized_TIMESTAMP)")
    parser.add_argument("--workers", type=int, default=127,
                       help="Number of worker processes (use 127 for 128-core CPU)")
    parser.add_argument("--chunk_size", type=int, default=1000,
                       help="Work chunk size for parallel processing")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Limit samples for testing (e.g., 1000000)")
    
    args = parser.parse_args()
    
    # Run tokenization
    pre_tokenize_txt_parallel(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        max_length=args.max_length,
        output_dir=args.output_dir,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples
    )