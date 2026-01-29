#!/usr/bin/env python
"""Generate BIG token files from text data in parallel."""

import json
import pickle
from pathlib import Path
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import argparse
import gc
import torch

# ========== CRITICAL: Set this BEFORE any imports ==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize_worker(args):
    """Worker function for parallel tokenization"""
    chunk, tokenizer_path, max_length, worker_id = args
    tokenizer_path = Path(tokenizer_path)
    
    # Import tokenizer INSIDE worker (critical!)
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    results = []
    for item in chunk:
        text = item.get('text', '')
        
        # Skip empty text
        if not text or len(text.strip()) < 10:
            continue

        # Tokenize
        try:
            tokenized = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='np'
            )
            
            results.append({
                'input_ids': tokenized['input_ids'].astype(np.int32)[0],
                'attention_mask': tokenized['attention_mask'].astype(np.uint8)[0],  # uint8节省空间
                'text': text[:500],  # 保存前500字符用于调试
                'text_path': item.get('text_path', ''),
                'latent_path': item.get('latent_path', ''),
                'doc_id': item.get('doc_id', -1),
                'chunk_id': item.get('chunk_id', -1),
                'chunk_length': item.get('chunk_length', 0),
                'latent_file': item.get('latent_file', ''),  # 对应的大文件
                'latent_idx': item.get('latent_idx', -1)     # 在大文件中的索引
            })
            
        except Exception as e:
            print(f"Worker {worker_id}: Error tokenizing text: {e}")
            continue

    return results

def create_unified_token_files(json_path, tokenizer_path, output_dir, 
                              max_length=512, num_workers=32, 
                              samples_per_file=100000):
    """
    创建统一的大token文件，与latent文件对应
    
    输出格式:
    - tokens_batch_{rank}_{idx}.npy: [num_samples, 2, max_length]
    - tokens_meta_batch_{rank}_{idx}.json: 元数据
    """
    
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    token_dir = output_dir / "tokens" / "train"
    token_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== UNIFIED TOKENIZATION ===")
    print(f"Input JSON: {json_path}")
    print(f"Output dir: {token_dir}")
    print(f"Max length: {max_length}")
    print(f"Workers: {num_workers}")
    print(f"Samples per file: {samples_per_file:,}")
    
    # 1. 加载数据
    print(f"\nLoading data from {json_path}...")
    start_time = time.time()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data):,} samples in {time.time()-start_time:.1f}s")
    
    # 2. 根据latent文件分组数据
    print("\nGrouping data by latent files...")
    data_by_latent_file = {}
    
    for item in data:
        latent_file = item.get('latent_file', '')
        if not latent_file:
            # 如果没有latent_file字段，从latent_path推断
            latent_path = item.get('latent_path', '')
            if latent_path:
                latent_file = Path(latent_path).name
            else:
                latent_file = 'unknown'
        
        if latent_file not in data_by_latent_file:
            data_by_latent_file[latent_file] = []
        
        # 添加latent_idx（在大文件中的位置）
        item['latent_file'] = latent_file
        item['latent_idx'] = len(data_by_latent_file[latent_file])
        
        data_by_latent_file[latent_file].append(item)
    
    print(f"Grouped into {len(data_by_latent_file)} latent file groups")
    
    # 3. 处理每个latent文件组
    all_saved_files = []
    
    for latent_file_idx, (latent_file_name, file_data) in enumerate(data_by_latent_file.items()):
        print(f"\nProcessing latent file {latent_file_name} "
              f"({len(file_data):,} samples)...")
        
        # 如果样本太多，分割成多个token文件
        num_token_files = (len(file_data) + samples_per_file - 1) // samples_per_file
        
        for token_file_idx in range(num_token_files):
            start_idx = token_file_idx * samples_per_file
            end_idx = min((token_file_idx + 1) * samples_per_file, len(file_data))
            
            file_data_chunk = file_data[start_idx:end_idx]
            
            if not file_data_chunk:
                continue
            
            # 3.1 并行tokenize这个chunk
            print(f"  Token file {token_file_idx+1}/{num_token_files}: "
                  f"{len(file_data_chunk):,} samples")
            
            # 创建worker chunks
            chunk_size = max(100, len(file_data_chunk) // (num_workers * 2))
            chunks = []
            for i in range(0, len(file_data_chunk), chunk_size):
                chunks.append(file_data_chunk[i:i+chunk_size])
            
            worker_args = [(chunk, tokenizer_path, max_length, i) 
                          for i, chunk in enumerate(chunks)]
            
            # 并行处理
            tokenized_data = []
            with mp.Pool(processes=min(num_workers, len(chunks)), 
                        maxtasksperchild=10) as pool:
                try:
                    with tqdm(total=len(chunks), desc=f"Tokenizing", leave=False) as pbar:
                        for result in pool.imap_unordered(tokenize_worker, worker_args, chunksize=1):
                            tokenized_data.extend(result)
                            pbar.update(1)
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
                    pool.terminate()
                    pool.join()
                    raise
            
            if not tokenized_data:
                print(f"  Warning: No tokenized data for this chunk")
                continue
            
            # 3.2 创建统一的数组
            num_samples = len(tokenized_data)
            print(f"  Creating unified array: {num_samples} samples")
            
            # [num_samples, 2, max_length]
            # 维度0: input_ids, 维度1: attention_mask
            token_array = np.zeros((num_samples, 2, max_length), dtype=np.int32)
            metadata = []
            
            for i, item in enumerate(tokenized_data):
                token_array[i, 0, :] = item['input_ids']      # input_ids
                token_array[i, 1, :] = item['attention_mask'] # attention_mask
                
                # 保存精简的元数据
                metadata.append({
                    'text': item['text'],
                    'text_path': item['text_path'],
                    'latent_path': item['latent_path'],
                    'latent_file': item['latent_file'],
                    'latent_idx': item['latent_idx'],
                    'doc_id': item['doc_id'],
                    'chunk_id': item['chunk_id'],
                    'chunk_length': item['chunk_length'],
                    'token_idx': i,
                    'token_file_idx': token_file_idx
                })
            
            # 3.3 保存文件
            # 文件名格式: tokens_latent{latent_file_idx}_batch{token_file_idx}.npy
            base_name = f"tokens_latent{latent_file_idx:04d}_batch{token_file_idx:04d}"
            token_file = token_dir / f"{base_name}.npy"
            meta_file = token_dir / f"{base_name}_meta.json"
            
            # 保存token数组（未压缩，加载最快）
            np.save(token_file, token_array)
            
            # 保存元数据
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            all_saved_files.append({
                'token_file': str(token_file),
                'meta_file': str(meta_file),
                'latent_file': latent_file_name,
                'latent_file_idx': latent_file_idx,
                'token_file_idx': token_file_idx,
                'num_samples': num_samples,
                'shape': token_array.shape
            })
            
            print(f"  ✓ Saved {token_file.name}: {num_samples:,} samples")
            
            # 清理内存
            del token_array
            del tokenized_data
            gc.collect()
    
    # 4. 创建全局索引
    print(f"\nCreating global index...")
    create_global_index(token_dir, all_saved_files, max_length)
    
    # 5. 保存tokenizer信息
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    tokenizer_info = {
        'vocab_size': tokenizer.vocab_size,
        'max_length': max_length,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'model_name': str(tokenizer_path),
        'total_samples': len(data),
        'total_token_files': len(all_saved_files),
        'samples_per_file': samples_per_file
    }
    
    with open(token_dir / "tokenizer_info.json", 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    # 性能统计
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("✅ UNIFIED TOKENIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   Total samples: {len(data):,}")
    print(f"   Total token files: {len(all_saved_files)}")
    print(f"   Samples per file: ~{samples_per_file:,}")
    print(f"   Speed: {len(data)/total_time:.0f} samples/sec")
    print(f"   Output directory: {token_dir}")
    print(f"{'='*60}")
    
    return all_saved_files

def create_global_index(token_dir, saved_files, max_length):
    """创建全局索引文件以便快速查找"""
    
    index = {
        'files': [],
        'total_samples': 0,
        'max_length': max_length,
        'file_mapping': {}  # latent_file -> [token_files]
    }
    
    # 按latent文件分组
    for file_info in saved_files:
        latent_file = file_info['latent_file']
        if latent_file not in index['file_mapping']:
            index['file_mapping'][latent_file] = []
        
        index['file_mapping'][latent_file].append({
            'token_file': Path(file_info['token_file']).name,
            'meta_file': Path(file_info['meta_file']).name,
            'num_samples': file_info['num_samples'],
            'latent_file_idx': file_info['latent_file_idx'],
            'token_file_idx': file_info['token_file_idx']
        })
        
        index['files'].append({
            'token_file': Path(file_info['token_file']).name,
            'meta_file': Path(file_info['meta_file']).name,
            'num_samples': file_info['num_samples'],
            'shape': file_info['shape']
        })
        
        index['total_samples'] += file_info['num_samples']
    
    # 保存索引
    index_path = token_dir / "global_index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Global index created: {index_path}")
    print(f"  - {len(index['files'])} token files")
    print(f"  - {index['total_samples']:,} total samples")
    print(f"  - {len(index['file_mapping'])} latent files mapped")

class UnifiedTokenDataset(torch.utils.data.Dataset):
    """加载统一大文件格式的token数据"""
    
    def __init__(self, token_dir, max_samples=None, verbose=True):
        self.token_dir = Path(token_dir)
        
        # 加载全局索引
        index_path = self.token_dir / "global_index.json"
        if not index_path.exists():
            raise ValueError(f"Global index not found: {index_path}")
        
        with open(index_path, 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        
        # 创建文件信息列表
        self.file_info = []
        self.total_samples = 0
        
        for file_entry in self.index['files']:
            token_file = self.token_dir / file_entry['token_file']
            meta_file = self.token_dir / file_entry['meta_file']
            
            if not token_file.exists():
                print(f"Warning: Token file not found: {token_file}")
                continue
            
            self.file_info.append({
                'token_file': token_file,
                'meta_file': meta_file,
                'start_idx': self.total_samples,
                'end_idx': self.total_samples + file_entry['num_samples'],
                'num_samples': file_entry['num_samples'],
                'shape': file_entry['shape']
            })
            
            self.total_samples += file_entry['num_samples']
        
        # 限制最大样本数
        if max_samples and max_samples < self.total_samples:
            self.total_samples = max_samples
        
        # 预加载元数据（可选）
        self.metadata_cache = {}
        self.cache_size = 5  # 缓存最近访问的5个文件的元数据
        
        if verbose:
            print(f"\nUnifiedTokenDataset 统计:")
            print(f"  文件数量: {len(self.file_info)}")
            print(f"  总样本数: {self.total_samples:,}")
            print(f"  序列长度: {self.index['max_length']}")
    
    def _load_metadata(self, meta_file):
        """加载元数据，带缓存"""
        meta_file = Path(meta_file)
        
        if str(meta_file) in self.metadata_cache:
            return self.metadata_cache[str(meta_file)]
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 添加到缓存
        self.metadata_cache[str(meta_file)] = metadata
        
        # 限制缓存大小
        if len(self.metadata_cache) > self.cache_size:
            # 移除最久未使用的
            oldest_key = next(iter(self.metadata_cache))
            del self.metadata_cache[oldest_key]
        
        return metadata
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        if idx >= self.total_samples:
            idx = idx % self.total_samples
        
        # 找到对应的文件
        for info in self.file_info:
            if idx < info['end_idx']:
                local_idx = idx - info['start_idx']
                
                try:
                    # 1. 加载token数据（内存映射）
                    token_array = np.load(info['token_file'], mmap_mode='r')
                    
                    # 形状: [num_samples, 2, max_length]
                    input_ids = torch.from_numpy(token_array[local_idx, 0, :]).long()
                    attention_mask = torch.from_numpy(token_array[local_idx, 1, :]).bool()
                    
                    # 2. 加载元数据
                    metadata = self._load_metadata(info['meta_file'])
                    meta = metadata[local_idx] if local_idx < len(metadata) else {}
                    
                    return {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'text': meta.get('text', ''),
                        'latent_file': meta.get('latent_file', ''),
                        'latent_idx': meta.get('latent_idx', -1),
                        'doc_id': meta.get('doc_id', -1),
                        'chunk_id': meta.get('chunk_id', -1),
                        'global_idx': idx,
                        'local_idx': local_idx,
                        'token_file': info['token_file'].name
                    }
                    
                except Exception as e:
                    print(f"Error loading sample {idx}: {e}")
                    # 返回空样本
                    max_length = self.index['max_length']
                    return {
                        'input_ids': torch.zeros(max_length, dtype=torch.long),
                        'attention_mask': torch.ones(max_length, dtype=torch.bool),
                        'text': f"Error sample {idx}",
                        'global_idx': idx,
                        'local_idx': -1
                    }
        
        # Fallback
        max_length = self.index['max_length']
        return {
            'input_ids': torch.zeros(max_length, dtype=torch.long),
            'attention_mask': torch.ones(max_length, dtype=torch.bool),
            'text': f"Fallback sample {idx}",
            'global_idx': idx,
            'local_idx': -1
        }

if __name__ == "__main__":
    # THIS IS CRITICAL FOR MULTIPROCESSING
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Generate unified big token files")
    parser.add_argument("--json_path", type=str, required=True,
                       help="Path to input JSON file (from embedding generation)")
    parser.add_argument("--tokenizer", type=str,
                       default="/path/to/tokenizer",
                       help="Tokenizer path")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str,
                       default="./output",
                       help="Output directory")
    parser.add_argument("--workers", type=int, default=32,
                       help="Number of worker processes")
    parser.add_argument("--samples_per_file", type=int, default=100000,
                       help="Samples per big file (推荐: 50000-200000)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行tokenization
    saved_files = create_unified_token_files(
        json_path=args.json_path,
        tokenizer_path=args.tokenizer,
        output_dir=output_dir,
        max_length=args.max_length,
        num_workers=args.workers,
        samples_per_file=args.samples_per_file
    )
    
    print(f"\n生成完成！")
    print(f"输出目录: {output_dir}/tokens/train/")
    print(f"共生成 {len(saved_files)} 个大文件")