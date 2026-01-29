#!/usr/bin/env python
"""使用内存映射的快速大文件生成器 - 无需h5py"""

import argparse
import json
from pathlib import Path
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
import time
from tqdm import tqdm
from datasets import load_dataset
import gc
from typing import List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_distributed():
    """Setup distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # 检查可用GPU
        available_gpus = torch.cuda.device_count()
        if local_rank >= available_gpus:
            print(f"Warning: local_rank {local_rank} >= available GPUs {available_gpus}")
            local_rank = local_rank % available_gpus
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        
        print(f"Rank {rank}/{world_size}, local_rank {local_rank}, device cuda:{local_rank}")
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

class MemmapBigFileGenerator:
    """使用内存映射的高效大文件生成器"""
    
    def __init__(self, args, rank, world_size, local_rank):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.base_name = f"train_rank{rank}"
        
        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if world_size > 1:
            dist.barrier()
        
        # 加载数据集
        print(f"Rank {rank}: Loading dataset...")
        self.dataset = self._load_dataset()
        
        # 分配数据给各个rank
        total_docs = len(self.dataset)
        docs_per_rank = total_docs // world_size
        self.start_idx = rank * docs_per_rank
        self.end_idx = self.start_idx + docs_per_rank if rank < world_size - 1 else total_docs
        
        # 加载模型
        print(f"Rank {rank}: Loading model...")
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # 初始化内存映射文件
        self._init_memmap_files()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Memory-Mapped Big File Generator")
            print(f"{'='*60}")
            print(f"Output: {self.output_dir}/{self.base_name}_*.npy")
            print(f"Model: {args.model}")
            print(f"Model path: {args.model_path}")
            print(f"Embedding dim: {args.embedding_dim}")
            print(f"Max length: {args.max_length}")
            print(f"Batch size: {args.batch_size}")
            print(f"Max chars: {args.max_chars}")
            print(f"World size: {world_size}")
            print(f"Total docs: {total_docs:,}")
            print(f"Docs per rank: {self.end_idx - self.start_idx:,}")
            print(f"{'='*60}")
    
    def _load_dataset(self):
        """加载数据集"""
        source_path = Path(self.args.source_dir) if self.args.source_dir else None
        
        if source_path and source_path.exists():
            print(f"Loading dataset from: {source_path}")
            
            # 尝试加载为datasets格式
            try:
                # 找parquet文件
                parquet_files = list(source_path.glob("*.parquet"))
                if parquet_files:
                    print(f"Found {len(parquet_files)} parquet files")
                    dataset = load_dataset("parquet", 
                                         data_files=[str(f) for f in parquet_files],
                                         split="train",
                                         num_proc=4)
                    print(f"Parquet dataset loaded: {len(dataset)} samples")
                    return dataset
                
                # 找arrow文件
                arrow_files = list(source_path.glob("*.arrow"))
                if arrow_files:
                    print(f"Found {len(arrow_files)} arrow files")
                    dataset = load_dataset("arrow", 
                                         data_files=[str(f) for f in arrow_files],
                                         split="train",
                                         num_proc=4)
                    print(f"Arrow dataset loaded: {len(dataset)} samples")
                    return dataset
                
                print("No supported dataset files found in source directory")
                
            except Exception as e:
                print(f"Error loading dataset from {source_path}: {e}")
        
        # 如果指定了数据集名称，尝试从huggingface下载
        if self.args.dataset:
            try:
                print(f"Loading dataset from HuggingFace: {self.args.dataset}")
                dataset = load_dataset(self.args.dataset, 
                                     split="train",
                                     num_proc=4,
                                     streaming=False)
                print(f"Loaded {len(dataset)} samples from {self.args.dataset}")
                return dataset
            except Exception as e:
                print(f"Error loading HuggingFace dataset: {e}")
        
        # 默认创建测试数据
        print("Creating sample data for testing...")
        from datasets import Dataset
        sample_texts = [{"text": f"This is sample text {i} for testing." * 10} 
                       for i in range(1000)]
        return Dataset.from_list(sample_texts)
    
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        model_path = self.args.model_path
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer
            
            print(f"Loading model from: {model_path}")
            
            # 加载SentenceTransformer模型
            model = SentenceTransformer(
                model_path,
                device=f"cuda:{self.local_rank}",
                trust_remote_code=True
            )
            
            # 获取tokenizer
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
            else:
                # 手动加载tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    local_files_only=True,
                    trust_remote_code=True
                )
            
            # 设置embedding维度
            actual_dim = model.get_sentence_embedding_dimension()
            target_dim = min(self.args.embedding_dim, actual_dim)
            model.embedding_dim = target_dim
            
            print(f"Model loaded, embedding dim: {target_dim} (actual: {actual_dim})")
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
            
            print(f"Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model/tokenizer: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _init_memmap_files(self):
        """初始化内存映射文件"""
        # 估算最大样本数（每文档约2-4个chunk）
        max_samples = (self.end_idx - self.start_idx) * 4
        
        print(f"Rank {self.rank}: Initializing memmap files for up to {max_samples:,} samples")
        
        # tokens内存映射文件
        self.tokens_file = self.output_dir / f"{self.base_name}_tokens.dat"
        self.tokens_shape = (max_samples, 2, self.args.max_length)
        self.tokens_mmap = np.memmap(
            self.tokens_file,
            dtype=np.int32,
            mode='w+',
            shape=self.tokens_shape
        )
        
        # latents内存映射文件
        self.latents_file = self.output_dir / f"{self.base_name}_latents.dat"
        self.latents_shape = (max_samples, self.args.embedding_dim)
        self.latents_mmap = np.memmap(
            self.latents_file,
            dtype=np.float32,
            mode='w+',
            shape=self.latents_shape
        )
        
        # 文本文件（单独存储）
        self.text_file = self.output_dir / f"{self.base_name}_texts.txt"
        self.text_fp = open(self.text_file, 'w', encoding='utf-8')
        
        # 索引文件
        self.index_file = self.output_dir / f"{self.base_name}_index.npy"
        
        # 当前写入位置
        self.current_idx = 0
        self.text_indices = []
    
    def process(self):
        """主处理流程"""
        print(f"Rank {self.rank}: Processing {self.end_idx - self.start_idx} documents...")
        
        # 处理文档
        pbar = tqdm(range(self.start_idx, self.end_idx), 
                   desc=f"Rank {self.rank} processing",
                   position=self.rank,
                   leave=False)
        
        for doc_idx in pbar:
            if doc_idx >= len(self.dataset):
                break
            
            try:
                item = self.dataset[doc_idx]
                text = item.get('text', '').strip()
                
                if not text or len(text) < self.args.min_chars:
                    continue
                
                # 分割文本
                chunks = self._split_text(text)
                
                for chunk in chunks:
                    # Tokenize
                    tokenized = self.tokenizer(
                        chunk,
                        truncation=True,
                        padding='max_length',
                        max_length=self.args.max_length,
                        return_tensors='np'
                    )
                    
                    # 保存tokens
                    if self.current_idx >= self.tokens_shape[0]:
                        # 扩展内存映射文件
                        self._expand_memmap()
                    
                    self.tokens_mmap[self.current_idx, 0, :] = tokenized['input_ids'].astype(np.int32)[0]
                    self.tokens_mmap[self.current_idx, 1, :] = tokenized['attention_mask'].astype(np.uint8)[0]
                    
                    # 保存文本和索引
                    self.text_fp.write(chunk + "\n")
                    self.text_indices.append(self.current_idx)
                    
                    self.current_idx += 1
                    
                    # 批量处理embedding
                    if len(self.text_indices) >= self.args.batch_size:
                        self._process_batch()
                    
                    # 更新进度条
                    pbar.set_postfix({'samples': self.current_idx})
                    
                    # 定期刷新
                    if self.current_idx % 10000 == 0:
                        self.tokens_mmap.flush()
                        self.text_fp.flush()
                        gc.collect()
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                print(f"\nRank {self.rank}: Error processing document {doc_idx}: {e}")
                continue
        
        pbar.close()
        
        # 处理最后一批
        if self.text_indices:
            self._process_batch()
        
        # 生成embedding
        print(f"\nRank {self.rank}: Generating embeddings for {self.current_idx} samples...")
        self._generate_embeddings()
        
        # 保存索引
        self._save_index()
        
        # 关闭文件
        self.text_fp.close()
        del self.tokens_mmap
        del self.latents_mmap
        
        print(f"\n✅ Rank {self.rank}: Saved {self.current_idx} samples")
        
        # 收集所有rank的信息
        if self.world_size > 1:
            total_counts = [0] * self.world_size
            dist.gather_object(self.current_idx, total_counts if self.rank == 0 else None, dst=0)
            
            if self.rank == 0:
                total_all = sum(total_counts)
                print(f"\n🎉 All ranks completed! Total samples: {total_all:,}")
                
                # 创建全局索引
                self._create_global_index(total_counts)
        
        return self.current_idx
    
    def _expand_memmap(self):
        """扩展内存映射文件"""
        print(f"Rank {self.rank}: Expanding memmap files...")
        
        # 扩展tokens
        new_tokens_shape = (self.tokens_shape[0] * 2, 2, self.args.max_length)
        new_tokens_file = self.output_dir / f"{self.base_name}_tokens_expanded.dat"
        
        # 创建新文件并复制数据
        new_tokens_mmap = np.memmap(
            new_tokens_file,
            dtype=np.int32,
            mode='w+',
            shape=new_tokens_shape
        )
        new_tokens_mmap[:self.tokens_shape[0]] = self.tokens_mmap
        
        # 更新引用
        self.tokens_mmap = new_tokens_mmap
        self.tokens_shape = new_tokens_shape
        
        # 扩展latents
        new_latents_shape = (self.latents_shape[0] * 2, self.args.embedding_dim)
        new_latents_file = self.output_dir / f"{self.base_name}_latents_expanded.dat"
        
        new_latents_mmap = np.memmap(
            new_latents_file,
            dtype=np.float32,
            mode='w+',
            shape=new_latents_shape
        )
        if hasattr(self, 'latents_written'):
            new_latents_mmap[:self.latents_shape[0]] = self.latents_mmap
        
        self.latents_mmap = new_latents_mmap
        self.latents_shape = new_latents_shape
        
        print(f"Rank {self.rank}: Expanded to {new_tokens_shape[0]:,} samples")
    
    def _process_batch(self):
        """处理一批文本生成embedding"""
        if not self.text_indices:
            return
        
        # 读取最近的文本
        start_idx = max(0, self.current_idx - len(self.text_indices))
        text_indices_to_process = self.text_indices.copy()
        self.text_indices = []  # 清空当前批次
        
        # 读取文本
        with open(self.text_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        batch_texts = []
        for idx in text_indices_to_process:
            if idx < len(all_lines):
                batch_texts.append(all_lines[idx].strip())
        
        if not batch_texts:
            return
        
        # 生成embedding
        with torch.no_grad():
            embeddings = self.model.encode(
                batch_texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                batch_size=len(batch_texts),
                device=f"cuda:{self.local_rank}",
                show_progress_bar=False
            )
        
        # 保存到内存映射
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        if embeddings_np.shape[1] > self.args.embedding_dim:
            embeddings_np = embeddings_np[:, :self.args.embedding_dim]
        
        for i, idx in enumerate(text_indices_to_process):
            if idx < self.latents_shape[0]:
                self.latents_mmap[idx, :] = embeddings_np[i]
        
        self.latents_mmap.flush()
        torch.cuda.empty_cache()
    
    def _generate_embeddings(self):
        """生成剩余的所有embedding"""
        if not hasattr(self, 'latents_written'):
            self.latents_written = 0
        
        remaining = self.current_idx - self.latents_written
        if remaining <= 0:
            return
        
        print(f"Rank {self.rank}: Generating {remaining} remaining embeddings...")
        
        # 读取文本
        with open(self.text_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        batch_size = self.args.batch_size
        
        pbar = tqdm(range(self.latents_written, self.current_idx, batch_size),
                   desc=f"Rank {self.rank} generating embeddings",
                   position=self.rank + self.world_size,
                   leave=False)
        
        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, self.current_idx)
            batch_indices = list(range(start_idx, end_idx))
            
            batch_texts = []
            for idx in batch_indices:
                if idx < len(all_lines):
                    batch_texts.append(all_lines[idx].strip())
            
            if not batch_texts:
                continue
            
            # 生成embedding
            with torch.no_grad():
                embeddings = self.model.encode(
                    batch_texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=len(batch_texts),
                    device=f"cuda:{self.local_rank}",
                    show_progress_bar=False
                )
            
            # 保存
            embeddings_np = embeddings.cpu().numpy().astype(np.float32)
            if embeddings_np.shape[1] > self.args.embedding_dim:
                embeddings_np = embeddings_np[:, :self.args.embedding_dim]
            
            for i, idx in enumerate(batch_indices):
                if idx < self.latents_shape[0]:
                    self.latents_mmap[idx, :] = embeddings_np[i]
            
            self.latents_written = end_idx
            
            # 定期刷新
            if start_idx % (batch_size * 100) == 0:
                self.latents_mmap.flush()
                torch.cuda.empty_cache()
        
        pbar.close()
        self.latents_mmap.flush()
    
    def _split_text(self, text):
        """简单文本分割"""
        if len(text) <= self.args.max_chars:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.args.max_chars, len(text))
            
            # 尝试在句子边界处分割
            chunk = text[start:end]
            if end < len(text):
                split_pos = -1
                for boundary in ['. ', '! ', '? ', '\n\n', '\n']:
                    pos = chunk.rfind(boundary)
                    if pos > len(chunk) * 0.5 and pos > split_pos:
                        split_pos = pos + len(boundary)
                
                if split_pos > 0:
                    chunk = chunk[:split_pos]
                    end = start + split_pos
            
            if len(chunk) >= self.args.min_chars:
                chunks.append(chunk)
            
            start = end
            
            # 添加重叠
            if self.args.overlap > 0 and start > self.args.overlap:
                start -= self.args.overlap
        
        return chunks
    
    def _save_index(self):
        """保存索引文件"""
        index = {
            'total_samples': self.current_idx,
            'tokens_file': str(self.tokens_file),
            'latents_file': str(self.latents_file),
            'text_file': str(self.text_file),
            'tokens_shape': (self.current_idx, 2, self.args.max_length),
            'latents_shape': (self.current_idx, self.args.embedding_dim),
            'embedding_dim': self.args.embedding_dim,
            'max_length': self.args.max_length,
            'rank': self.rank
        }
        
        np.save(self.index_file, index)
    
    def _create_global_index(self, total_counts):
        """创建全局索引文件（只在rank 0执行）"""
        index = {
            'total_samples': sum(total_counts),
            'samples_per_rank': total_counts,
            'embedding_dim': self.args.embedding_dim,
            'max_length': self.args.max_length,
            'max_chars': self.args.max_chars,
            'model': self.args.model,
            'model_path': self.args.model_path,
            'world_size': self.world_size,
            'created_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'source_dir': self.args.source_dir,
            'files': [f"train_rank{r}" for r in range(self.world_size)]
        }
        
        index_path = self.output_dir / "global_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        print(f"📁 Global index saved: {index_path}")

def main():
    parser = argparse.ArgumentParser(description="Memory-Mapped Big File Generator")
    
    # 必需参数
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-path", required=True, help="Local model path")
    
    # 数据集参数
    parser.add_argument("--dataset", default="openwebtext", help="Dataset name")
    parser.add_argument("--source-dir", help="Dataset source directory")
    
    # 模型参数
    parser.add_argument("--model", default="qwen", choices=["qwen", "e5", "sonar"], help="Model type")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimension")
    
    # 文本处理参数
    parser.add_argument("--max-length", type=int, default=512, help="Max token length")
    parser.add_argument("--max-chars", type=int, default=4096, help="Max characters per chunk")
    parser.add_argument("--min-chars", type=int, default=50, help="Min characters per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between chunks")
    
    # 处理参数
    parser.add_argument("--batch-size", type=int, default=32, help="Processing batch size")
    parser.add_argument("--samples-per-file", type=int, default=200000, help="Target samples per file")
    
    args = parser.parse_args()
    
    # 设置分布式
    rank, world_size, local_rank = setup_distributed()
    
    try:
        # 运行生成器
        generator = MemmapBigFileGenerator(args, rank, world_size, local_rank)
        total_samples = generator.process()
        
        print(f"\nRank {rank}: Process completed successfully!")
        
    except Exception as e:
        print(f"\nRank {rank}: Error in processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # 清理
        if world_size > 1:
            dist.barrier()
            if rank == 0:
                print("\n" + "="*60)
                print("All processes completed!")
                print("="*60)
            dist.destroy_process_group()

if __name__ == "__main__":
    main()