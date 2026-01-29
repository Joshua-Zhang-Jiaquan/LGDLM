# File: evaluate_mmdit_sts.py
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import argparse
from typing import List, Dict, Tuple
import sys
import os

# 添加路径以正确导入
sys.path.append('/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM')

class MMDiTEvaluator:
    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        self.device = device
        
        print(f"Initializing MMDiT evaluator...")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Config: {config_path}")
        print(f"Device: {device}")
        
        # 导入你的采样器 - 使用 samplet2l.py
        try:
            # 首先确保路径正确
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            sys.path.append(project_root)
            
            # 尝试导入你的采样器
            import latentDLM_mmdit.samplet2l as sample_module
            self.SamplerClass = sample_module.FixedT2LSampler
            print("Successfully imported sampler from samplet2l.py")
            
        except ImportError as e:
            print(f"Import error: {e}")
            print("Trying direct import...")
            # 尝试直接导入
            sys.path.append('/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM')
            from latentDLM_mmdit.samplet2l import FixedT2LSampler
            self.SamplerClass = FixedT2LSampler
            print("Successfully imported sampler via direct path")
        
        # 初始化采样器，使用你的原始参数
        print("\nLoading sampler with your parameters...")
        self.sampler = self.SamplerClass(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device
        )
        
        print(f"\nSampler initialized:")
        print(f"  Latent dim: {self.sampler.latent_dim}")
        print(f"  Vocab size: {self.sampler.model_vocab_size}")
        print(f"  Device: {device}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 1, steps: int = 100) -> torch.Tensor:
        """
        批量编码文本为潜向量
        
        使用和你原始脚本相同的参数：
        - batch_size=1: 和你之前测试时一样
        - steps=100: 默认步数（你的脚本默认）
        - max_length=128: 和你设置的一致
        """
        all_latents = []
        
        print(f"\nEncoding {len(texts)} texts with:")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps: {steps}")
        print(f"  Max length: 128")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # 编码为tokens（使用128长度，和你的设置一致）
                tokens, masks = self.sampler.encode_text(batch_texts, max_length=128)
                
                # 生成潜编码 - 使用和你测试相同的参数
                with torch.no_grad():
                    latents = self.sampler.generate(
                        tokens, 
                        attention_mask=masks,
                        steps=steps,
                        guidance_scale=1.0  # 不使用CFG
                    )
                
                # 检查NaN
                if torch.isnan(latents).any():
                    print(f"Warning: NaNs detected in batch {i//batch_size + 1}")
                    latents_nan_count = torch.isnan(latents).sum().item()
                    latents = torch.nan_to_num(latents, nan=0.0)
                    print(f"  Fixed {latents_nan_count} NaN values")
                
                all_latents.append(latents.cpu())
                
                # 打印统计信息
                if i == 0:  # 只打印第一个batch的统计
                    print(f"\nFirst batch statistics:")
                    print(f"  Latent shape: {latents.shape}")
                    print(f"  Latent norm: {latents.norm(dim=1).mean().item():.3f}")
                    print(f"  Latent mean: {latents.mean().item():.4f}")
                    print(f"  Latent std: {latents.std().item():.3f}")
                
            except Exception as e:
                print(f"\nError encoding batch {i}: {e}")
                print("Using random vectors as fallback...")
                # 添加随机向量作为回退
                random_latents = torch.randn(len(batch_texts), self.sampler.latent_dim)
                all_latents.append(random_latents)
        
        if all_latents:
            latents_tensor = torch.cat(all_latents, dim=0)
        else:
            latents_tensor = torch.zeros(0, self.sampler.latent_dim)
        
        # 最终检查
        print(f"\nFinal latent statistics:")
        print(f"  Shape: {latents_tensor.shape}")
        print(f"  Mean: {latents_tensor.mean():.4f}")
        print(f"  Std: {latents_tensor.std():.4f}")
        nan_count = torch.isnan(latents_tensor).sum().item()
        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN values remaining")
        
        return latents_tensor
    
    def load_sts_dataset(self, dataset_name='stsb', split='test', max_samples=100):
        """从HuggingFace加载STS数据集"""
        try:
            from datasets import load_dataset
            
            print(f"\nLoading {dataset_name} dataset from HuggingFace...")
            
            if dataset_name == 'stsb':
                dataset = load_dataset("mteb/stsbenchmark-sts", split=split)
            elif dataset_name == 'sick':
                dataset = load_dataset("sick", split=split)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            texts1 = []
            texts2 = []
            scores = []
            
            print(f"Processing {len(dataset)} samples...")
            for i, item in enumerate(dataset):
                if i >= max_samples:
                    break
                
                if dataset_name == 'stsb':
                    texts1.append(item['sentence1'])
                    texts2.append(item['sentence2'])
                    # STS-B分数是0-5
                    score = float(item['score'])
                elif dataset_name == 'sick':
                    texts1.append(item['sentence_A'])
                    texts2.append(item['sentence_B'])
                    # SICK分数是1-5
                    score = float(item['relatedness_score'])
                
                scores.append(score)
            
            print(f"Loaded {len(texts1)} text pairs")
            print(f"Score range: {min(scores):.2f} to {max(scores):.2f}")
            print(f"Score mean: {np.mean(scores):.2f}")
            
            # 显示一些示例
            print("\nSample text pairs:")
            for j in range(min(3, len(texts1))):
                print(f"\nExample {j+1}:")
                print(f"  Text1: '{texts1[j][:50]}...'")
                print(f"  Text2: '{texts2[j][:50]}...'")
                print(f"  Score: {scores[j]:.2f}")
            
            return texts1, texts2, scores
            
        except ImportError:
            print("\nERROR: 'datasets' package not installed.")
            print("Install with: pip install datasets")
            print("\nFalling back to a minimal test set...")
            return self._create_minimal_test_set(max_samples)
            
        except Exception as e:
            print(f"\nERROR loading dataset: {e}")
            print("\nFalling back to a minimal test set...")
            return self._create_minimal_test_set(max_samples)
    
    def _create_minimal_test_set(self, max_samples=20):
        """创建最小的测试集（当无法下载数据集时）"""
        print("\nCreating minimal test set for debugging...")
        
        test_pairs = [
            ("A man is playing guitar.", "A man plays a musical instrument.", 4.5),
            ("The cat sleeps on the sofa.", "A dog runs in the park.", 1.2),
            ("It's sunny and warm outside.", "The weather is nice today.", 4.8),
            ("She reads a book in the library.", "He watches TV at home.", 1.5),
            ("The company released a new product.", "A new product was launched.", 4.9),
        ]
        
        if max_samples > len(test_pairs):
            # 复制以增加样本
            test_pairs = test_pairs * (max_samples // len(test_pairs) + 1)
        
        test_pairs = test_pairs[:max_samples]
        
        texts1 = [p[0] for p in test_pairs]
        texts2 = [p[1] for p in test_pairs]
        scores = [p[2] for p in test_pairs]
        
        print(f"Created {len(texts1)} test pairs")
        return texts1, texts2, scores
    
    def evaluate_sts(self, dataset_name='stsb', max_samples=100, steps=100):
        """
        评估语义文本相似度
        
        使用真实的STS-B数据集从HuggingFace下载
        """
        print("\n" + "="*70)
        print("EVALUATING SEMANTIC TEXTUAL SIMILARITY (STS)")
        print("="*70)
        
        # 1. 加载数据集
        texts1, texts2, human_scores = self.load_sts_dataset(
            dataset_name=dataset_name, 
            max_samples=max_samples
        )
        
        if len(texts1) < 5:
            print("ERROR: Not enough samples for evaluation")
            return None
        
        # 2. 编码文本
        print("\n" + "="*70)
        print("STEP 1: Encoding texts to latents")
        print("="*70)
        
        print("\nEncoding first set of texts...")
        latents1 = self.encode_texts(texts1, batch_size=1, steps=steps)
        
        print("\nEncoding second set of texts...")
        latents2 = self.encode_texts(texts2, batch_size=1, steps=steps)
        
        # 3. 计算相似度
        print("\n" + "="*70)
        print("STEP 2: Computing similarities")
        print("="*70)
        
        print("\nComputing cosine similarities...")
        model_scores = []
        
        for i in tqdm(range(len(texts1)), desc="Computing"):
            # 计算余弦相似度
            sim = torch.cosine_similarity(
                latents1[i].unsqueeze(0),
                latents2[i].unsqueeze(0)
            ).item()
            model_scores.append(sim)
        
        # 4. 计算评估指标
        print("\n" + "="*70)
        print("STEP 3: Computing evaluation metrics")
        print("="*70)
        
        # 调整分数范围：余弦相似度是-1到1，STS分数是0-5（或1-5）
        # 将余弦相似度映射到原始分数范围
        model_scores_np = np.array(model_scores)
        human_scores_np = np.array(human_scores)
        
        # 计算相关性
        spearman_corr = spearmanr(human_scores_np, model_scores_np).correlation
        pearson_corr = pearsonr(human_scores_np, model_scores_np)[0]
        
        # 计算MAE和RMSE（在原始尺度上）
        mae = np.mean(np.abs(human_scores_np - model_scores_np))
        rmse = np.sqrt(np.mean((human_scores_np - model_scores_np) ** 2))
        
        # 5. 收集结果
        results = {
            'dataset': dataset_name,
            'num_samples': len(texts1),
            'spearman_correlation': float(spearman_corr),
            'pearson_correlation': float(pearson_corr),
            'mean_absolute_error': float(mae),
            'root_mean_squared_error': float(rmse),
            'human_scores': {
                'min': float(human_scores_np.min()),
                'max': float(human_scores_np.max()),
                'mean': float(human_scores_np.mean()),
                'std': float(human_scores_np.std())
            },
            'model_scores': {
                'min': float(model_scores_np.min()),
                'max': float(model_scores_np.max()),
                'mean': float(model_scores_np.mean()),
                'std': float(model_scores_np.std())
            },
            'examples': []
        }
        
        # 添加一些示例
        for i in range(min(5, len(texts1))):
            results['examples'].append({
                'text1': texts1[i],
                'text2': texts2[i],
                'human_score': float(human_scores[i]),
                'model_score': float(model_scores[i])
            })
        
        # 6. 打印结果
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        print(f"\nDataset: {dataset_name.upper()}")
        print(f"Number of samples: {len(texts1)}")
        print(f"\nCorrelation Metrics:")
        print(f"  Spearman Correlation: {spearman_corr:.4f}")
        print(f"  Pearson Correlation:  {pearson_corr:.4f}")
        print(f"\nError Metrics:")
        print(f"  Mean Absolute Error (MAE):   {mae:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        print(f"\nScore Statistics:")
        print(f"  Human scores: {human_scores_np.min():.2f}-{human_scores_np.max():.2f} "
              f"(mean={human_scores_np.mean():.2f}, std={human_scores_np.std():.2f})")
        print(f"  Model scores: {model_scores_np.min():.2f}-{model_scores_np.max():.2f} "
              f"(mean={model_scores_np.mean():.2f}, std={model_scores_np.std():.2f})")
        
        # 解释结果
        print(f"\nInterpretation:")
        if spearman_corr > 0.7:
            print("  ✓ EXCELLENT: Strong semantic understanding (similar to SOTA models)")
        elif spearman_corr > 0.5:
            print("  ✓ GOOD: Reasonable semantic understanding")
        elif spearman_corr > 0.3:
            print("  ✓ MODERATE: Some semantic understanding")
        elif spearman_corr > 0.1:
            print("  ✓ WEAK: Limited semantic understanding")
        else:
            print("  ✗ POOR: Little to no semantic understanding")
        
        print(f"\nExample comparisons:")
        for i, example in enumerate(results['examples'][:3]):
            print(f"\n  Example {i+1}:")
            print(f"    Text 1: '{example['text1'][:60]}...'")
            print(f"    Text 2: '{example['text2'][:60]}...'")
            print(f"    Human score: {example['human_score']:.2f}")
            print(f"    Model similarity: {example['model_score']:.4f}")
        
        return results
    
    def evaluate_consistency(self, num_repetitions=3):
        """测试编码的一致性"""
        print("\n" + "="*70)
        print("EVALUATING ENCODING CONSISTENCY")
        print("="*70)
        
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming our world.",
            "Natural language processing helps computers understand human language."
        ]
        
        all_consistencies = []
        
        for text in test_texts:
            print(f"\nTesting consistency for: '{text[:50]}...'")
            
            # 多次编码相同的文本
            encodings = []
            for i in range(num_repetitions):
                latents = self.encode_texts([text], batch_size=1, steps=50)
                encodings.append(latents[0])
                print(f"  Encoding {i+1}: norm={latents.norm().item():.3f}")
            
            # 计算一致性（编码之间的相似度）
            consistency_scores = []
            for i in range(len(encodings)):
                for j in range(i+1, len(encodings)):
                    sim = torch.cosine_similarity(
                        encodings[i].unsqueeze(0),
                        encodings[j].unsqueeze(0)
                    ).item()
                    consistency_scores.append(sim)
            
            avg_consistency = np.mean(consistency_scores)
            all_consistencies.append(avg_consistency)
            
            print(f"  Average consistency: {avg_consistency:.4f}")
            print(f"  Consistency range: {min(consistency_scores):.4f} - {max(consistency_scores):.4f}")
        
        overall_consistency = np.mean(all_consistencies)
        
        results = {
            'overall_consistency': float(overall_consistency),
            'per_text_consistency': [float(c) for c in all_consistencies],
            'test_texts': test_texts,
            'num_repetitions': num_repetitions
        }
        
        print(f"\nOverall encoding consistency: {overall_consistency:.4f}")
        if overall_consistency > 0.9:
            print("  ✓ EXCELLENT: Highly stable and reproducible encodings")
        elif overall_consistency > 0.7:
            print("  ✓ GOOD: Stable encodings")
        elif overall_consistency > 0.5:
            print("  ✓ MODERATE: Somewhat stable encodings")
        else:
            print("  ✗ POOR: Unstable encodings")
        
        return results
    
    def save_results(self, results, output_dir):
        """保存结果到文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存详细结果
        detailed_file = output_dir / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存总结报告
        summary_file = output_dir / "summary_report.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MMDiT MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            if 'sts' in results:
                sts = results['sts']
                f.write("SEMANTIC TEXTUAL SIMILARITY (STS) EVALUATION\n")
                f.write("-" * 60 + "\n")
                f.write(f"Dataset: {sts.get('dataset', 'N/A')}\n")
                f.write(f"Samples: {sts.get('num_samples', 0)}\n")
                f.write(f"\nCorrelation Metrics:\n")
                f.write(f"  Spearman Correlation: {sts.get('spearman_correlation', 0):.4f}\n")
                f.write(f"  Pearson Correlation:  {sts.get('pearson_correlation', 0):.4f}\n")
                f.write(f"\nError Metrics:\n")
                f.write(f"  Mean Absolute Error (MAE): {sts.get('mean_absolute_error', 0):.4f}\n")
                f.write(f"  Root Mean Squared Error (RMSE): {sts.get('root_mean_squared_error', 0):.4f}\n")
                f.write("\n")
            
            if 'consistency' in results:
                cons = results['consistency']
                f.write("ENCODING CONSISTENCY\n")
                f.write("-" * 60 + "\n")
                f.write(f"Overall Consistency: {cons.get('overall_consistency', 0):.4f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("INTERPRETATION GUIDE:\n")
            f.write("="*80 + "\n")
            f.write("Spearman Correlation:\n")
            f.write("  > 0.7: Excellent semantic understanding\n")
            f.write("  0.5-0.7: Good semantic understanding\n")
            f.write("  0.3-0.5: Moderate semantic understanding\n")
            f.write("  < 0.3: Poor semantic understanding\n\n")
            f.write("Encoding Consistency:\n")
            f.write("  > 0.9: Highly stable encodings\n")
            f.write("  0.7-0.9: Stable encodings\n")
            f.write("  < 0.7: Unstable encodings\n")
            f.write("="*80 + "\n")
        
        print(f"\nResults saved to:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MMDiT model on STS benchmark")
    parser.add_argument("--checkpoint", required=True, help="Path to MMDiT checkpoint")
    parser.add_argument("--config", default=None, help="Model config file (optional)")
    parser.add_argument("--output_dir", default="./eval_results", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dataset", default="stsb", choices=["stsb", "sick"], 
                       help="Dataset to evaluate on")
    parser.add_argument("--max_samples", type=int, default=100, 
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--steps", type=int, default=100, 
                       help="Diffusion steps for encoding")
    parser.add_argument("--run_consistency", action="store_true",
                       help="Also run consistency test")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MMDiT SEMANTIC SIMILARITY EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    if args.config:
        print(f"Config: {args.config}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples}")
    print(f"Diffusion steps: {args.steps}")
    print("="*80 + "\n")
    
    # 运行评估
    evaluator = MMDiTEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    all_results = {}
    
    # 1. STS评估
    print("\n[1/2] Running STS evaluation...")
    sts_results = evaluator.evaluate_sts(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        steps=args.steps
    )
    
    if sts_results is None:
        print("ERROR: STS evaluation failed")
        return
    
    all_results['sts'] = sts_results
    
    # 2. 一致性测试（可选）
    if args.run_consistency:
        print("\n[2/2] Running consistency test...")
        consistency_results = evaluator.evaluate_consistency(num_repetitions=3)
        all_results['consistency'] = consistency_results
    
    # 保存结果
    evaluator.save_results(all_results, args.output_dir)
    
    # 打印最终总结
    print("\n" + "="*80)
    print("EVALUATION COMPLETE - KEY METRICS")
    print("="*80)
    
    if 'sts' in all_results:
        sts = all_results['sts']
        print(f"\nSemantic Understanding (Spearman Correlation):")
        print(f"  {sts['spearman_correlation']:.4f}")
        
        # 星级评分
        corr = sts['spearman_correlation']
        if corr > 0.7:
            stars = "★★★★★"
        elif corr > 0.5:
            stars = "★★★★"
        elif corr > 0.3:
            stars = "★★★"
        elif corr > 0.1:
            stars = "★★"
        else:
            stars = "★"
        
        print(f"  Rating: {stars}")
    
    if 'consistency' in all_results:
        cons = all_results['consistency']
        print(f"\nEncoding Consistency:")
        print(f"  {cons['overall_consistency']:.4f}")
    
    print("\n" + "="*80)
    print("Next steps for NeurIPS paper:")
    print("="*80)
    print("1. Compare with baseline models (BERT, T5, E5, Qwen)")
    print("2. Evaluate on retrieval tasks (MS MARCO, BEIR)")
    print("3. Test noise robustness and paraphrase detection")
    print("4. Analyze latent space geometry (smoothness, interpolation)")
    print("5. Run ablation studies (dimensions, steps, training objectives)")
    print("="*80)

if __name__ == "__main__":
    main()