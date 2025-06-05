"""
LightVision Evaluation System

This module implements comprehensive evaluation metrics for the image retrieval system,
including Recall@K, mAP, MRR, and nDCG as mentioned in the progress report.
"""

import torch
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mobileclip
from retrieval_framework import LightVisionRetrieval, RetrievalConfig
from sklearn.metrics import average_precision_score
import time

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    test_image_dir: str
    test_captions_file: str
    output_dir: str = "evaluation_results"
    k_values: List[int] = None
    batch_size: int = 32
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 5, 10]
        os.makedirs(self.output_dir, exist_ok=True)

class RetrievalMetrics:
    """Implements standard retrieval evaluation metrics"""
    
    @staticmethod
    def recall_at_k(relevant_items: List[int], retrieved_items: List[int], k: int) -> float:
        """
        Calculate Recall@K
        Args:
            relevant_items: List of relevant item indices
            retrieved_items: List of retrieved item indices (top-k)
            k: Number of top items to consider
        """
        if not relevant_items:
            return 0.0
            
        retrieved_k = set(retrieved_items[:k])
        relevant_set = set(relevant_items)
        
        intersection = len(retrieved_k.intersection(relevant_set))
        return intersection / len(relevant_set)
    
    @staticmethod
    def precision_at_k(relevant_items: List[int], retrieved_items: List[int], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
            
        retrieved_k = set(retrieved_items[:k])
        relevant_set = set(relevant_items)
        
        intersection = len(retrieved_k.intersection(relevant_set))
        return intersection / k
    
    @staticmethod
    def average_precision(relevant_items: List[int], retrieved_items: List[int]) -> float:
        """Calculate Average Precision (AP)"""
        if not relevant_items:
            return 0.0
            
        relevant_set = set(relevant_items)
        ap = 0.0
        relevant_count = 0
        
        for i, item in enumerate(retrieved_items):
            if item in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
                
        return ap / len(relevant_set) if relevant_set else 0.0
    
    @staticmethod
    def reciprocal_rank(relevant_items: List[int], retrieved_items: List[int]) -> float:
        """Calculate Reciprocal Rank"""
        relevant_set = set(relevant_items)
        
        for i, item in enumerate(retrieved_items):
            if item in relevant_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    @staticmethod
    def dcg_at_k(relevance_scores: List[float], k: int) -> float:
        """Calculate Discounted Cumulative Gain at K"""
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            dcg += relevance_scores[i] / np.log2(i + 2)
        return dcg
    
    @staticmethod
    def ndcg_at_k(relevance_scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K"""
        dcg = RetrievalMetrics.dcg_at_k(relevance_scores, k)
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = RetrievalMetrics.dcg_at_k(ideal_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0

class DatasetLoader:
    """Handles loading and preparation of test datasets"""
    
    @staticmethod
    def load_flickr_captions(captions_file: str) -> Dict[str, List[str]]:
        """Load Flickr8K/30K style captions"""
        image_captions = {}
        
        if captions_file.endswith('.json'):
            with open(captions_file, 'r') as f:
                data = json.load(f)
                
            for img_name, caption_data in data.items():
                if isinstance(caption_data, dict):
                    # Handle custom format with short/long captions
                    captions = []
                    if 'short_caption' in caption_data:
                        captions.append(caption_data['short_caption'])
                    if 'long_caption' in caption_data:
                        captions.append(caption_data['long_caption'])
                    image_captions[img_name] = captions
                else:
                    # Handle simple format
                    image_captions[img_name] = [caption_data]
                    
        else:
            # Handle text format (Flickr8K style)
            with open(captions_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_info = parts[0]
                        caption = parts[1]
                        
                        # Extract image name (remove caption index if present)
                        if '#' in img_info:
                            img_name = img_info.split('#')[0]
                        else:
                            img_name = img_info
                            
                        if img_name not in image_captions:
                            image_captions[img_name] = []
                        image_captions[img_name].append(caption)
        
        return image_captions
    
    @staticmethod
    def create_test_queries(image_captions: Dict[str, List[str]], 
                          num_queries: int = None) -> List[Dict]:
        """Create test queries from captions"""
        queries = []
        
        for img_name, captions in image_captions.items():
            for caption in captions:
                query = {
                    'text': caption,
                    'relevant_images': [img_name],
                    'image_name': img_name
                }
                queries.append(query)
                
        if num_queries and len(queries) > num_queries:
            # Sample random subset
            import random
            queries = random.sample(queries, num_queries)
            
        return queries

class ModelEvaluator:
    """Main evaluation class"""
    
    def __init__(self, eval_config: EvaluationConfig):
        self.eval_config = eval_config
        self.results = {}
        
    def evaluate_model(self, model_config: RetrievalConfig, 
                      model_name: str = "model") -> Dict:
        """Evaluate a single model configuration"""
        
        print(f"\n=== Evaluating {model_name} ===")
        
        # Initialize retrieval system
        retrieval_system = LightVisionRetrieval(model_config)
        retrieval_system.load_model()
        
        # Build database from test images
        print("Building test database...")
        retrieval_system.build_database(
            self.eval_config.test_image_dir,
            self.eval_config.test_captions_file
        )
        
        # Load test queries
        print("Loading test queries...")
        image_captions = DatasetLoader.load_flickr_captions(
            self.eval_config.test_captions_file
        )
        test_queries = DatasetLoader.create_test_queries(image_captions)
        
        print(f"Evaluating on {len(test_queries)} queries...")
        
        # Evaluate queries
        all_metrics = []
        search_times = []
        
        for query in tqdm(test_queries, desc=f"Evaluating {model_name}"):
            start_time = time.time()
            
            # Perform search
            results = retrieval_system.search(query['text'], k=max(self.eval_config.k_values))
            
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            # Extract retrieved image names
            retrieved_images = []
            for result in results:
                img_name = os.path.basename(result['image_path'])
                retrieved_images.append(img_name)
            
            # Calculate metrics
            relevant_images = query['relevant_images']
            relevant_indices = []
            relevance_scores = []
            
            for i, retrieved_img in enumerate(retrieved_images):
                if retrieved_img in relevant_images:
                    relevant_indices.append(i)
                    relevance_scores.append(1.0)
                else:
                    relevance_scores.append(0.0)
            
            query_metrics = {
                'query_text': query['text'],
                'relevant_images': relevant_images,
                'retrieved_images': retrieved_images,
                'search_time': search_time
            }
            
            # Calculate metrics for different K values
            for k in self.eval_config.k_values:
                query_metrics[f'recall_at_{k}'] = RetrievalMetrics.recall_at_k(
                    relevant_indices, list(range(len(retrieved_images))), k
                )
                query_metrics[f'precision_at_{k}'] = RetrievalMetrics.precision_at_k(
                    relevant_indices, list(range(len(retrieved_images))), k
                )
                query_metrics[f'ndcg_at_{k}'] = RetrievalMetrics.ndcg_at_k(
                    relevance_scores, k
                )
            
            query_metrics['average_precision'] = RetrievalMetrics.average_precision(
                relevant_indices, list(range(len(retrieved_images)))
            )
            query_metrics['reciprocal_rank'] = RetrievalMetrics.reciprocal_rank(
                relevant_indices, list(range(len(retrieved_images)))
            )
            
            all_metrics.append(query_metrics)
        
        # Aggregate results
        aggregated_results = self._aggregate_metrics(all_metrics, model_name)
        aggregated_results['avg_search_time'] = np.mean(search_times)
        aggregated_results['total_queries'] = len(test_queries)
        
        return aggregated_results
    
    def _aggregate_metrics(self, all_metrics: List[Dict], model_name: str) -> Dict:
        """Aggregate metrics across all queries"""
        
        results = {'model_name': model_name}
        
        # Calculate mean for each metric
        metric_keys = [key for key in all_metrics[0].keys() 
                      if key not in ['query_text', 'relevant_images', 'retrieved_images']]
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            results[f'mean_{key}'] = np.mean(values)
            results[f'std_{key}'] = np.std(values)
        
        # Calculate mAP (Mean Average Precision)
        ap_values = [m['average_precision'] for m in all_metrics]
        results['mAP'] = np.mean(ap_values)
        
        # Calculate MRR (Mean Reciprocal Rank)
        rr_values = [m['reciprocal_rank'] for m in all_metrics]
        results['MRR'] = np.mean(rr_values)
        
        return results
    
    def compare_models(self, model_configs: List[Tuple[RetrievalConfig, str]]) -> Dict:
        """Compare multiple model configurations"""
        
        comparison_results = {}
        
        for config, name in model_configs:
            results = self.evaluate_model(config, name)
            comparison_results[name] = results
        
        # Save results
        self._save_comparison_results(comparison_results)
        self._create_visualizations(comparison_results)
        
        return comparison_results
    
    def _save_comparison_results(self, results: Dict):
        """Save comparison results to files"""
        
        # Save raw results as JSON
        results_file = os.path.join(self.eval_config.output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary table
        summary_data = []
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            # Add key metrics
            for k in self.eval_config.k_values:
                row[f'Recall@{k}'] = f"{model_results[f'mean_recall_at_{k}']:.4f}"
                row[f'Precision@{k}'] = f"{model_results[f'mean_precision_at_{k}']:.4f}"
                row[f'nDCG@{k}'] = f"{model_results[f'mean_ndcg_at_{k}']:.4f}"
            
            row['mAP'] = f"{model_results['mAP']:.4f}"
            row['MRR'] = f"{model_results['MRR']:.4f}"
            row['Avg Search Time (s)'] = f"{model_results['avg_search_time']:.4f}"
            
            summary_data.append(row)
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        csv_file = os.path.join(self.eval_config.output_dir, "evaluation_summary.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to {self.eval_config.output_dir}")
        print("\nSummary:")
        print(df.to_string(index=False))
    
    def _create_visualizations(self, results: Dict):
        """Create evaluation visualizations"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Recall@K comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Recall@K
        models = list(results.keys())
        recall_data = {}
        for k in self.eval_config.k_values:
            recall_data[f'Recall@{k}'] = [results[model][f'mean_recall_at_{k}'] for model in models]
        
        df_recall = pd.DataFrame(recall_data, index=models)
        df_recall.plot(kind='bar', ax=axes[0, 0], title='Recall@K Comparison')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision@K
        precision_data = {}
        for k in self.eval_config.k_values:
            precision_data[f'Precision@{k}'] = [results[model][f'mean_precision_at_{k}'] for model in models]
        
        df_precision = pd.DataFrame(precision_data, index=models)
        df_precision.plot(kind='bar', ax=axes[0, 1], title='Precision@K Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # mAP and MRR
        map_mrr_data = {
            'mAP': [results[model]['mAP'] for model in models],
            'MRR': [results[model]['MRR'] for model in models]
        }
        
        df_map_mrr = pd.DataFrame(map_mrr_data, index=models)
        df_map_mrr.plot(kind='bar', ax=axes[1, 0], title='mAP and MRR Comparison')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Search time comparison
        search_times = [results[model]['avg_search_time'] for model in models]
        axes[1, 1].bar(models, search_times)
        axes[1, 1].set_title('Average Search Time Comparison')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_config.output_dir, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Detailed metrics heatmap
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = []
        metric_names = []
        
        for k in self.eval_config.k_values:
            metric_names.extend([f'Recall@{k}', f'Precision@{k}', f'nDCG@{k}'])
        metric_names.extend(['mAP', 'MRR'])
        
        for model in models:
            model_scores = []
            for k in self.eval_config.k_values:
                model_scores.extend([
                    results[model][f'mean_recall_at_{k}'],
                    results[model][f'mean_precision_at_{k}'],
                    results[model][f'mean_ndcg_at_{k}']
                ])
            model_scores.extend([results[model]['mAP'], results[model]['MRR']])
            heatmap_data.append(model_scores)
        
        heatmap_df = pd.DataFrame(heatmap_data, index=models, columns=metric_names)
        
        sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Score'})
        plt.title('Model Performance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_config.output_dir, 'performance_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.eval_config.output_dir}")

def create_model_configs() -> List[Tuple[RetrievalConfig, str]]:
    """Create different model configurations for comparison"""
    
    configs = []
    
    # Base MobileCLIP model
    base_config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path=None,  # Use pretrained model
        device='cuda' if torch.cuda.is_available() else 'cpu',
        top_k=10
    )
    configs.append((base_config, "Base MobileCLIP"))
    
    # Fine-tuned model (replace with your actual checkpoint path)
    finetuned_config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path='checkpoints/mobileclip_finetuned_epoch1_last.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        top_k=10
    )
    configs.append((finetuned_config, "Fine-tuned MobileCLIP"))
    
    # You can add more configurations here, for example:
    # - Different checkpoint epochs
    # - Different FAISS index types
    # - Different model variants
    
    return configs

def run_comprehensive_evaluation():
    """Run a comprehensive evaluation comparing multiple models"""
    
    # Configuration
    eval_config = EvaluationConfig(
        test_image_dir="data/Images",  # Replace with your test image directory
        test_captions_file="data/captions.txt",  # Replace with your test captions file
        output_dir="evaluation_results",
        k_values=[1, 5, 10]
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(eval_config)
    
    # Get model configurations
    model_configs = create_model_configs()
    
    print("Starting comprehensive evaluation...")
    print(f"Test images: {eval_config.test_image_dir}")
    print(f"Test captions: {eval_config.test_captions_file}")
    print(f"Models to evaluate: {[name for _, name in model_configs]}")
    
    # Run comparison
    results = evaluator.compare_models(model_configs)
    
    return results

def analyze_query_length_performance(model_config: RetrievalConfig, 
                                   eval_config: EvaluationConfig):
    """Analyze performance based on query length (short vs long captions)"""
    
    print("\n=== Query Length Analysis ===")
    
    # Initialize retrieval system
    retrieval_system = LightVisionRetrieval(model_config)
    retrieval_system.load_model()
    retrieval_system.build_database(eval_config.test_image_dir, eval_config.test_captions_file)
    
    # Load captions and categorize by length
    image_captions = DatasetLoader.load_flickr_captions(eval_config.test_captions_file)
    
    short_queries = []
    long_queries = []
    
    for img_name, captions in image_captions.items():
        for caption in captions:
            query = {
                'text': caption,
                'relevant_images': [img_name],
                'length': len(caption.split())
            }
            
            if query['length'] <= 10:  # Threshold for short vs long
                short_queries.append(query)
            else:
                long_queries.append(query)
    
    print(f"Short queries (≤10 words): {len(short_queries)}")
    print(f"Long queries (>10 words): {len(long_queries)}")
    
    # Evaluate both categories
    categories = [("Short Queries", short_queries), ("Long Queries", long_queries)]
    length_results = {}
    
    for category_name, queries in categories:
        if not queries:
            continue
            
        print(f"\nEvaluating {category_name}...")
        category_metrics = []
        
        for query in tqdm(queries, desc=f"Processing {category_name}"):
            results = retrieval_system.search(query['text'], k=10)
            
            retrieved_images = [os.path.basename(r['image_path']) for r in results]
            relevant_images = query['relevant_images']
            relevant_indices = [i for i, img in enumerate(retrieved_images) if img in relevant_images]
            
            metrics = {
                'recall_at_1': RetrievalMetrics.recall_at_k(relevant_indices, list(range(len(retrieved_images))), 1),
                'recall_at_5': RetrievalMetrics.recall_at_k(relevant_indices, list(range(len(retrieved_images))), 5),
                'average_precision': RetrievalMetrics.average_precision(relevant_indices, list(range(len(retrieved_images))))
            }
            category_metrics.append(metrics)
        
        # Aggregate results
        length_results[category_name] = {
            'mean_recall_at_1': np.mean([m['recall_at_1'] for m in category_metrics]),
            'mean_recall_at_5': np.mean([m['recall_at_5'] for m in category_metrics]),
            'mean_average_precision': np.mean([m['average_precision'] for m in category_metrics]),
            'count': len(queries)
        }
    
    # Print results
    print("\n=== Query Length Analysis Results ===")
    for category, results in length_results.items():
        print(f"\n{category}:")
        print(f"  Count: {results['count']}")
        print(f"  Recall@1: {results['mean_recall_at_1']:.4f}")
        print(f"  Recall@5: {results['mean_recall_at_5']:.4f}")
        print(f"  mAP: {results['mean_average_precision']:.4f}")
    
    return length_results

def quick_evaluation_demo():
    """Quick demonstration of the evaluation system"""
    
    print("=== Quick Evaluation Demo ===")
    
    # Simple configuration for demo
    eval_config = EvaluationConfig(
        test_image_dir="data/Images",
        test_captions_file="data/captions.txt",
        output_dir="demo_results",
        k_values=[1, 5]
    )
    
    # Base model configuration
    model_config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path=None,  # Base model
        device='cuda' if torch.cuda.is_available() else 'cpu',
        top_k=5
    )
    
    # Run evaluation
    evaluator = ModelEvaluator(eval_config)
    results = evaluator.evaluate_model(model_config, "Demo Model")
    
    print("\n=== Demo Results ===")
    print(f"Recall@1: {results['mean_recall_at_1']:.4f}")
    print(f"Recall@5: {results['mean_recall_at_5']:.4f}")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"MRR: {results['MRR']:.4f}")
    print(f"Average search time: {results['avg_search_time']:.4f}s")
    
    return results

def main():
    """Main evaluation script"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='LightVision Model Evaluation')
    parser.add_argument('--mode', choices=['demo', 'full', 'length_analysis'], 
                       default='demo', help='Evaluation mode')
    parser.add_argument('--test_images', type=str, default='data/Images',
                       help='Path to test images directory')
    parser.add_argument('--test_captions', type=str, default='data/captions.txt',
                       help='Path to test captions file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        quick_evaluation_demo()
        
    elif args.mode == 'full':
        # Update configurations with command line arguments
        eval_config = EvaluationConfig(
            test_image_dir=args.test_images,
            test_captions_file=args.test_captions,
            output_dir=args.output_dir
        )
        
        # Run full evaluation
        run_comprehensive_evaluation()
        
    elif args.mode == 'length_analysis':
        eval_config = EvaluationConfig(
            test_image_dir=args.test_images,
            test_captions_file=args.test_captions,
            output_dir=args.output_dir
        )
        
        model_config = RetrievalConfig(
            checkpoint_path=args.checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        analyze_query_length_performance(model_config, eval_config)

if __name__ == "__main__":
    main()