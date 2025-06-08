import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import torch
from .retriever import FAISSRetriever

def calculate_recall_at_k(true_positives, total_queries, k_values=[1, 5, 10]):
    """Calculate Recall@K metrics"""
    recalls = {}
    for k in k_values:
        recalls[f'recall@{k}'] = true_positives.get(k, 0) / total_queries if total_queries > 0 else 0.0
    return recalls

def calculate_mean_average_precision(retrieval_results, k_values=[1, 5, 10]):
    """
    Calculate Mean Average Precision (MAP) metrics
    
    Args:
        retrieval_results: List of dicts containing 'is_relevant' and 'rank' for each result
        k_values: List of k values for MAP@K calculation
    
    Returns:
        dict: MAP@K scores
    """
    maps = {}
    
    for k in k_values:
        # Get results up to rank k
        k_results = [r for r in retrieval_results if r['rank'] <= k]
        
        if not k_results:
            maps[f'map@{k}'] = 0.0
            continue
        
        # Calculate precision at each relevant position
        relevant_count = 0
        precision_sum = 0.0
        
        for result in k_results:
            if result['is_relevant']:
                relevant_count += 1
                precision_at_i = relevant_count / result['rank']
                precision_sum += precision_at_i
        
        # Average precision for this query
        if relevant_count > 0:
            avg_precision = precision_sum / relevant_count
        else:
            avg_precision = 0.0
        
        maps[f'map@{k}'] = avg_precision
    
    return maps

def calculate_mean_reciprocal_rank(retrieval_results, k_values=[1, 5, 10]):
    """
    Calculate Mean Reciprocal Rank (MRR) metrics
    
    Args:
        retrieval_results: List of dicts containing 'is_relevant' and 'rank' for each result
        k_values: List of k values for MRR@K calculation
    
    Returns:
        dict: MRR@K scores
    """
    mrrs = {}
    
    for k in k_values:
        # Find first relevant result within top-k
        first_relevant_rank = None
        
        for result in retrieval_results:
            if result['rank'] <= k and result['is_relevant']:
                first_relevant_rank = result['rank']
                break
        
        # Reciprocal rank
        if first_relevant_rank is not None:
            mrrs[f'mrr@{k}'] = 1.0 / first_relevant_rank
        else:
            mrrs[f'mrr@{k}'] = 0.0
    
    return mrrs

def calculate_ndcg(retrieval_results, k_values=[1, 5, 10]):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) metrics
    
    Args:
        retrieval_results: List of dicts containing 'is_relevant', 'rank', and 'similarity' for each result
        k_values: List of k values for NDCG@K calculation
    
    Returns:
        dict: NDCG@K scores
    """
    import math
    
    ndcgs = {}
    
    for k in k_values:
        # Get results up to rank k
        k_results = [r for r in retrieval_results if r['rank'] <= k]
        
        if not k_results:
            ndcgs[f'ndcg@{k}'] = 0.0
            continue
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for result in k_results:
            relevance = 1.0 if result['is_relevant'] else 0.0
            rank = result['rank']
            
            if rank == 1:
                dcg += relevance
            else:
                dcg += relevance / math.log2(rank)
        
        # Calculate IDCG (Ideal DCG) - assuming binary relevance
        # For binary relevance, IDCG is the DCG when all relevant items are ranked first
        relevant_count = sum(1 for r in retrieval_results if r['is_relevant'])
        ideal_relevant_count = min(relevant_count, k)
        
        idcg = 0.0
        for i in range(1, ideal_relevant_count + 1):
            if i == 1:
                idcg += 1.0
            else:
                idcg += 1.0 / math.log2(i)
        
        # Calculate NDCG
        if idcg > 0:
            ndcgs[f'ndcg@{k}'] = dcg / idcg
        else:
            ndcgs[f'ndcg@{k}'] = 0.0
    
    return ndcgs

def aggregate_metrics(all_query_metrics, total_queries):
    """
    Aggregate metrics across all queries
    
    Args:
        all_query_metrics: List of metric dicts for each query
        total_queries: Total number of queries
    
    Returns:
        dict: Aggregated metrics
    """
    if not all_query_metrics or total_queries == 0:
        return {}
    
    # Initialize aggregated metrics
    aggregated = {}
    
    # Get all metric keys from the first query
    if all_query_metrics:
        metric_keys = all_query_metrics[0].keys()
        
        for key in metric_keys:
            # Sum all values for this metric across queries
            total_value = sum(query_metrics[key] for query_metrics in all_query_metrics)
            # Calculate mean
            aggregated[key] = total_value / total_queries
    
    return aggregated

def evaluate_retrieval(config, retriever, dataset, split='test', k_values=[1, 5, 10], verbose=True):
    """
    Evaluate retrieval performance using Recall@K metrics
    
    Args:
        retriever: FAISSRetriever instance
        dataset: CustomDataset instance
        split: Which split to evaluate ('train' or 'test')
        k_values: List of k values for Recall@K calculation
        verbose: Whether to print progress
    
    Returns:
        dict: Evaluation results
    """
    if verbose:
        print(f"Evaluating retrieval on {split} split...")
    
    # Get all data for the split
    if split == 'train':
        data = dataset.train_data
    else:
        data = dataset.test_data
    
    # Initialize counters for different caption types
    short_caption_results = {k: 0 for k in k_values}
    long_caption_results = {k: 0 for k in k_values}
    sub_caption_results = {k: 0 for k in k_values}  # For sub-captions if needed
    
    # Initialize lists to store per-query metrics for aggregation
    short_map_results = []
    short_mrr_results = []
    short_ndcg_results = []
    
    long_map_results = []
    long_mrr_results = []
    long_ndcg_results = []
    
    sub_map_results = []
    sub_mrr_results = []
    sub_ndcg_results = []
    
    total_queries = len(data)
    
    sample_return = None  # For returning a sample query and results
    
    initial_k = total_queries
    
    if verbose:
        print(f"Processing {total_queries} queries...")
    
    # Evaluate each image-caption pair
    for idx, item in enumerate(data):
        if verbose and idx % 200 == 0:
            print(f"Processing query {idx + 1}/{total_queries}")
        
        target_image_name = item['image_name']
        
        # Test short captions
        short_caption = item['short_caption']
        short_results = retriever.retrieve(short_caption, k=max(k_values))
        
        # Prepare retrieval results for metric calculation
        short_retrieval_results = []
        for result in short_results:
            short_retrieval_results.append({
                'rank': result['rank'],
                'is_relevant': result['image_name'] == target_image_name,
                'similarity': result.get('similarity', result.get('cumulative_score', 0.0))
            })
        
        # Calculate metrics for short caption
        short_map = calculate_mean_average_precision(short_retrieval_results, k_values)
        short_mrr = calculate_mean_reciprocal_rank(short_retrieval_results, k_values)
        short_ndcg = calculate_ndcg(short_retrieval_results, k_values)
        
        short_map_results.append(short_map)
        short_mrr_results.append(short_mrr)
        short_ndcg_results.append(short_ndcg)
        
        # Check if target image is in top-k for each k
        for k in k_values:
            top_k_images = [result['image_name'] for result in short_results[:k]]
            if target_image_name in top_k_images:
                short_caption_results[k] += 1
        
        # Test long captions
        long_caption = item['long_caption']
        long_results = retriever.retrieve(long_caption, k=max(k_values))
        
        # Prepare retrieval results for metric calculation
        long_retrieval_results = []
        for result in long_results:
            long_retrieval_results.append({
                'rank': result['rank'],
                'is_relevant': result['image_name'] == target_image_name,
                'similarity': result.get('similarity', result.get('cumulative_score', 0.0))
            })
        
        # Calculate metrics for long caption
        long_map = calculate_mean_average_precision(long_retrieval_results, k_values)
        long_mrr = calculate_mean_reciprocal_rank(long_retrieval_results, k_values)
        long_ndcg = calculate_ndcg(long_retrieval_results, k_values)
        
        long_map_results.append(long_map)
        long_mrr_results.append(long_mrr)
        long_ndcg_results.append(long_ndcg)
        
        # Check if target image is in top-k for each k
        for k in k_values:
            top_k_images = [result['image_name'] for result in long_results[:k]]
            if target_image_name in top_k_images:
                long_caption_results[k] += 1

        


        sub_captions = item.get('long_splitted_caption', [])
        sub_results = retriever.retrieve_with_subsections(sub_captions, k=max(k_values), initial_k=config.get('retriever.initial_k', 20))
        
        # Prepare retrieval results for metric calculation
        sub_retrieval_results = []
        for result in sub_results:
            sub_retrieval_results.append({
                'rank': result['rank'],
                'is_relevant': result['image_name'] == target_image_name,
                'similarity': result.get('similarity', result.get('cumulative_score', 0.0))
            })
        
        # Calculate metrics for sub captions
        sub_map = calculate_mean_average_precision(sub_retrieval_results, k_values)
        sub_mrr = calculate_mean_reciprocal_rank(sub_retrieval_results, k_values)
        sub_ndcg = calculate_ndcg(sub_retrieval_results, k_values)
        
        sub_map_results.append(sub_map)
        sub_mrr_results.append(sub_mrr)
        sub_ndcg_results.append(sub_ndcg)
        
        for k in k_values:
            top_k_images = [result['image_name'] for result in sub_results[:k]]
            if target_image_name in top_k_images:
                sub_caption_results[k] += 1

        if idx == 0:
            sample_return = {
                'short_caption': short_caption,
                'long_caption': long_caption,
                'short_results': short_results[:5],  # Return top 5 for sample
                'long_results': long_results[:5],
                'sub_results': sub_results[:5]
            }

    # Calculate final metrics
    short_recalls = calculate_recall_at_k(short_caption_results, total_queries, k_values)
    long_recalls = calculate_recall_at_k(long_caption_results, total_queries, k_values)
    sub_recalls = calculate_recall_at_k(sub_caption_results, total_queries, k_values)
    
    # Aggregate MAP, MRR, and NDCG metrics
    short_map_avg = aggregate_metrics(short_map_results, total_queries)
    short_mrr_avg = aggregate_metrics(short_mrr_results, total_queries)
    short_ndcg_avg = aggregate_metrics(short_ndcg_results, total_queries)
    
    long_map_avg = aggregate_metrics(long_map_results, total_queries)
    long_mrr_avg = aggregate_metrics(long_mrr_results, total_queries)
    long_ndcg_avg = aggregate_metrics(long_ndcg_results, total_queries)
    
    sub_map_avg = aggregate_metrics(sub_map_results, total_queries)
    sub_mrr_avg = aggregate_metrics(sub_mrr_results, total_queries)
    sub_ndcg_avg = aggregate_metrics(sub_ndcg_results, total_queries)

    # Calculate average recalls
    avg_recalls = {}
    for k in k_values:
        avg_recalls[f'recall@{k}'] = (short_recalls[f'recall@{k}'] + long_recalls[f'recall@{k}'] + sub_recalls[f'recall@{k}']) / 3
    
    # Calculate average MAP, MRR, NDCG across caption types
    avg_map = {}
    avg_mrr = {}
    avg_ndcg = {}
    
    for k in k_values:
        map_key = f'map@{k}'
        mrr_key = f'mrr@{k}'
        ndcg_key = f'ndcg@{k}'
        
        avg_map[map_key] = (short_map_avg.get(map_key, 0) + long_map_avg.get(map_key, 0) + sub_map_avg.get(map_key, 0)) / 3
        avg_mrr[mrr_key] = (short_mrr_avg.get(mrr_key, 0) + long_mrr_avg.get(mrr_key, 0) + sub_mrr_avg.get(mrr_key, 0)) / 3
        avg_ndcg[ndcg_key] = (short_ndcg_avg.get(ndcg_key, 0) + long_ndcg_avg.get(ndcg_key, 0) + sub_ndcg_avg.get(ndcg_key, 0)) / 3
    
    results = {
        'split': split,
        'total_queries': total_queries,
        'short_caption_recalls': short_recalls,
        'long_caption_recalls': long_recalls,
        'sub_caption_recalls': sub_recalls,
        'average_recalls': avg_recalls,
        'short_caption_map': short_map_avg,
        'long_caption_map': long_map_avg,
        'sub_caption_map': sub_map_avg,
        'average_map': avg_map,
        'short_caption_mrr': short_mrr_avg,
        'long_caption_mrr': long_mrr_avg,
        'sub_caption_mrr': sub_mrr_avg,
        'average_mrr': avg_mrr,
        'short_caption_ndcg': short_ndcg_avg,
        'long_caption_ndcg': long_ndcg_avg,
        'sub_caption_ndcg': sub_ndcg_avg,
        'average_ndcg': avg_ndcg,
        'k_values': k_values
    }
    
    if verbose:
        print(f"\n=== {split.upper()} SPLIT RESULTS ===")
        print(f"Total queries: {total_queries}")
        print("\nShort Caption Results:")
        for k in k_values:
            print(f"  Recall@{k}: {short_recalls[f'recall@{k}']:.4f}")
            print(f"  MAP@{k}: {short_map_avg.get(f'map@{k}', 0):.4f}")
            print(f"  MRR@{k}: {short_mrr_avg.get(f'mrr@{k}', 0):.4f}")
            print(f"  NDCG@{k}: {short_ndcg_avg.get(f'ndcg@{k}', 0):.4f}")
        print("\nLong Caption Results:")
        for k in k_values:
            print(f"  Recall@{k}: {long_recalls[f'recall@{k}']:.4f}")
            print(f"  MAP@{k}: {long_map_avg.get(f'map@{k}', 0):.4f}")
            print(f"  MRR@{k}: {long_mrr_avg.get(f'mrr@{k}', 0):.4f}")
            print(f"  NDCG@{k}: {long_ndcg_avg.get(f'ndcg@{k}', 0):.4f}")
        print("\nSub Caption Results:")
        for k in k_values:
            print(f"  Recall@{k}: {sub_recalls[f'recall@{k}']:.4f}")
            print(f"  MAP@{k}: {sub_map_avg.get(f'map@{k}', 0):.4f}")
            print(f"  MRR@{k}: {sub_mrr_avg.get(f'mrr@{k}', 0):.4f}")
            print(f"  NDCG@{k}: {sub_ndcg_avg.get(f'ndcg@{k}', 0):.4f}")
        print("\nAverage Results:")
        for k in k_values:
            print(f"  Recall@{k}: {avg_recalls[f'recall@{k}']:.4f}")
            print(f"  MAP@{k}: {avg_map.get(f'map@{k}', 0):.4f}")
            print(f"  MRR@{k}: {avg_mrr.get(f'mrr@{k}', 0):.4f}")
            print(f"  NDCG@{k}: {avg_ndcg.get(f'ndcg@{k}', 0):.4f}")

    return results, sample_return

def evaluate_dataset(model=None, testDataset=None, config=None, tokenizer=None, k_values=[1, 5, 10], 
                    force_rebuild_index=False, verbose=True, only_test=False):
    """
    Main evaluation function for the dataset
    
    Args:
        model: Loaded model (e.g., MobileCLIP)
        testDataset: CustomDataset instance
        config: Configuration object
        tokenizer: Text tokenizer for the model
        k_values: List of k values for Recall@K calculation
        force_rebuild_index: Whether to force rebuild FAISS index
        verbose: Whether to print progress
    
    Returns:
        dict: Complete evaluation results
    """
    
    if model is None or testDataset is None or config is None or tokenizer is None:
        raise ValueError("model, testDataset, config, and tokenizer are required")
    
    if verbose:
        print("Starting dataset evaluation...")
        print(f"Model: {config.get('model.name', 'Unknown')}")
        print(f"Device: {config.get('model.device', 'Unknown')}")
    
    # Verify dataset split
    if not testDataset.verify_no_overlap():
        if verbose:
            print("WARNING: Train/test split has overlapping images!")
    
    # Get split info
    split_info = testDataset.get_split_info()
    if verbose and split_info:
        print(f"Dataset split: {split_info['train_size']} train, {split_info['test_size']} test")
    
    results = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_name': config.get('model.name', 'Unknown'),
        'device': config.get('model.device', 'cpu'),
        'dataset_info': split_info,
        'k_values': k_values
    }
    
    sample_results = []
    
    # Evaluate on both train and test splits
    for split in ['train', 'test'] if not only_test else ['test']:
        if verbose:
            print(f"\n{'='*50}")
            print(f"Evaluating {split} split")
            print(f"{'='*50}")
        
        # Create retriever for this split
        retriever = FAISSRetriever(config, model, testDataset, tokenizer, split=split)
        
        # Build or load FAISS index
        retriever.build_or_load_index(force_rebuild=force_rebuild_index, verbose=verbose)
        
        # Evaluate retrieval performance
        split_results, sample_result = evaluate_retrieval(
            config,
            retriever, testDataset, split=split, 
            k_values=k_values, verbose=verbose
        )
        
        sample_results.append(sample_result)
        
        results[f'{split}_results'] = split_results
    
    # Save results to file
    save_results(results, config, verbose=verbose)
    
    if verbose:
        print(f"\n{'='*50}")
        print("EVALUATION COMPLETE")
        print(f"{'='*50}")
        print_summary(results)
    
    return results, sample_results

def save_results(results, config, verbose=True):
    """Save evaluation results to a JSON file"""
    try:
        project_root = config.get('project.root', '.')
        results_dir = os.path.join(project_root, 'data', 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename with timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get('model.name', 'unknown')
        filename = f"evaluation_{model_name}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if verbose:
            print(f"Results saved to: {filepath}")
        
    except Exception as e:
        if verbose:
            print(f"Error saving results: {e}")

def print_summary(results):
    """Print a summary of evaluation results"""
    print("\nEVALUATION SUMMARY")
    print("-" * 40)
    
    for split in ['train', 'test']:
        if f'{split}_results' in results:
            split_results = results[f'{split}_results']
            print(f"\n{split.upper()} Split:")
            
            avg_recalls = split_results['average_recalls']
            avg_map = split_results.get('average_map', {})
            avg_mrr = split_results.get('average_mrr', {})
            avg_ndcg = split_results.get('average_ndcg', {})
            
            for k_metric, value in avg_recalls.items():
                k = k_metric.split('@')[1]
                print(f"  Recall@{k}: {value:.4f}")
                print(f"  MAP@{k}: {avg_map.get(f'map@{k}', 0):.4f}")
                print(f"  MRR@{k}: {avg_mrr.get(f'mrr@{k}', 0):.4f}")
                print(f"  NDCG@{k}: {avg_ndcg.get(f'ndcg@{k}', 0):.4f}")
                print()
    
    # Compare train vs test if both exist
    if 'train_results' in results and 'test_results' in results:
        print(f"\nTrain vs Test Comparison:")
        train_avg_recall = results['train_results']['average_recalls']
        test_avg_recall = results['test_results']['average_recalls']
        train_avg_map = results['train_results'].get('average_map', {})
        test_avg_map = results['test_results'].get('average_map', {})
        train_avg_mrr = results['train_results'].get('average_mrr', {})
        test_avg_mrr = results['test_results'].get('average_mrr', {})
        train_avg_ndcg = results['train_results'].get('average_ndcg', {})
        test_avg_ndcg = results['test_results'].get('average_ndcg', {})
        
        for k_metric in train_avg_recall.keys():
            k = k_metric.split('@')[1]
            
            # Recall comparison
            train_val = train_avg_recall[k_metric]
            test_val = test_avg_recall[k_metric]
            diff = train_val - test_val
            print(f"  Recall@{k}: Train={train_val:.4f}, Test={test_val:.4f}, Diff={diff:+.4f}")
            
            # MAP comparison
            map_key = f'map@{k}'
            train_map = train_avg_map.get(map_key, 0)
            test_map = test_avg_map.get(map_key, 0)
            map_diff = train_map - test_map
            print(f"  MAP@{k}: Train={train_map:.4f}, Test={test_map:.4f}, Diff={map_diff:+.4f}")
            
            # MRR comparison
            mrr_key = f'mrr@{k}'
            train_mrr = train_avg_mrr.get(mrr_key, 0)
            test_mrr = test_avg_mrr.get(mrr_key, 0)
            mrr_diff = train_mrr - test_mrr
            print(f"  MRR@{k}: Train={train_mrr:.4f}, Test={test_mrr:.4f}, Diff={mrr_diff:+.4f}")
            
            # NDCG comparison
            ndcg_key = f'ndcg@{k}'
            train_ndcg = train_avg_ndcg.get(ndcg_key, 0)
            test_ndcg = test_avg_ndcg.get(ndcg_key, 0)
            ndcg_diff = train_ndcg - test_ndcg
            print(f"  NDCG@{k}: Train={train_ndcg:.4f}, Test={test_ndcg:.4f}, Diff={ndcg_diff:+.4f}")
            print()
            
            
def plot_sample_result(sample_results, dataset=None):
    """Plot sample results for short and long captions showing retrieved images"""
    import matplotlib.pyplot as plt
    from PIL import Image
    import os
    
    if dataset is None:
        print("Dataset is required to load images")
        return
    
    for idx, sample in enumerate(sample_results):
        if sample is None:
            continue
        
        short_caption = sample['short_caption']
        long_caption = sample['long_caption']
        short_results = sample['short_results']
        long_results = sample['long_results']
        
        # Create figure with subplots for images
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Sample {idx + 1}: Retrieved Images Comparison", fontsize=16)
        
        # Plot short caption results
        axes[0, 0].text(0.5, 0.5, f"Short Caption:\n{short_caption}", 
                       ha='center', va='center', wrap=True, fontsize=10)
        axes[0, 0].axis('off')
        
        for i, result in enumerate(short_results[:4]):  # Show 4 images + 1 text
            try:
                # Load image
                image_path = os.path.join(dataset.images_dir, result['image_name'])
                img = Image.open(image_path)
                
                # Display image
                axes[0, i + 1].imshow(img)
                axes[0, i + 1].set_title(f"#{i + 1}: {result['image_name']}\nScore: {result['score']:.3f}", 
                                        fontsize=8)
                axes[0, i + 1].axis('off')
            except Exception as e:
                axes[0, i + 1].text(0.5, 0.5, f"Error loading\n{result['image_name']}", 
                                   ha='center', va='center', fontsize=8)
                axes[0, i + 1].axis('off')
        
        # Plot long caption results
        axes[1, 0].text(0.5, 0.5, f"Long Caption:\n{long_caption}", 
                       ha='center', va='center', wrap=True, fontsize=10)
        axes[1, 0].axis('off')
        
        for i, result in enumerate(long_results[:4]):  # Show 4 images + 1 text
            try:
                # Load image
                image_path = os.path.join(dataset.images_dir, result['image_name'])
                img = Image.open(image_path)
                
                # Display image
                axes[1, i + 1].imshow(img)
                axes[1, i + 1].set_title(f"#{i + 1}: {result['image_name']}\nScore: {result['score']:.3f}", 
                                        fontsize=8)
                axes[1, i + 1].axis('off')
            except Exception as e:
                axes[1, i + 1].text(0.5, 0.5, f"Error loading\n{result['image_name']}", 
                                   ha='center', va='center', fontsize=8)
                axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
