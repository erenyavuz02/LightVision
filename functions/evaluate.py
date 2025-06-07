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

def evaluate_retrieval(retriever, dataset, split='test', k_values=[1, 5, 10], verbose=True):
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
    
    total_queries = len(data)
    
    sample_return = None  # For returning a sample query and results
    
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
        
        # Check if target image is in top-k for each k
        for k in k_values:
            top_k_images = [result['image_name'] for result in short_results[:k]]
            if target_image_name in top_k_images:
                short_caption_results[k] += 1
        
        # Test long captions
        long_caption = item['long_caption']
        long_results = retriever.retrieve(long_caption, k=max(k_values))
        
        # Check if target image is in top-k for each k
        for k in k_values:
            top_k_images = [result['image_name'] for result in long_results[:k]]
            if target_image_name in top_k_images:
                long_caption_results[k] += 1
                
        if idx == 0:
            sample_return = {
                'short_caption': short_caption,
                'long_caption': long_caption,
                'short_results': short_results[:5],  # Return top 5 for sample
                'long_results': long_results[:5]
            }
    
    # Calculate final metrics
    short_recalls = calculate_recall_at_k(short_caption_results, total_queries, k_values)
    long_recalls = calculate_recall_at_k(long_caption_results, total_queries, k_values)
    
    # Calculate average recalls
    avg_recalls = {}
    for k in k_values:
        avg_recalls[f'recall@{k}'] = (short_recalls[f'recall@{k}'] + long_recalls[f'recall@{k}']) / 2
    
    results = {
        'split': split,
        'total_queries': total_queries,
        'short_caption_recalls': short_recalls,
        'long_caption_recalls': long_recalls,
        'average_recalls': avg_recalls,
        'k_values': k_values
    }
    
    if verbose:
        print(f"\n=== {split.upper()} SPLIT RESULTS ===")
        print(f"Total queries: {total_queries}")
        print("\nShort Caption Results:")
        for k in k_values:
            print(f"  Recall@{k}: {short_recalls[f'recall@{k}']:.4f}")
        print("\nLong Caption Results:")
        for k in k_values:
            print(f"  Recall@{k}: {long_recalls[f'recall@{k}']:.4f}")
        print("\nAverage Results:")
        for k in k_values:
            print(f"  Recall@{k}: {avg_recalls[f'recall@{k}']:.4f}")
    
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
            for k_metric, value in avg_recalls.items():
                print(f"  {k_metric}: {value:.4f}")
    
    # Compare train vs test if both exist
    if 'train_results' in results and 'test_results' in results:
        print(f"\nTrain vs Test Comparison:")
        train_avg = results['train_results']['average_recalls']
        test_avg = results['test_results']['average_recalls']
        
        for k_metric in train_avg.keys():
            train_val = train_avg[k_metric]
            test_val = test_avg[k_metric]
            diff = train_val - test_val
            print(f"  {k_metric}: Train={train_val:.4f}, Test={test_val:.4f}, Diff={diff:+.4f}")
            
            
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
