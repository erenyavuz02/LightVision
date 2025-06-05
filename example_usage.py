"""
Example Usage of LightVision Retrieval and Evaluation System

This script demonstrates practical usage of both the retrieval framework
and evaluation system with your LightVision project.
"""

import os
import sys
import json
from typing import Dict, List

# Import your modules
from retrieval_framework import LightVisionRetrieval, RetrievalConfig, ImageEmbeddingDatabase
from evaluation_system import ModelEvaluator, EvaluationConfig, create_model_configs

def example_1_basic_retrieval():
    """Example 1: Basic image retrieval with a single model"""
    
    print("="*60)
    print("EXAMPLE 1: BASIC IMAGE RETRIEVAL")
    print("="*60)
    
    # Configure the retrieval system
    config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path='checkpoints/mobileclip_finetuned_epoch1_last.pt',  # Your trained model
        device='cuda',  # Use 'cpu' if no GPU
        top_k=5
    )
    
    # Initialize retrieval system
    retrieval_system = LightVisionRetrieval(config)
    
    # Load the model
    print("Loading model...")
    retrieval_system.load_model()
    
    # Build database from your images
    print("Building image database...")
    image_directory = "data/Images"  # Replace with your image directory
    captions_file = "data/all_captions.json"  # Replace with your captions file
    
    # Check if database already exists to save time
    db_path = "example_database.pkl"
    if os.path.exists(db_path):
        print("Loading existing database...")
        retrieval_system.load_database(db_path)
    else:
        print("Creating new database...")
        retrieval_system.build_database(image_directory, captions_file)
        retrieval_system.save_database(db_path)
    
    # Example queries
    test_queries = [
        "A man sitting on a bench",
        "A dog running in a park",
        "People walking on the street",
        "A car parked in front of a building"
    ]
    
    print("\nRunning example queries...")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retrieval_system.search(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Similarity: {result['similarity']:.4f}")
            print(f"     Image: {os.path.basename(result['image_path'])}")
            if result['metadata'].get('caption'):
                print(f"     Caption: {result['metadata']['caption'][:100]}...")

def example_2_model_comparison():
    """Example 2: Compare base model vs fine-tuned model"""
    
    print("\n" + "="*60)
    print("EXAMPLE 2: MODEL COMPARISON EVALUATION")
    print("="*60)
    
    # Set up evaluation configuration
    eval_config = EvaluationConfig(
        test_image_dir="data/Images",
        test_captions_file="data/all_captions.json",
        output_dir="comparison_results",
        k_values=[1, 5, 10]
    )
    
    # Create model configurations to compare
    model_configs = [
        (RetrievalConfig(
            model_name='mobileclip_s0',
            checkpoint_path=None,  # Base model
            device='cuda'
        ), "Base MobileCLIP"),
        
        (RetrievalConfig(
            model_name='mobileclip_s0',
            checkpoint_path='checkpoints/mobileclip_finetuned_epoch1_last.pt',  # Your fine-tuned model
            device='cuda'
        ), "Fine-tuned MobileCLIP")
    ]
    
    # Run evaluation
    evaluator = ModelEvaluator(eval_config)
    results = evaluator.compare_models(model_configs)
    
    # Print comparison summary
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"  Recall@1:  {model_results['mean_recall_at_1']:.4f}")
        print(f"  Recall@5:  {model_results['mean_recall_at_5']:.4f}")
        print(f"  Recall@10: {model_results['mean_recall_at_10']:.4f}")
        print(f"  mAP:       {model_results['mAP']:.4f}")
        print(f"  MRR:       {model_results['MRR']:.4f}")
        print(f"  Avg Time:  {model_results['avg_search_time']:.4f}s")

def example_3_query_length_analysis():
    """Example 3: Analyze performance on short vs long queries"""
    
    print("\n" + "="*60)
    print("EXAMPLE 3: QUERY LENGTH ANALYSIS")
    print("="*60)
    
    # Configuration
    model_config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path='checkpoints/mobileclip_finetuned_epoch1_last.pt',
        device='cuda'
    )
    
    eval_config = EvaluationConfig(
        test_image_dir="data/Images",
        test_captions_file="data/all_captions.json",
        output_dir="length_analysis_results"
    )
    
    # Import the analysis function
    from evaluation_system import analyze_query_length_performance
    
    # Run analysis
    length_results = analyze_query_length_performance(model_config, eval_config)
    
    # Print detailed comparison
    if len(length_results) >= 2:
        short_results = length_results.get("Short Queries", {})
        long_results = length_results.get("Long Queries", {})
        
        print("\n" + "="*40)
        print("QUERY LENGTH COMPARISON")
        print("="*40)
        
        metrics = ['mean_recall_at_1', 'mean_recall_at_5', 'mean_average_precision']
        
        for metric in metrics:
            if metric in short_results and metric in long_results:
                short_val = short_results[metric]
                long_val = long_results[metric]
                improvement = ((long_val - short_val) / short_val * 100) if short_val > 0 else 0
                
                print(f"\n{metric.replace('mean_', '').replace('_', ' ').title()}:")
                print(f"  Short queries: {short_val:.4f}")
                print(f"  Long queries:  {long_val:.4f}")
                print(f"  Improvement:   {improvement:+.2f}%")

def example_4_progressive_detail_test():
    """Example 4: Test the progressive detail recognition from your experiments"""
    
    print("\n" + "="*60)
    print("EXAMPLE 4: PROGRESSIVE DETAIL RECOGNITION TEST")
    print("="*60)
    
    # Load your fine-tuned model
    config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path='checkpoints/mobileclip_finetuned_epoch1_last.pt',
        device='cuda'
    )
    
    retrieval_system = LightVisionRetrieval(config)
    retrieval_system.load_model()
    
    # Test progressive captions (from your experiment 2)
    test_captions = [
        "A man.",
        "A man sitting on a bench.",
        "A man sitting on a red bench in a park.",
        "A man sitting on a red bench in a park holding a yellow umbrella.",
        "A man sitting on a red bench in a park holding a yellow umbrella while feeding pigeons."
    ]
    
    print("Testing progressive detail recognition...")
    print("Encoding captions and computing similarities...")
    
    # Encode all captions
    caption_embeddings = []
    for caption in test_captions:
        embedding = retrieval_system.encode_text(caption)
        caption_embeddings.append(embedding)
    
    # Compute similarities between consecutive captions
    import numpy as np
    
    print("\nProgressive Caption Analysis:")
    print("-" * 50)
    
    for i, caption in enumerate(test_captions):
        length = len(caption.split())
        print(f"Caption {i+1} ({length} words): {caption}")
        
        if i > 0:
            # Compute similarity with previous caption
            sim = np.dot(caption_embeddings[i].flatten(), caption_embeddings[i-1].flatten())
            print(f"  Similarity to previous: {sim:.4f}")
    
    print("\nThis test shows how well the model handles incremental detail addition.")

def example_5_batch_evaluation():
    """Example 5: Batch evaluation on your custom dataset"""
    
    print("\n" + "="*60)
    print("EXAMPLE 5: BATCH EVALUATION ON CUSTOM DATASET")
    print("="*60)
    
    # Load your custom dataset with short and long captions
    captions_file = "data/all_captions.json"
    
    if not os.path.exists(captions_file):
        print(f"Captions file not found: {captions_file}")
        print("Please ensure your captions file exists.")
        return
    
    with open(captions_file, 'r') as f:
        captions_data = json.load(f)
    
    print(f"Loaded {len(captions_data)} image-caption pairs")
    
    # Set up evaluation for both short and long captions
    eval_config = EvaluationConfig(
        test_image_dir="data/Images",
        test_captions_file=captions_file,
        output_dir="batch_evaluation_results",
        k_values=[1, 3, 5, 10]
    )
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path=None,  # Base model
        device='cuda'
    )
    
    evaluator = ModelEvaluator(eval_config)
    base_results = evaluator.evaluate_model(base_config, "Base MobileCLIP")
    
    # Evaluate fine-tuned model
    print("\nEvaluating fine-tuned model...")
    finetuned_config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path='checkpoints/mobileclip_finetuned_epoch1_last.pt',
        device='cuda'
    )
    
    finetuned_results = evaluator.evaluate_model(finetuned_config, "Fine-tuned MobileCLIP")
    
    # Compare results
    print("\n" + "="*50)
    print("BATCH EVALUATION COMPARISON")
    print("="*50)
    
    improvements = {}
    metrics_to_compare = ['mean_recall_at_1', 'mean_recall_at_5', 'mean_recall_at_10', 'mAP', 'MRR']
    
    for metric in metrics_to_compare:
        if metric in base_results and metric in finetuned_results:
            base_val = base_results[metric]
            finetuned_val = finetuned_results[metric]
            improvement = ((finetuned_val - base_val) / base_val * 100) if base_val > 0 else 0
            improvements[metric] = improvement
            
            print(f"\n{metric.replace('mean_', '').replace('_', ' ').title()}:")
            print(f"  Base model:       {base_val:.4f}")
            print(f"  Fine-tuned model: {finetuned_val:.4f}")
            print(f"  Improvement:      {improvement:+.2f}%")
    
    # Summary
    avg_improvement = np.mean(list(improvements.values()))
    print(f"\nAverage improvement: {avg_improvement:+.2f}%")
    
    return base_results, finetuned_results

def create_performance_report():
    """Create a comprehensive performance report"""
    
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE PERFORMANCE REPORT")
    print("="*60)
    
    report = {
        "project": "LightVision",
        "timestamp": "",
        "models_evaluated": [],
        "datasets": {},
        "results": {},
        "conclusions": []
    }
    
    # Run all evaluations and collect results
    try:
        print("Running comprehensive evaluation...")
        
        # Model comparison
        eval_config = EvaluationConfig(
            test_image_dir="data/Images",
            test_captions_file="data/all_captions.json",
            output_dir="comprehensive_report",
            k_values=[1, 5, 10]
        )
        
        model_configs = create_model_configs()
        evaluator = ModelEvaluator(eval_config)
        comparison_results = evaluator.compare_models(model_configs)
        
        # Add to report
        report["results"]["model_comparison"] = comparison_results
        
        # Generate conclusions
        if len(comparison_results) >= 2:
            base_model = list(comparison_results.keys())[0]
            finetuned_model = list(comparison_results.keys())[1]
            
            base_recall1 = comparison_results[base_model]['mean_recall_at_1']
            finetuned_recall1 = comparison_results[finetuned_model]['mean_recall_at_1']
            
            improvement = ((finetuned_recall1 - base_recall1) / base_recall1 * 100) if base_recall1 > 0 else 0
            
            if improvement > 0:
                report["conclusions"].append(f"Fine-tuning improved Recall@1 by {improvement:.2f}%")
            else:
                report["conclusions"].append(f"Fine-tuning decreased Recall@1 by {abs(improvement):.2f}%")
            
            # Check if fine-tuning helped with long queries
            base_map = comparison_results[base_model]['mAP']
            finetuned_map = comparison_results[finetuned_model]['mAP']
            map_improvement = ((finetuned_map - base_map) / base_map * 100) if base_map > 0 else 0
            
            if map_improvement > 0:
                report["conclusions"].append(f"Fine-tuning improved mAP by {map_improvement:.2f}%")
        
        # Save report
        import datetime
        report["timestamp"] = datetime.datetime.now().isoformat()
        
        with open("performance_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nPerformance report saved to: performance_report.json")
        print("Detailed results available in: comprehensive_report/")
        
        return report
        
    except Exception as e:
        print(f"Error creating performance report: {e}")
        return None

def main():
    """Run example usage scenarios"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='LightVision Example Usage')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5], 
                       help='Run specific example (1-5)')
    parser.add_argument('--all', action='store_true',
                       help='Run all examples')
    parser.add_argument('--report', action='store_true',
                       help='Create comprehensive performance report')
    
    args = parser.parse_args()
    
    if args.example == 1 or args.all:
        example_1_basic_retrieval()
    
    if args.example == 2 or args.all:
        example_2_model_comparison()
    
    if args.example == 3 or args.all:
        example_3_query_length_analysis()
    
    if args.example == 4 or args.all:
        example_4_progressive_detail_test()
    
    if args.example == 5 or args.all:
        example_5_batch_evaluation()
    
    if args.report:
        create_performance_report()
    
    if not any([args.example, args.all, args.report]):
        print("LightVision Example Usage")
        print("Available examples:")
        print("  1. Basic image retrieval")
        print("  2. Model comparison evaluation") 
        print("  3. Query length analysis")
        print("  4. Progressive detail recognition test")
        print("  5. Batch evaluation on custom dataset")
        print("\nUsage:")
        print("  python example_usage.py --example 1")
        print("  python example_usage.py --all")
        print("  python example_usage.py --report")

if __name__ == "__main__":
    main()