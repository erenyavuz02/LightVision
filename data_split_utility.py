"""
Data Split Utility for LightVision

This module handles splitting the Flickr8K dataset with LLaVA captions 
into training and testing sets for proper evaluation.
"""

import json
import os
import random
from typing import Dict, Tuple, List
import shutil
from collections import defaultdict

class DatasetSplitter:
    """Handles splitting dataset into train/test sets"""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducible splits"""
        random.seed(seed)
        self.seed = seed
    
    def load_llava_captions(self, captions_file: str) -> Dict:
        """Load LLaVA-generated captions from JSON file"""
        print(f"Loading captions from {captions_file}...")
        
        with open(captions_file, 'r') as f:
            captions_data = json.load(f)
        
        print(f"Loaded {len(captions_data)} image-caption pairs")
        return captions_data
    
    def analyze_caption_stats(self, captions_data: Dict):
        """Analyze caption statistics"""
        short_lengths = []
        long_lengths = []
        
        for image_name, captions in captions_data.items():
            if 'short_caption' in captions:
                short_lengths.append(len(captions['short_caption'].split()))
            if 'long_detailed' in captions:
                long_lengths.append(len(captions['long_detailed'].split()))
        
        stats = {
            'total_images': len(captions_data),
            'short_captions': {
                'count': len(short_lengths),
                'avg_length': sum(short_lengths) / len(short_lengths) if short_lengths else 0,
                'min_length': min(short_lengths) if short_lengths else 0,
                'max_length': max(short_lengths) if short_lengths else 0
            },
            'long_captions': {
                'count': len(long_lengths),
                'avg_length': sum(long_lengths) / len(long_lengths) if long_lengths else 0,
                'min_length': min(long_lengths) if long_lengths else 0,
                'max_length': max(long_lengths) if long_lengths else 0
            }
        }
        
        return stats
    
    def clean_and_standardize_captions(self, captions_data: Dict) -> Dict:
        """Clean and standardize caption format"""
        cleaned_data = {}
        
        for image_name, captions in captions_data.items():
            cleaned_captions = {}
            
            # Handle different possible keys
            if 'short_caption' in captions:
                cleaned_captions['short_caption'] = captions['short_caption'].strip()
            
            # Handle long caption (could be 'long_detailed' or 'long_caption')
            if 'long_detailed' in captions:
                cleaned_captions['long_caption'] = captions['long_detailed'].strip()
            elif 'long_caption' in captions:
                cleaned_captions['long_caption'] = captions['long_caption'].strip()
            
            # Only keep images that have both caption types
            if 'short_caption' in cleaned_captions and 'long_caption' in cleaned_captions:
                cleaned_data[image_name] = cleaned_captions
        
        print(f"Cleaned data: {len(cleaned_data)} images with both short and long captions")
        return cleaned_data
    
    def create_train_test_split(self, captions_data: Dict, 
                               test_ratio: float = 0.125,  # 1k out of 8k = 0.125
                               images_dir: str = None) -> Tuple[Dict, Dict]:
        """
        Split data into training and testing sets
        
        Args:
            captions_data: Dictionary of image captions
            test_ratio: Ratio of data to use for testing (default 0.125 for 1k/8k)
            images_dir: Path to images directory (to verify image existence)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        
        # Filter to only include images that actually exist
        if images_dir:
            existing_images = {}
            for image_name, captions in captions_data.items():
                image_path = os.path.join(images_dir, image_name)
                if os.path.exists(image_path):
                    existing_images[image_name] = captions
            
            print(f"Found {len(existing_images)} existing images out of {len(captions_data)}")
            captions_data = existing_images
        
        # Get list of all image names
        all_images = list(captions_data.keys())
        random.shuffle(all_images)
        
        # Calculate split sizes
        total_images = len(all_images)
        test_size = int(total_images * test_ratio)
        train_size = total_images - test_size
        
        print(f"Total images: {total_images}")
        print(f"Training set: {train_size} images ({(train_size/total_images)*100:.1f}%)")
        print(f"Test set: {test_size} images ({(test_size/total_images)*100:.1f}%)")
        
        # Split the data
        test_images = all_images[:test_size]
        train_images = all_images[test_size:]
        
        # Create train and test dictionaries
        train_data = {img: captions_data[img] for img in train_images}
        test_data = {img: captions_data[img] for img in test_images}
        
        return train_data, test_data
    
    def save_split_data(self, train_data: Dict, test_data: Dict, 
                       output_dir: str = "data"):
        """Save train and test splits to separate files"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        train_file = os.path.join(output_dir, "train_captions.json")
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Training data saved to: {train_file}")
        
        # Save test data
        test_file = os.path.join(output_dir, "test_captions.json")
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Test data saved to: {test_file}")
        
        # Create a metadata file
        metadata = {
            'split_info': {
                'seed': self.seed,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'test_ratio': len(test_data) / (len(train_data) + len(test_data))
            },
            'files': {
                'train_captions': 'train_captions.json',
                'test_captions': 'test_captions.json'
            }
        }
        
        metadata_file = os.path.join(output_dir, "split_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Split metadata saved to: {metadata_file}")
        
        return train_file, test_file, metadata_file

def create_evaluation_dataset(images_dir: str, captions_file: str, 
                            output_dir: str = "data", test_ratio: float = 0.125):
    """
    Complete function to create train/test split from LLaVA captions
    
    Args:
        images_dir: Directory containing Flickr8K images
        captions_file: JSON file with LLaVA-generated captions
        output_dir: Output directory for split files
        test_ratio: Ratio for test set (default 0.125 = 1k/8k)
    """
    
    print("="*60)
    print("CREATING EVALUATION DATASET")
    print("="*60)
    
    # Initialize splitter
    splitter = DatasetSplitter(seed=42)
    
    # Load and analyze data
    captions_data = splitter.load_llava_captions(captions_file)
    
    print("\nOriginal caption statistics:")
    stats = splitter.analyze_caption_stats(captions_data)
    print(f"  Total images: {stats['total_images']}")
    print(f"  Short captions: {stats['short_captions']['count']} (avg: {stats['short_captions']['avg_length']:.1f} words)")
    print(f"  Long captions: {stats['long_captions']['count']} (avg: {stats['long_captions']['avg_length']:.1f} words)")
    
    # Clean and standardize
    print("\nCleaning and standardizing captions...")
    cleaned_data = splitter.clean_and_standardize_captions(captions_data)
    
    # Create train/test split
    print("\nCreating train/test split...")
    train_data, test_data = splitter.create_train_test_split(
        cleaned_data, test_ratio=test_ratio, images_dir=images_dir
    )
    
    # Save split data
    print("\nSaving split data...")
    train_file, test_file, metadata_file = splitter.save_split_data(
        train_data, test_data, output_dir
    )
    
    # Analyze splits
    print("\nSplit analysis:")
    train_stats = splitter.analyze_caption_stats(train_data)
    test_stats = splitter.analyze_caption_stats(test_data)
    
    print(f"Training set:")
    print(f"  Images: {train_stats['total_images']}")
    print(f"  Short captions avg length: {train_stats['short_captions']['avg_length']:.1f} words")
    print(f"  Long captions avg length: {train_stats['long_captions']['avg_length']:.1f} words")
    
    print(f"Test set:")
    print(f"  Images: {test_stats['total_images']}")
    print(f"  Short captions avg length: {test_stats['short_captions']['avg_length']:.1f} words")
    print(f"  Long captions avg length: {test_stats['long_captions']['avg_length']:.1f} words")
    
    return {
        'train_file': train_file,
        'test_file': test_file,
        'metadata_file': metadata_file,
        'train_data': train_data,
        'test_data': test_data
    }

def convert_to_standard_format(llava_captions_file: str, output_file: str):
    """
    Convert LLaVA format to standard format for compatibility
    
    Args:
        llava_captions_file: Input file with LLaVA captions
        output_file: Output file in standard format
    """
    
    with open(llava_captions_file, 'r') as f:
        llava_data = json.load(f)
    
    # Convert format
    standard_data = {}
    for image_name, captions in llava_data.items():
        if isinstance(captions, dict):
            # Handle LLaVA format
            if 'short_caption' in captions and 'long_detailed' in captions:
                standard_data[image_name] = {
                    'short_caption': captions['short_caption'],
                    'long_caption': captions['long_detailed']  # Rename to standard format
                }
        else:
            # Handle simple string format
            standard_data[image_name] = captions
    
    # Save in standard format
    with open(output_file, 'w') as f:
        json.dump(standard_data, f, indent=2)
    
    print(f"Converted {len(standard_data)} captions to standard format")
    print(f"Saved to: {output_file}")
    
    return standard_data

def main():
    """Example usage"""
    
    # Configuration (update these paths to match your setup)
    images_dir = "/content/data/Images"  # Your Flickr8K images directory
    captions_file = "/content/data/captions_database.json"  # Your LLaVA captions file
    output_dir = "/content/data"
    
    # Check if files exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(captions_file):
        print(f"Error: Captions file not found: {captions_file}")
        return
    
    # Create evaluation dataset
    result = create_evaluation_dataset(
        images_dir=images_dir,
        captions_file=captions_file,
        output_dir=output_dir,
        test_ratio=0.125  # 1k out of 8k for testing
    )
    
    print("\n" + "="*60)
    print("DATASET SPLIT COMPLETED!")
    print("="*60)
    print("Generated files:")
    print(f"  Training data: {result['train_file']}")
    print(f"  Test data: {result['test_file']}")
    print(f"  Metadata: {result['metadata_file']}")
    print("\nYou can now use:")
    print("  - train_captions.json for model training")
    print("  - test_captions.json for evaluation")

if __name__ == "__main__":
    main()