"""
Setup and Implementation Guide for LightVision Retrieval System

This script provides step-by-step guidance and utility functions for setting up
and running the LightVision image retrieval and evaluation system.
"""

import os
import sys
import json
import shutil
from typing import Dict, List
import subprocess

class LightVisionSetup:
    """Setup utility for LightVision system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.required_dirs = [
            "data/Images",
            "data",
            "checkpoints",
            "evaluation_results",
            "logs"
        ]
        
    def check_dependencies(self) -> bool:
        """Check if all required packages are installed"""
        required_packages = [
            "torch", "torchvision", "numpy", "PIL", "tqdm", 
            "faiss-cpu", "matplotlib", "seaborn", "pandas", 
            "scikit-learn", "mobileclip"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == "PIL":
                    import PIL
                elif package == "mobileclip":
                    # This might need to be installed from your local path
                    print("Note: Make sure mobileclip is available in your Python path")
                else:
                    __import__(package)
                print(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"✗ {package} - MISSING")
        
        if missing_packages:
            print(f"\nMissing packages: {missing_packages}")
            print("Install them using: pip install " + " ".join(missing_packages))
            return False
        
        print("All dependencies are satisfied!")
        return True
    
    def create_directory_structure(self):
        """Create required directory structure"""
        print("Creating directory structure...")
        
        for dir_path in self.required_dirs:
            full_path = os.path.join(self.project_root, dir_path)
            os.makedirs(full_path, exist_ok=True)
            print(f"✓ Created: {full_path}")
    
    def setup_data_structure(self):
        """Guide user through data setup"""
        print("\n" + "="*50)
        print("DATA SETUP GUIDE")
        print("="*50)
        
        data_dir = os.path.join(self.project_root, "data")
        images_dir = os.path.join(data_dir, "Images")
        
        print(f"\n1. Image Data:")
        print(f"   - Place your test images in: {images_dir}")
        print(f"   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        
        print(f"\n2. Caption Data:")
        print(f"   - Place your captions file in: {data_dir}")
        print(f"   - Supported formats:")
        print(f"     • JSON format: {data_dir}/captions.json")
        print(f"       Example: {{'image1.jpg': 'caption text', ...}}")
        print(f"     • Text format: {data_dir}/captions.txt")
        print(f"       Example: image1.jpg\\tcaption text")
        
        print(f"\n3. Model Checkpoints:")
        checkpoints_dir = os.path.join(self.project_root, "checkpoints")
        print(f"   - Place your trained model checkpoints in: {checkpoints_dir}")
        print(f"   - Example: mobileclip_finetuned_epoch1_last.pt")
        
        return data_dir, images_dir, checkpoints_dir
    
    def validate_data(self, data_dir: str, images_dir: str) -> bool:
        """Validate that data is properly set up"""
        print("\nValidating data setup...")
        
        # Check if images directory exists and has images
        if not os.path.exists(images_dir):
            print(f"✗ Images directory not found: {images_dir}")
            return False
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"✗ No image files found in: {images_dir}")
            return False
        
        print(f"✓ Found {len(image_files)} images")
        
        # Check for captions file
        captions_files = [
            os.path.join(data_dir, "captions.json"),
            os.path.join(data_dir, "captions.txt"),
            os.path.join(data_dir, "all_captions.json")
        ]
        
        captions_found = False
        for captions_file in captions_files:
            if os.path.exists(captions_file):
                print(f"✓ Found captions file: {captions_file}")
                captions_found = True
                break
        
        if not captions_found:
            print(f"✗ No captions file found. Looking for: {captions_files}")
            return False
        
        return True
    
    def create_sample_config(self):
        """Create sample configuration files"""
        config_dir = os.path.join(self.project_root, "configs")
        os.makedirs(config_dir, exist_ok=True)
        
        # Sample retrieval config
        retrieval_config = {
            "model_name": "mobileclip_s0",
            "checkpoint_path": "checkpoints/mobileclip_finetuned_epoch1_last.pt",
            "device": "cuda",
            "embedding_dim": 512,
            "batch_size": 32,
            "top_k": 10
        }
        
        with open(os.path.join(config_dir, "retrieval_config.json"), 'w') as f:
            json.dump(retrieval_config, f, indent=2)
        
        # Sample evaluation config
        eval_config = {
            "test_image_dir": "data/Images",
            "test_captions_file": "data/captions.txt",
            "output_dir": "evaluation_results",
            "k_values": [1, 5, 10],
            "batch_size": 32
        }
        
        with open(os.path.join(config_dir, "evaluation_config.json"), 'w') as f:
            json.dump(eval_config, f, indent=2)
        
        print(f"✓ Created sample configs in: {config_dir}")

def step_by_step_guide():
    """Provide step-by-step implementation guide"""
    
    print("\n" + "="*60)
    print("LIGHTVISION IMPLEMENTATION GUIDE")
    print("="*60)
    
    steps = [
        {
            "title": "1. Environment Setup",
            "description": "Set up Python environment and install dependencies",
            "actions": [
                "Create virtual environment: python -m venv lightvision_env",
                "Activate environment: source lightvision_env/bin/activate (Linux/Mac) or lightvision_env\\Scripts\\activate (Windows)",
                "Install PyTorch: pip install torch torchvision",
                "Install other dependencies: pip install faiss-cpu matplotlib seaborn pandas scikit-learn pillow tqdm",
                "Ensure mobileclip is in Python path"
            ]
        },
        {
            "title": "2. Project Structure",
            "description": "Set up project directory structure",
            "actions": [
                "Run: python setup_and_implementation_guide.py --setup",
                "This will create all necessary directories"
            ]
        },
        {
            "title": "3. Data Preparation",
            "description": "Prepare your image and caption data",
            "actions": [
                "Copy your images to data/Images/",
                "Prepare captions file (JSON or TXT format)",
                "Place trained model checkpoints in checkpoints/",
                "Validate setup: python setup_and_implementation_guide.py --validate"
            ]
        },
        {
            "title": "4. Basic Retrieval",
            "description": "Test basic retrieval functionality",
            "actions": [
                "Run: python retrieval_framework.py",
                "This will start interactive retrieval system",
                "Test with simple queries like 'a dog' or 'people walking'"
            ]
        },
        {
            "title": "5. Model Evaluation",
            "description": "Evaluate model performance",
            "actions": [
                "Quick demo: python evaluation_system.py --mode demo",
                "Full evaluation: python evaluation_system.py --mode full",
                "Query length analysis: python evaluation_system.py --mode length_analysis"
            ]
        },
        {
            "title": "6. Results Analysis",
            "description": "Analyze evaluation results",
            "actions": [
                "Check evaluation_results/ directory for outputs",
                "View evaluation_summary.csv for metric comparison",
                "Check visualization files: model_comparison.png, performance_heatmap.png"
            ]
        }
    ]
    
    for i, step in enumerate(steps):
        print(f"\n{step['title']}")
        print("-" * len(step['title']))
        print(step['description'])
        print("\nActions:")
        for action in step['actions']:
            print(f"  • {action}")
        
        if i < len(steps) - 1:
            input("\nPress Enter to continue to next step...")

def quick_test():
    """Run a quick test to verify everything is working"""
    print("\n" + "="*40)
    print("QUICK SYSTEM TEST")
    print("="*40)
    
    try:
        # Test imports
        print("Testing imports...")
        import torch
        import numpy as np
        import faiss
        from PIL import Image
        print("✓ Core imports successful")
        
        # Test model loading (if available)
        try:
            import mobileclip
            print("✓ MobileCLIP import successful")
        except ImportError:
            print("✗ MobileCLIP not available - make sure it's in your Python path")
        
        # Test FAISS
        print("Testing FAISS...")
        test_embeddings = np.random.random((10, 512)).astype(np.float32)
        index = faiss.IndexFlatIP(512)
        index.add(test_embeddings)
        query = np.random.random((1, 512)).astype(np.float32)
        scores, indices = index.search(query, 3)
        print("✓ FAISS working correctly")
        
        print("\n✓ All systems operational!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return False

def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LightVision Setup and Guide')
    parser.add_argument('--setup', action='store_true', 
                       help='Run initial setup')
    parser.add_argument('--validate', action='store_true',
                       help='Validate data setup')
    parser.add_argument('--guide', action='store_true',
                       help='Show implementation guide')
    parser.add_argument('--test', action='store_true',
                       help='Run quick system test')
    parser.add_argument('--project-root', type=str, default='.',
                       help='Project root directory')
    
    args = parser.parse_args()
    
    setup = LightVisionSetup(args.project_root)
    
    if args.setup:
        print("Running LightVision setup...")
        setup.check_dependencies()
        setup.create_directory_structure()
        setup.create_sample_config()
        data_dir, images_dir, checkpoints_dir = setup.setup_data_structure()
        print("\nSetup complete! Follow the data setup guide above.")
        
    elif args.validate:
        print("Validating LightVision setup...")
        data_dir = os.path.join(args.project_root, "data")
        images_dir = os.path.join(data_dir, "Images")
        
        if setup.validate_data(data_dir, images_dir):
            print("\n✓ Data validation successful!")
        else:
            print("\n✗ Data validation failed. Please check the setup.")
            
    elif args.guide:
        step_by_step_guide()
        
    elif args.test:
        quick_test()
        
    else:
        print("LightVision Setup Utility")
        print("Use --help to see available options")
        print("\nQuick start:")
        print("  python setup_and_implementation_guide.py --setup")
        print("  python setup_and_implementation_guide.py --guide")

if __name__ == "__main__":
    main()