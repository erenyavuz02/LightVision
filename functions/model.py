import os
import subprocess
from pathlib import Path
import mobileclip
import torch

def download_base_model(config, verbose=True):
    """
    Download the pretrained MobileCLIP model if it doesn't exist.
    
    Args:
        config: ConfigManager object with model configuration
        verbose: Whether to print progress messages
    
    Returns:
        bool: True if model is available (downloaded or already exists), False if download failed
    """
    try:
        # Get paths from config
        project_root = config.get('project.root')
        checkpoint_dir = os.path.join(project_root, config.get('model.checkpoint_dir'))
        base_model_path = os.path.join(project_root, config.get('model.base_model_path'))
        download_url = config.get('model.download_url')
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check if base model already exists
        if os.path.exists(base_model_path):
            if verbose:
                print("✅ Base model already exists.")
            return True
        
        # Download base model
        if verbose:
            print("📥 Downloading base MobileCLIP model...")
        
        # Use wget to download the model
        result = subprocess.run(
            ["wget", download_url, "-P", checkpoint_dir],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            if verbose:
                print("✅ Base model downloaded successfully!")
            return True
        else:
            if verbose:
                print(f"❌ Download failed: {result.stderr}")
            return False
            
    except Exception as e:
        if verbose:
            print(f"❌ Error downloading base model: {str(e)}")
        return False

def load_model(config, verbose=True):
    """
    Load the MobileCLIP model from the specified path.
    
    Args:
        config: ConfigManager object with model configuration
        verbose: Whether to print progress messages
    
    Returns:
        tuple: (model, preprocess, tokenizer) or (None, None, None) if loading failed
    """
    try:
        # Get model configuration
        project_root = config.get('project.root')
        base_model_path = os.path.join(project_root, config.get('model.base_model_path'))
        model_name = config.get('model.name')
        
        # Check if CUDA is available and update config accordingly
        if torch.cuda.is_available():
            device = 'cuda'
            config.update('model.device', 'cuda')
            if verbose:
                print(f"🚀 CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            config.update('model.device', 'cpu')
            if verbose:
                print("💻 CUDA not available, using CPU")
        
        if verbose:
            print(f"Loading {model_name} model...")
        
        if not os.path.exists(base_model_path):
            if verbose:
                print("❌ Base model file does not exist. Please download it first.")
            return None, None, None
        
        # Create model and transforms
        if verbose:
            print(f"Loading from checkpoint: {base_model_path}")
        
        model, _, preprocess = mobileclip.create_model_and_transforms(
            model_name, pretrained=base_model_path
        )
        
        # Move model to device
        model.to(device)
        
        # Get tokenizer
        tokenizer = mobileclip.get_tokenizer(model_name)
        
        if verbose:
            print(f"✅ Model loaded successfully on {device}")
        
        return model, preprocess, tokenizer
    
    except Exception as e:
        if verbose:
            print(f"❌ Error loading model: {str(e)}")
        return None, None, None