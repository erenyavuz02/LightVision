import os
import requests
import zipfile
from pathlib import Path
from torch.utils.data import Dataset
import json
import random
import numpy as np
from PIL import Image
import torch

class DatasetDownloader:
    """A class to handle dataset downloading and extraction operations."""
    
    def __init__(self, config):
        """Initialize the DatasetDownloader with configuration."""
        self.config = config
    
    def download_file(self, url: str, destination_path: Path, verbose: bool = True) -> None:
        """Download a file from URL to destination path with progress indication."""
        if verbose:
            print(f"Downloading from {url}")
            print(f"Saving to: {destination_path}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress tracking
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(destination_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Simple progress indication
                        if verbose and total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end="", flush=True)
            
            if verbose:
                print(f"\nDownload complete: {destination_path}")
            
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Download error: {e}")
            raise
        except Exception as e:
            if verbose:
                print(f"Unexpected error during download: {e}")
            raise

    def extract_zip_file(self, zip_path: Path, destination_folder: Path, verbose: bool = True) -> None:
        """Extract a zip file to the destination folder."""
        if verbose:
            print(f"Extracting {zip_path} to {destination_folder}...")
        
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                if verbose:
                    file_list = zip_ref.namelist()
                    print(f"Extracting {len(file_list)} files...")
                
                zip_ref.extractall(destination_folder)
                
            if verbose:
                print("Extraction complete.")
            
        except zipfile.BadZipFile:
            if verbose:
                print(f"Error: {zip_path} is not a valid zip file")
            raise
        except Exception as e:
            if verbose:
                print(f"Extraction error: {e}")
            raise

    def download_dataset(self, verbose=False) -> bool:
        """Download and extract the Flickr8k dataset using configuration."""
        try:
            # Get config values
            project_root = Path(self.config.get('project.root', '.'))
            dataset_config = self.config.get_section('dataset')
            
            # Setup paths
            data_folder = project_root / "data"
            images_folder_name = dataset_config.get('images_folder_name', 'Images')
            output_folder_name = dataset_config.get('output_folder_name', 'output')
            zip_filename = dataset_config.get('zip_filename', 'flickr8k.zip')
            download_url = dataset_config.get('download_url')
            
            # Create directories
            images_path = data_folder / images_folder_name
            output_path = data_folder / output_folder_name
            data_folder.mkdir(exist_ok=True)
            output_path.mkdir(exist_ok=True)
            
            if verbose:
                print(f"Starting download process for {dataset_config.get('name', 'dataset')}...")
            
            # Check if dataset already exists
            if images_path.exists() and any(images_path.iterdir()):
                if verbose:
                    image_count = len(list(images_path.glob('*')))
                    print(f"Dataset already exists: {image_count} files found")
                return True
            
            # Check download URL
            if not download_url:
                if verbose:
                    print("Error: No download URL provided in configuration")
                return False
            
            # Download and extract
            zip_file_path = data_folder / zip_filename
            
            if verbose:
                print("Downloading dataset...")
            self.download_file(download_url, zip_file_path, verbose)
            
            if verbose:
                print("Extracting dataset...")
            self.extract_zip_file(zip_file_path, data_folder, verbose)
            
            # Clean up zip file
            zip_file_path.unlink()
            
            # Verify extraction
            if images_path.exists():
                if verbose:
                    image_count = len(list(images_path.glob('*')))
                    print(f"✅ Dataset setup complete! Found {image_count} files")
                return True
            else:
                if verbose:
                    print("⚠️ Warning: Images directory not found after extraction")
                return False
                
        except Exception as e:
            if verbose:
                print(f"❌ Failed to set up dataset: {e}")
            return False


class CustomDataset(Dataset):
    """Dataset class for loading images and captions with train/test splitting"""
    
    def __init__(self, config, test_ratio=0.125, transform=None, device=None, force_resplit=False):
        """
        Initialize the CustomDataset
        
        Args:
            config: Configuration object containing paths and settings
            test_ratio: Ratio of data to use for testing (default 0.125)
            transform: Optional transform to apply to images
            device: torch device to load images to (cuda/cpu)
            force_resplit: Force recreation of the split even if it exists
        """
        self.config = config
        self.test_ratio = test_ratio
        self.transform = transform
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get paths from config
        project_root = config.get('project.root', '.')
        self.images_dir = os.path.join(project_root, 'data', 'Images')
        self.captions_file = os.path.join(project_root, 'data', 'custom_captions.json')
        self.split_cache_file = os.path.join(project_root, 'data', 'dataset_split_cache.json')
        
        # Load and split data
        self._load_and_split_data(force_resplit)
        
    def _create_split(self, all_captions, force_resplit=False):
        """Create or load the train/test split"""
        
        # Check if we can load existing split
        if not force_resplit and os.path.exists(self.split_cache_file):
            try:
                with open(self.split_cache_file, 'r') as f:
                    cached_split = json.load(f)
                
                # Verify the cached split is compatible
                if (cached_split.get('test_ratio') == self.test_ratio and
                    cached_split.get('total_images') == len(all_captions)):
                    
                    self.train_images = set(cached_split['train_images'])
                    self.test_images = set(cached_split['test_images'])
                    self.split_metadata = cached_split
                    return
            except Exception:
                pass  # Create new split if loading fails
        
        # Create new split
        all_image_names = list(all_captions.keys())
        
        # Create reproducible split
        random.seed(42)
        random.shuffle(all_image_names)
        
        # Calculate split sizes
        total_size = len(all_image_names)
        test_size = int(total_size * self.test_ratio)
        
        # Split the data
        self.test_images = set(all_image_names[:test_size])
        self.train_images = set(all_image_names[test_size:])
        
        # Create metadata
        self.split_metadata = {
            'test_ratio': self.test_ratio,
            'total_images': total_size,
            'train_size': len(self.train_images),
            'test_size': len(self.test_images),
            'seed': 42,
            'train_images': list(self.train_images),
            'test_images': list(self.test_images)
        }
        
        # Save split cache
        try:
            with open(self.split_cache_file, 'w') as f:
                json.dump(self.split_metadata, f, indent=2)
        except Exception:
            pass  # Continue if can't save cache
        
    def _load_and_split_data(self, force_resplit=False):
        """Load captions and create train/test split"""
        
        # Load captions from JSON file
        if not os.path.exists(self.captions_file):
            raise FileNotFoundError(f"Captions file not found: {self.captions_file}")
            
        with open(self.captions_file, 'r') as f:
            all_captions = json.load(f)
        
        # Filter to only include images that exist and have both caption types
        self.valid_captions = {}
        for image_name, captions in all_captions.items():
            image_path = os.path.join(self.images_dir, image_name)
            
            # Check if image exists
            if not os.path.exists(image_path):
                continue
                
            # Extract captions based on the JSON structure
            short_caption = None
            long_caption = None
            
            if isinstance(captions, dict):
                short_caption = captions.get('short_caption')
                long_caption = captions.get('long_detailed')
                
                # Fallback to other possible keys
                if not short_caption:
                    short_caption = captions.get('short')
                if not long_caption:
                    long_caption = captions.get('long_caption') or captions.get('long')
                    
            elif isinstance(captions, str):
                short_caption = captions
                long_caption = captions
            
            # Only include if we have both caption types
            if short_caption and long_caption:
                self.valid_captions[image_name] = {
                    'short_caption': short_caption.strip(),
                    'long_caption': long_caption.strip()
                }
        
        if not self.valid_captions:
            raise ValueError("No valid image-caption pairs found")
        
        # Create split
        self._create_split(self.valid_captions, force_resplit)
        
        # Prepare data for both splits
        self._prepare_split_data()
    
    def _prepare_split_data(self):
        """Prepare data arrays for both train and test splits"""
        self.train_data = []
        self.test_data = []
        
        for image_name, captions in self.valid_captions.items():
            item = {
                'image_name': image_name,
                'image_path': os.path.join(self.images_dir, image_name),
                'short_caption': captions['short_caption'],
                'long_caption': captions['long_caption']
            }
            
            if image_name in self.train_images:
                self.train_data.append(item)
            elif image_name in self.test_images:
                self.test_data.append(item)
    
    def get_split_info(self):
        """Get information about the current split"""
        return self.split_metadata.copy() if hasattr(self, 'split_metadata') else None
    
    def verify_no_overlap(self):
        """Verify there's no overlap between train and test splits"""
        overlap = self.train_images.intersection(self.test_images)
        if overlap:
            print(f"WARNING: Found {len(overlap)} overlapping images!")
            return False
        return True
    
    def get_dataloader(self, split='train', batch_size=32, shuffle=None, num_workers=0):
        """Get a PyTorch DataLoader for specified split"""
        from torch.utils.data import DataLoader
        
        if split == 'train':
            indices = list(range(len(self.train_data)))
            dataset = TrainTestSubset(self, self.train_data, indices)
        elif split == 'test':
            indices = list(range(len(self.test_data)))
            dataset = TrainTestSubset(self, self.test_data, indices)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'")
        
        if shuffle is None:
            shuffle = (split == 'train')
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
    
    def get_statistics(self, split='both'):
        """Get dataset statistics"""
        def calc_stats(data):
            if not data:
                return {'total_samples': 0, 'short_captions': {}, 'long_captions': {}}
                
            short_lengths = [len(item['short_caption'].split()) for item in data]
            long_lengths = [len(item['long_caption'].split()) for item in data]
            
            return {
                'total_samples': len(data),
                'short_captions': {
                    'avg_length': sum(short_lengths) / len(short_lengths),
                    'min_length': min(short_lengths),
                    'max_length': max(short_lengths)
                },
                'long_captions': {
                    'avg_length': sum(long_lengths) / len(long_lengths),
                    'min_length': min(long_lengths),
                    'max_length': max(long_lengths)
                }
            }
        
        if split == 'train':
            return calc_stats(self.train_data)
        elif split == 'test':
            return calc_stats(self.test_data)
        elif split == 'both':
            return {
                'train': calc_stats(self.train_data),
                'test': calc_stats(self.test_data),
                'total': calc_stats(self.train_data + self.test_data)
            }
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'test', or 'both'")
    
    def get_all_captions(self, split='train', caption_type='short'):
        """Get all captions of specified type for specified split"""
        data = self.train_data if split == 'train' else self.test_data
        
        if caption_type == 'short':
            return [item['short_caption'] for item in data]
        elif caption_type == 'long':
            return [item['long_caption'] for item in data]
        else:
            raise ValueError(f"Invalid caption_type: {caption_type}. Must be 'short' or 'long'")
    
    def get_image_paths(self, split='train'):
        """Get all image paths for specified split"""
        data = self.train_data if split == 'train' else self.test_data
        return [item['image_path'] for item in data]
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        # Stack tensors if they're already tensors, otherwise keep as list
        images = []
        for item in batch:
            img = item['image']
            if isinstance(img, torch.Tensor):
                images.append(img)
            else:
                images.append(img)  # Keep PIL images as list
        
        # Try to stack if all are tensors
        if all(isinstance(img, torch.Tensor) for img in images):
            try:
                images = torch.stack(images)
            except:
                pass  # Keep as list if stacking fails
        
        return {
            'images': images,
            'image_names': [item['image_name'] for item in batch],
            'image_paths': [item['image_path'] for item in batch],
            'short_captions': [item['short_caption'] for item in batch],
            'long_captions': [item['long_caption'] for item in batch]
        }


class TrainTestSubset:
    """Helper class to create a subset for train/test data"""
    
    def __init__(self, parent_dataset, data, indices):
        self.parent_dataset = parent_dataset
        self.data = data
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        item = self.data[actual_idx]
        
        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            
            # Apply transform if provided
            if self.parent_dataset.transform:
                image = self.parent_dataset.transform(image)
            
            # Move to device if image is a tensor
            if isinstance(image, torch.Tensor) and self.parent_dataset.device:
                image = image.to(self.parent_dataset.device)
                
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a dummy image
            image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
            if self.parent_dataset.transform:
                image = self.parent_dataset.transform(image)
            if isinstance(image, torch.Tensor) and self.parent_dataset.device:
                image = image.to(self.parent_dataset.device)
        
        return {
            'image': image,
            'image_name': item['image_name'],
            'image_path': item['image_path'],
            'short_caption': item['short_caption'],
            'long_caption': item['long_caption']
        }