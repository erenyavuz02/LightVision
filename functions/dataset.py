import os
import requests
import zipfile
from pathlib import Path


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
        """
        Download and extract the Flickr8k dataset using configuration.
        
        Args:
            verbose: Whether to show detailed output
            
        Returns:
            bool: True if successful, False otherwise
        """
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
                print(f"Description: {dataset_config.get('description', 'No description available')}")
            
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
            if verbose:
                print("Cleanup complete.")
            
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
        
        
        
        
        
class CustomDataset:
    """
    This class loads the Dataset from the custom capitons json file
    splits it into training and testing parts
    
    """
    # TODO: complete this class