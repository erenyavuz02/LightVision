import os
import pickle
import faiss
import numpy as np
import torch
from pathlib import Path

class FAISSRetriever:
    """FAISS-based image retrieval system for evaluating image-text models"""
    
    def __init__(self, config, model, dataset, split='test'):
        """
        Initialize the FAISS retriever
        
        Args:
            config: Configuration object
            model: The loaded model (e.g., MobileCLIP)
            dataset: CustomDataset instance
            split: Which split to use ('train' or 'test')
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.split = split
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Paths for saving/loading FAISS index and metadata
        project_root = config.get('project.root', '.')
        self.cache_dir = os.path.join(project_root, 'data', 'faiss_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        model_name = config.get('model.name', 'mobileclip')
        self.index_path = os.path.join(self.cache_dir, f'{model_name}_{split}_index.faiss')
        self.metadata_path = os.path.join(self.cache_dir, f'{model_name}_{split}_metadata.pkl')
        
        self.index = None
        self.image_paths = []
        self.image_names = []
        self.embeddings = None
        
    def _extract_image_embeddings(self, verbose=True):
        """Extract embeddings for all images in the dataset split"""
        if verbose:
            print(f"Extracting image embeddings for {self.split} split...")
        
        # Get dataloader
        dataloader = self.dataset.get_dataloader(
            split=self.split, 
            batch_size=32, 
            shuffle=False,
            num_workers=0
        )
        
        embeddings_list = []
        image_paths_list = []
        image_names_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if verbose and batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                
                # Get images and move to device
                images = batch['images']
                if isinstance(images, list):
                    # Convert PIL images to tensor batch
                    images = torch.stack([img for img in images])
                images = images.to(self.device)
                
                # Extract image features
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
                
                embeddings_list.append(image_features.cpu().numpy())
                image_paths_list.extend(batch['image_paths'])
                image_names_list.extend(batch['image_names'])
        
        # Concatenate all embeddings
        self.embeddings = np.vstack(embeddings_list)
        self.image_paths = image_paths_list
        self.image_names = image_names_list
        
        if verbose:
            print(f"Extracted {self.embeddings.shape[0]} image embeddings with dimension {self.embeddings.shape[1]}")
    
    def _build_faiss_index(self, verbose=True):
        """Build FAISS index from image embeddings"""
        if verbose:
            print("Building FAISS index...")
        
        # Create FAISS index (using L2 distance, which is equivalent to cosine similarity for normalized vectors)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for normalized vectors
        
        # Add embeddings to index
        self.index.add(self.embeddings.astype(np.float32))
        
        if verbose:
            print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def save_index(self, verbose=True):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
            
            # Save metadata
            metadata = {
                'image_paths': self.image_paths,
                'image_names': self.image_names,
                'embeddings_shape': self.embeddings.shape,
                'split': self.split
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            if verbose:
                print(f"Index saved to {self.index_path}")
                print(f"Metadata saved to {self.metadata_path}")
            
        except Exception as e:
            if verbose:
                print(f"Error saving index: {e}")
    
    def load_index(self, verbose=True):
        """Load FAISS index and metadata from disk"""
        try:
            if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.image_paths = metadata['image_paths']
            self.image_names = metadata['image_names']
            
            if verbose:
                print(f"Loaded index with {self.index.ntotal} vectors")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"Error loading index: {e}")
            return False
    
    def build_or_load_index(self, force_rebuild=False, verbose=True):
        """Build FAISS index or load from cache"""
        if not force_rebuild and self.load_index(verbose):
            if verbose:
                print("Using cached FAISS index")
            return
        
        if verbose:
            print("Building new FAISS index...")
        
        # Extract embeddings and build index
        self._extract_image_embeddings(verbose)
        self._build_faiss_index(verbose)
        self.save_index(verbose)
    
    def retrieve(self, query_text, k=10, caption_type='short'):
        """
        Retrieve top-k most similar images for a given text query
        
        Args:
            query_text: Text query string
            k: Number of results to return
            caption_type: Not used in retrieval, kept for compatibility
            
        Returns:
            list: List of dictionaries with image info and similarity scores
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_or_load_index() first.")
        
        # Encode query text
        self.model.eval()
        with torch.no_grad():
            # Tokenize text (assuming mobileclip tokenizer)
            from functions.model import load_model
            _, _, tokenizer = load_model(self.config, verbose=False)
            
            text_tokens = tokenizer([query_text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Search in FAISS index
        query_embedding = text_features.cpu().numpy().astype(np.float32)
        similarities, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            results.append({
                'rank': i + 1,
                'image_name': self.image_names[idx],
                'image_path': self.image_paths[idx],
                'similarity': float(similarity),
                'index': int(idx)
            })
        
        return results

def retrieve(query, faiss_index_path, k):
    """
    Legacy function for backward compatibility
    Note: This is a simplified version. Use FAISSRetriever class for full functionality.
    """
    # This function would need additional parameters to work properly
    # Recommend using FAISSRetriever class instead
    raise NotImplementedError("Use FAISSRetriever class for retrieval functionality")