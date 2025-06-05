"""
LightVision Retrieval Framework

This module implements the image retrieval system using the fine-tuned MobileCLIP model
with FAISS for efficient similarity search.
"""

import torch
import torch.nn.functional as F
import numpy as np
import faiss
import pickle
import json
import os
from PIL import Image
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import mobileclip
from tqdm import tqdm
import time

@dataclass
class RetrievalConfig:
    """Configuration class for the retrieval system"""
    model_name: str = 'mobileclip_s0'
    checkpoint_path: Optional[str] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_dim: int = 512
    faiss_index_type: str = 'IndexFlatIP'  # Inner Product for cosine similarity
    batch_size: int = 32
    top_k: int = 10
    
class ImageEmbeddingDatabase:
    """Manages image embeddings and metadata"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.embeddings = None
        self.image_paths = []
        self.image_metadata = {}
        self.faiss_index = None
        
    def add_embeddings(self, embeddings: np.ndarray, image_paths: List[str], 
                      metadata: Optional[Dict] = None):
        """Add embeddings to the database"""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            
        self.image_paths.extend(image_paths)
        
        if metadata:
            self.image_metadata.update(metadata)
    
    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        if self.embeddings is None:
            raise ValueError("No embeddings available to build index")
            
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )
        
        # Create FAISS index
        if self.config.faiss_index_type == 'IndexFlatIP':
            self.faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
        elif self.config.faiss_index_type == 'IndexIVFFlat':
            # For larger datasets, use IVF index
            quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(
                quantizer, self.config.embedding_dim, min(100, len(normalized_embeddings) // 10)
            )
            self.faiss_index.train(normalized_embeddings.astype(np.float32))
        else:
            raise ValueError(f"Unsupported index type: {self.config.faiss_index_type}")
            
        # Add embeddings to index
        self.faiss_index.add(normalized_embeddings.astype(np.float32))
        print(f"Built FAISS index with {self.faiss_index.ntotal} embeddings")
    
    def search(self, query_embedding: np.ndarray, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar images"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first")
            
        k = k or self.config.top_k
        
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding, keepdims=True)
        query_normalized = query_normalized.astype(np.float32).reshape(1, -1)
        
        # Search
        similarities, indices = self.faiss_index.search(query_normalized, k)
        
        return similarities[0], indices[0]
    
    def save(self, filepath: str):
        """Save the database to disk"""
        data = {
            'embeddings': self.embeddings,
            'image_paths': self.image_paths,
            'image_metadata': self.image_metadata,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        # Save FAISS index separately
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, filepath.replace('.pkl', '_faiss.index'))
    
    def load(self, filepath: str):
        """Load the database from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.embeddings = data['embeddings']
        self.image_paths = data['image_paths']
        self.image_metadata = data['image_metadata']
        self.config = data['config']
        
        # Load FAISS index
        faiss_path = filepath.replace('.pkl', '_faiss.index')
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)

class LightVisionRetrieval:
    """Main retrieval system class"""
    
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self.database = ImageEmbeddingDatabase(config)
        
    def load_model(self):
        """Load the MobileCLIP model"""
        print(f"Loading model: {self.config.model_name}")
        
        # Load base model
        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            # Load fine-tuned model
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.checkpoint_path
            )
            print(f"Loaded fine-tuned model from {self.config.checkpoint_path}")
        else:
            # Load base model
            self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
                self.config.model_name,
                pretrained=True
            )
            print("Loaded base MobileCLIP model")
            
        self.model.to(self.config.device)
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = mobileclip.get_tokenizer(self.config.model_name)
        
    def encode_images(self, image_paths: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode a batch of images to embeddings"""
        embeddings = []
        
        iterator = tqdm(image_paths, desc="Encoding images") if show_progress else image_paths
        
        for i in range(0, len(image_paths), self.config.batch_size):
            batch_paths = image_paths[i:i + self.config.batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = self.preprocess(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.config.device)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        batch_embeddings = self.model.encode_image(batch_tensor)
                        batch_embeddings = F.normalize(batch_embeddings, dim=-1)
                        
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        tokens = self.tokenizer([text]).to(self.config.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                text_embedding = self.model.encode_text(tokens)
                text_embedding = F.normalize(text_embedding, dim=-1)
                
        return text_embedding.cpu().numpy()
    
    def build_database(self, image_directory: str, captions_file: Optional[str] = None):
        """Build the image embedding database"""
        print(f"Building database from {image_directory}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for root, dirs, files in os.walk(image_directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images")
        
        # Load captions if available
        captions_data = {}
        if captions_file and os.path.exists(captions_file):
            if captions_file.endswith('.json'):
                with open(captions_file, 'r') as f:
                    captions_data = json.load(f)
            elif captions_file.endswith('.txt'):
                with open(captions_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            caption = parts[1]
                            captions_data[img_name] = caption
        
        # Encode images
        embeddings = self.encode_images(image_paths)
        
        # Prepare metadata
        metadata = {}
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            metadata[i] = {
                'path': img_path,
                'filename': img_name,
                'caption': captions_data.get(img_name, '')
            }
        
        # Add to database
        self.database.add_embeddings(embeddings, image_paths, metadata)
        self.database.build_faiss_index()
        
        print(f"Database built with {len(image_paths)} images")
    
    def search(self, query: str, k: int = None) -> List[Dict]:
        """Search for images similar to the text query"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")
            
        k = k or self.config.top_k
        
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Search
        similarities, indices = self.database.search(query_embedding, k)
        
        # Prepare results
        results = []
        for sim, idx in zip(similarities, indices):
            if idx < len(self.database.image_paths):
                result = {
                    'similarity': float(sim),
                    'image_path': self.database.image_paths[idx],
                    'metadata': self.database.image_metadata.get(idx, {})
                }
                results.append(result)
        
        return results
    
    def save_database(self, filepath: str):
        """Save the database to disk"""
        self.database.save(filepath)
        print(f"Database saved to {filepath}")
    
    def load_database(self, filepath: str):
        """Load the database from disk"""
        self.database.load(filepath)
        print(f"Database loaded from {filepath}")

def main():
    """Example usage of the retrieval system"""
    
    # Configuration
    config = RetrievalConfig(
        model_name='mobileclip_s0',
        checkpoint_path='checkpoints/mobileclip_finetuned_epoch1_last.pt',  # Set to None for base model
        device='cuda' if torch.cuda.is_available() else 'cpu',
        top_k=5
    )
    
    # Initialize retrieval system
    retrieval_system = LightVisionRetrieval(config)
    
    # Load model
    retrieval_system.load_model()
    
    # Build database from images
    # Replace with your actual image directory and captions file
    image_directory = "data/Images"
    captions_file = "data/captions.txt"  # Optional
    
    # Check if database exists
    db_path = "image_database.pkl"
    if os.path.exists(db_path):
        print("Loading existing database...")
        retrieval_system.load_database(db_path)
    else:
        print("Building new database...")
        retrieval_system.build_database(image_directory, captions_file)
        retrieval_system.save_database(db_path)
    
    # Interactive query loop
    print("\n=== LightVision Image Retrieval System ===")
    print("Enter text queries to search for images. Type 'quit' to exit.")
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        try:
            start_time = time.time()
            results = retrieval_system.search(query, k=5)
            search_time = time.time() - start_time
            
            print(f"\nSearch completed in {search_time:.3f} seconds")
            print(f"Top {len(results)} results for: '{query}'")
            print("-" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"{i}. Similarity: {result['similarity']:.4f}")
                print(f"   Image: {result['image_path']}")
                if result['metadata'].get('caption'):
                    print(f"   Caption: {result['metadata']['caption']}")
                print()
                
        except Exception as e:
            print(f"Error during search: {e}")

if __name__ == "__main__":
    main()