import os
import pickle
import faiss
import numpy as np
import torch
from pathlib import Path

class FAISSRetriever:
    """FAISS-based image retrieval system for evaluating image-text models"""
    
    def __init__(self, config, model, dataset, tokenizer, split='test'):
        """
        Initialize the FAISS retriever
        
        Args:
            config: Configuration object
            model: The loaded model (e.g., MobileCLIP)
            dataset: CustomDataset instance
            tokenizer: Text tokenizer for the model
            split: Which split to use ('train' or 'test')
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
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
            # Use the tokenizer passed during initialization
            text_tokens = self.tokenizer([query_text]).to(self.device)
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

    def retrieve_with_subsections(self, caption_list, k=10, initial_k=20):
        """
        Retrieve images using multi-subsection approach with pre-divided captions.
        
        Args:
            caption_list: List of caption strings (already divided subsections)
            k: Final number of results to return (e.g., top 10 final results)
            initial_k: Number of candidates to retrieve PER EACH caption/subsection 
                      (e.g., top 20 images for caption 1, top 20 for caption 2, etc.)
                      These candidates are then scored and combined
            
        Returns:
            list: List of dictionaries with image info and cumulative scores
            
        Example:
            If initial_k=20 and you have 3 captions:
            - Caption 1 retrieves top 20 candidate images (scored with weight 9)
            - Caption 2 retrieves top 20 candidate images (scored with weight 4) 
            - Caption 3 retrieves top 20 candidate images (scored with weight 1)
            - All candidates are combined, scores summed, and top k=10 final results returned
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_or_load_index() first.")
        
        if not isinstance(caption_list, list):
            raise ValueError("caption_list must be a list of caption strings")
        
        if len(caption_list) == 0:
            raise ValueError("caption_list cannot be empty")
        
        # If only one caption, use standard retrieval
        if len(caption_list) == 1:
            return self.retrieve(caption_list[0], k)
        
        print(f"Processing {len(caption_list)} caption subsections")
        for i, caption in enumerate(caption_list, 1):
            print(f"  Caption {i}: {caption[:100]}...")
        
        # Calculate weights for subsections (same logic as training)
        weights = self._calculate_subsection_weights(len(caption_list))
        
        # Retrieve candidates for each subsection
        candidate_scores = {}  # image_index -> cumulative_score
        candidate_info = {}    # image_index -> image_info
        
        for caption_idx, (caption, weight) in enumerate(zip(caption_list, weights)):
            print(f"Processing caption {caption_idx + 1} with weight {weight:.3f}")
            
            # Retrieve top candidates for this caption
            caption_results = self.retrieve(caption, initial_k)
            
            # Add weighted scores to candidates
            for result in caption_results:
                idx = result['index']
                weighted_score = result['similarity'] * weight
                
                if idx in candidate_scores:
                    candidate_scores[idx] += weighted_score
                else:
                    candidate_scores[idx] = weighted_score
                    candidate_info[idx] = {
                        'image_name': result['image_name'],
                        'image_path': result['image_path'],
                        'index': idx
                    }
        
        # Sort candidates by cumulative score
        sorted_candidates = sorted(
            candidate_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Prepare final results
        final_results = []
        for rank, (idx, cumulative_score) in enumerate(sorted_candidates[:k], 1):
            result = candidate_info[idx].copy()
            result.update({
                'rank': rank,
                'cumulative_score': float(cumulative_score),
                'similarity': float(cumulative_score)  # For compatibility
            })
            final_results.append(result)
        
        return final_results
    
    def _calculate_subsection_weights(self, num_subsections: int) -> list:
        """
        Calculate weights for subsections based on their importance.
        Same logic as in mod_77_token_training.py
        
        Args:
            num_subsections: Number of subsections
        
        Returns:
            List of weights for each subsection
        """
        if num_subsections == 1:
            return [1.0]
        
        weights = []
        for i in range(num_subsections):
            if i == 0:
                # First subsection gets weight of 9
                weight = 9.0
            elif i == 1:
                # Second subsection gets weight of 4
                weight = 4.0
            elif i == 2:
                # Third subsection gets weight of 1
                weight = 1.0
            else:
                # For subsections beyond 3, continue with decreasing squared importance
                weight = 1.0 / (4 ** (i - 2))
        
            weights.append(weight)
        
        # Normalize weights so they sum to the number of subsections
        weight_sum = sum(weights)
        normalized_weights = [w * num_subsections / weight_sum for w in weights]
        
        return normalized_weights
    
    def evaluate_retrieval_with_subsections(self, queries_list, ground_truth, k_values=[1, 5, 10], initial_k=20):
        """
        Evaluate retrieval performance using multi-subsection approach.
        
        Args:
            queries_list: List of query caption lists (each query is a list of captions)
            ground_truth: Dictionary mapping query_id to correct image names
            k_values: List of k values to evaluate
            initial_k: Number of candidates per subsection
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {f'recall@{k}': [] for k in k_values}
        results.update({f'precision@{k}': [] for k in k_values})
        
        print(f"Evaluating {len(queries_list)} queries with subsection-based retrieval...")
        
        for query_idx, caption_list in enumerate(queries_list):
            if query_idx % 10 == 0:
                print(f"Processing query {query_idx + 1}/{len(queries_list)}")
            
            # Get ground truth for this query
            gt_images = ground_truth.get(query_idx, set())
            if isinstance(gt_images, str):
                gt_images = {gt_images}
            elif isinstance(gt_images, list):
                gt_images = set(gt_images)
            
            # Retrieve with subsections
            retrieved = self.retrieve_with_subsections(
                caption_list, 
                k=max(k_values), 
                initial_k=initial_k
            )
            
            # Calculate metrics for each k
            for k in k_values:
                retrieved_k = retrieved[:k]
                retrieved_names = {result['image_name'] for result in retrieved_k}
                
                # Calculate recall and precision
                hits = len(gt_images.intersection(retrieved_names))
                recall = hits / len(gt_images) if gt_images else 0
                precision = hits / k if k > 0 else 0
                
                results[f'recall@{k}'].append(recall)
                results[f'precision@{k}'].append(precision)
        
        # Calculate average metrics
        avg_results = {}
        for metric, values in results.items():
            avg_results[metric] = sum(values) / len(values) if values else 0
        
        return avg_results
    
    def compare_retrieval_methods(self, query_input, k=10, initial_k=20):
        """
        Compare standard retrieval vs subsection-based retrieval.
        
        Args:
            query_input: Either a string (for standard) or list of strings (for subsections)
            k: Number of results to return
            initial_k: Number of candidates per subsection for multi-subsection method
            
        Returns:
            Dictionary with results from both methods
        """
        if isinstance(query_input, str):
            # Standard retrieval
            standard_results = self.retrieve(query_input, k)
            subsection_results = None
            query_text = query_input
        elif isinstance(query_input, list):
            # Use first caption for standard retrieval comparison
            query_text = " ".join(query_input)  # Combine for standard retrieval
            standard_results = self.retrieve(query_text, k)
            subsection_results = self.retrieve_with_subsections(query_input, k, initial_k)
        else:
            raise ValueError("query_input must be either a string or list of strings")
        
        comparison = {
            'query': query_input,
            'query_type': 'string' if isinstance(query_input, str) else 'list',
            'standard_retrieval': standard_results,
            'subsection_retrieval': subsection_results,
            'methods_differ': standard_results != subsection_results if subsection_results else False
        }
        
        return comparison
    
    def batch_retrieve_with_subsections(self, queries_list, k=10, initial_k=20, verbose=True):
        """
        Perform subsection-based retrieval for multiple queries.
        
        Args:
            queries_list: List of query caption lists (each query is a list of captions)
            k: Number of results per query
            initial_k: Number of candidates per subsection
            verbose: Whether to print progress
            
        Returns:
            List of retrieval results for each query
        """
        results = []
        
        for i, caption_list in enumerate(queries_list):
            if verbose and i % 10 == 0:
                print(f"Processing query {i + 1}/{len(queries_list)}")
            
            query_results = self.retrieve_with_subsections(caption_list, k, initial_k)
            results.append({
                'query_id': i,
                'query_captions': caption_list,
                'results': query_results
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