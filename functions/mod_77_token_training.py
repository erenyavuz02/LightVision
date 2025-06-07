import torch
import torch.nn.functional as F
import math

def calculate_subsection_weights(num_subsections: int) -> list:
    """
    Calculate weights for subsections based on their importance.
    First subsection gets highest weight, subsequent ones follow squared importance decay.
    
    Args:
        num_subsections: Number of subsections
    
    Returns:
        List of weights for each subsection (normalized to sum to 1.0)
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
            # Following pattern: 9, 4, 1, 0.25, 0.0625, etc.
            weight = 1.0 / (4 ** (i - 2))
    
        weights.append(weight)
    
    # Normalize weights so they sum to 1.0
    # This ensures each sample contributes equally to the loss regardless of subsection count
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    return normalized_weights


def weighted_contrastive_loss_subsections(image_features, text_subsections_batch, temperature=0.07):
    """
    OPTIMIZED: Calculate contrastive loss using weighted similarity scores between sub-captions and images.
    For each sample, compute dot products between image and each sub-caption, then weight and sum them.
    
    Args:
        image_features: Image embeddings [batch_size, embed_dim]
        text_subsections_batch: List of lists of text embeddings for each sample
        temperature: Temperature parameter for contrastive loss
    
    Returns:
        Contrastive loss using weighted similarity scores
    """
    batch_size = image_features.shape[0]
    device = image_features.device
    
    # Normalize image features once
    image_features = F.normalize(image_features, dim=-1)
    
    # Pre-normalize all text subsections and prepare for batch processing
    normalized_subsections_batch = []
    weights_batch = []
    
    for text_idx in range(batch_size):
        text_subsections = text_subsections_batch[text_idx]
        if len(text_subsections) == 0:
            # Handle empty subsections case
            normalized_subsections_batch.append([])
            weights_batch.append([])
            continue
            
        # Normalize all subsections for this sample
        normalized_subsections = [F.normalize(subsection, dim=-1) for subsection in text_subsections]
        normalized_subsections_batch.append(normalized_subsections)
        
        # Get weights for this sample
        weights = calculate_subsection_weights(len(text_subsections))
        weights_batch.append(torch.tensor(weights, device=device, dtype=torch.float32))
    
    # Create similarity matrices more efficiently
    sim_i2t = torch.zeros(batch_size, batch_size, device=device)
    sim_t2i = torch.zeros(batch_size, batch_size, device=device)
    
    # Compute similarities using matrix operations where possible
    for text_idx in range(batch_size):
        if len(normalized_subsections_batch[text_idx]) == 0:
            continue
            
        # Stack subsections for this text sample
        text_subsections = torch.stack(normalized_subsections_batch[text_idx])  # [num_subsections, embed_dim]
        weights = weights_batch[text_idx]  # [num_subsections]
        
        # Compute similarities between all images and all subsections of this text
        # image_features: [batch_size, embed_dim]
        # text_subsections: [num_subsections, embed_dim]
        similarities = torch.matmul(image_features, text_subsections.T)  # [batch_size, num_subsections]
        
        # Apply weights and sum
        weighted_similarities = torch.matmul(similarities, weights)  # [batch_size]
        
        # Store in similarity matrices
        sim_i2t[:, text_idx] = weighted_similarities
        sim_t2i[text_idx, :] = weighted_similarities
    
    # Apply temperature scaling
    sim_i2t = sim_i2t / temperature
    sim_t2i = sim_t2i / temperature
    
    # Create labels
    labels = torch.arange(batch_size, device=device)
    
    # Calculate contrastive loss
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)
    
    # Return average of both directions
    total_loss = (loss_i2t + loss_t2i) / 2
    
    return total_loss

def mod_77_contrastive_loss(model, image_features, text_subsections_batch, temperature=0.07):
    """
    Modified contrastive loss that handles subsections with weighted importance.
    
    Args:
        model: The CLIP model
        image_features: Image embeddings [batch_size, embed_dim]
        text_subsections_batch: List of lists containing text subsections for each sample
        temperature: Temperature parameter
    
    Returns:
        Weighted contrastive loss
    """
    return weighted_contrastive_loss_subsections(image_features, text_subsections_batch, temperature)

def mod_77_long_clip_loss(model, image_embedding, long_subsections_batch, short_embedding, temperature=0.07):
    """
    Modified LongCLIP loss that handles subsections with weighted importance.
    
    Args:
        model: The CLIP model
        image_embedding: Image embeddings [batch_size, embed_dim]
        long_subsections_batch: List of lists containing long text subsections for each sample
        short_embedding: Short text embeddings [batch_size, embed_dim]
        temperature: Temperature parameter
    
    Returns:
        Combined weighted loss
    """
    # Normalize features
    image_features_long = F.normalize(image_embedding, dim=1)
    text_features_short = F.normalize(short_embedding, dim=1)
    
    # Simple PCA approximation using SVD for dimensionality reduction
    # This replaces the problematic PCA import
    U, S, V = torch.svd(image_features_long)
    image_features_short = torch.matmul(image_features_long, V[:, :32])  # Reduce to 32 dimensions
    image_features_short = F.normalize(image_features_short, dim=1)
    
    # Calculate weighted loss for long subsections
    loss_long = mod_77_contrastive_loss(model, image_features_long, long_subsections_batch, temperature)
    
    # Calculate standard contrastive loss for short text
    sim_i2ts = torch.matmul(image_features_short, text_features_short.T) / temperature
    sim_ts2i = torch.matmul(text_features_short, image_features_short.T) / temperature
    
    # Apply logit scale if available
    if hasattr(model, 'logit_scale'):
        logit_scale = model.logit_scale
        if isinstance(logit_scale, torch.nn.Parameter):
            sim_i2ts = logit_scale.exp() * sim_i2ts
            sim_ts2i = logit_scale.exp() * sim_ts2i
    
    # Create targets
    bs = image_embedding.size(0)
    targets = torch.arange(bs, device=image_embedding.device)
    
    # Calculate short text loss
    loss_short = (
        F.cross_entropy(sim_i2ts, targets, label_smoothing=0.1)
        + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.1)
    ) / 2
    
    # Combine losses
    total_loss = (loss_long + loss_short) / 2
    
    return total_loss


def validate_subsection_weights():
    """
    Test function to validate the weight calculation logic.
    """
    print("Testing subsection weight calculation:")
    for i in range(1, 8):
        weights = calculate_subsection_weights(i)
        print(f"Subsections: {i}, Weights: {[f'{w:.4f}' for w in weights]}, Sum: {sum(weights):.4f}")

def test_loss_function():
    """
    Test the optimized loss function implementation.
    """
    print("\n" + "="*50)
    print("TESTING OPTIMIZED LOSS FUNCTION")
    print("="*50)
    
    # Create test data
    batch_size = 4
    embed_dim = 512
    device = torch.device('cpu')
    
    # Mock image features (with gradients enabled)
    image_features = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
    
    # Mock text subsections batch (different numbers of subsections per sample)
    text_subsections_batch = []
    for i in range(batch_size):
        num_subsections = (i % 3) + 1  # 1, 2, 3, 1 subsections
        subsections = [torch.randn(embed_dim, device=device, requires_grad=True) for _ in range(num_subsections)]
        text_subsections_batch.append(subsections)
    
    print(f"Test setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Subsections per sample: {[len(subs) for subs in text_subsections_batch]}")
    
    # Test the loss function
    try:
        loss = weighted_contrastive_loss_subsections(image_features, text_subsections_batch)
        print(f"\n✓ Loss function executed successfully!")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Loss requires grad: {loss.requires_grad}")
        
        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful!")
        
        # Check gradients
        print(f"✓ Image features gradients: {image_features.grad is not None}")
        for i, subsections in enumerate(text_subsections_batch):
            grad_status = [sub.grad is not None for sub in subsections]
            print(f"✓ Text subsections {i+1} gradients: {grad_status}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error in loss function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the weight calculation
    validate_subsection_weights()
    
    print("\nWeight distribution explanation:")
    print("- 1st subsection: Most important (weight = 9)")
    print("- 2nd subsection: Important (weight = 4)")  
    print("- 3rd subsection: Less important (weight = 1)")
    print("- 4th+ subsections: Exponentially decreasing importance")
    print("- All weights are normalized to maintain loss scale")
    
    # Test the optimized loss function
    test_loss_function()


def encode_text_subsections_batch(model, tokenizer, long_splitted_captions, device):
    """
    Encode multiple text subsections for a batch of samples using batch processing.
    This function matches the approach used in train.py.
    
    Args:
        model: The CLIP model
        tokenizer: Text tokenizer
        long_splitted_captions: List of lists, where each inner list contains text subsections for one sample
        device: Device to run on
    
    Returns:
        List of tensors, each containing encoded features for subsections of one sample
    """
    # Flatten all subsections from all samples in the batch
    all_subsections = []
    subsection_counts = []  # Track how many subsections each sample has
    
    for sample_subsections in long_splitted_captions:
        subsection_counts.append(len(sample_subsections))
        all_subsections.extend(sample_subsections)  # Flatten to single list
    
    if not all_subsections:
        return []
    
    # Tokenize all subsections at once
    tokenized_subsections = tokenizer(all_subsections).to(device)
    
    # Encode all subsections
    text_subsection_features = model.encode_text(tokenized_subsections)
    
    # Reshape back to per-sample structure for loss calculation
    # Split the features back according to subsection_counts
    subsection_features_per_sample = []
    start_idx = 0
    for count in subsection_counts:
        end_idx = start_idx + count
        sample_features = text_subsection_features[start_idx:end_idx]
        subsection_features_per_sample.append(sample_features)
        start_idx = end_idx
    
    return subsection_features_per_sample

def prepare_subsections_batch_optimized(long_splitted_captions, tokenizer, device):
    """
    Prepare a batch of subsections for training using optimized batch processing.
    This function is now redundant since tokenization is handled in encode_text_subsections_batch,
    but kept for compatibility.
    
    Args:
        long_splitted_captions: List of subsection lists for each sample in the batch
        tokenizer: Text tokenizer
        device: Device to run on
    
    Returns:
        Tuple of (all_subsections_flat, subsection_counts)
    """
    # Flatten all subsections from all samples
    all_subsections = []
    subsection_counts = []
    
    for sample_subsections in long_splitted_captions:
        subsection_counts.append(len(sample_subsections))
        all_subsections.extend(sample_subsections)
    
    return all_subsections, subsection_counts

def process_batch_subsections(model, tokenizer, long_splitted_captions, device):
    """
    Complete pipeline to process batch subsections - tokenize, encode, and restructure.
    This is the main function that replicates the logic from train.py.
    
    Args:
        model: The CLIP model
        tokenizer: Text tokenizer
        long_splitted_captions: List of lists containing text subsections for each sample
        device: Device to run on
    
    Returns:
        List of tensors containing encoded features for each sample's subsections
    """
    # Step 1: Flatten all subsections
    all_subsections = []
    subsection_counts = []
    
    for sample_subsections in long_splitted_captions:
        subsection_counts.append(len(sample_subsections))
        all_subsections.extend(sample_subsections)
    
    if not all_subsections:
        return []
    
    # Step 2: Tokenize all subsections at once
    tokenized_subsections = tokenizer(all_subsections).to(device)
    
    # Step 3: Encode all subsections
    text_subsection_features = model.encode_text(tokenized_subsections)
    
    # Step 4: Reshape back to per-sample structure
    subsection_features_per_sample = []
    start_idx = 0
    for count in subsection_counts:
        end_idx = start_idx + count
        sample_features = text_subsection_features[start_idx:end_idx]
        subsection_features_per_sample.append(sample_features)
        start_idx = end_idx
    
    return subsection_features_per_sample