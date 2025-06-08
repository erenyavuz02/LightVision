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
        weight = 1 / num_subsections  # Base weight for all subsections
        weights.append(weight)
    
    return weights


def weighted_contrastive_loss_subsections(image_features, text_subsections_batch, temperature=0.07):
    """
    Calculate contrastive loss using weighted similarity scores between sub-captions and images.
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
    
    sim_i2t = torch.zeros(batch_size, batch_size, device=device)
    sim_t2i = torch.zeros(batch_size, batch_size, device=device)
    
    for text_idx in range(batch_size):
        text_subsections = text_subsections_batch[text_idx]
        num_subsections = len(text_subsections)
        
        if num_subsections == 0:
            continue
        
        weights = calculate_subsection_weights(num_subsections)
        
        for img_idx in range(batch_size):
            current_image = image_features[img_idx]
            weighted_similarity = 0.0
            
            for subsection_idx, subsection_features in enumerate(text_subsections):
                similarity = torch.dot(current_image, subsection_features)
                weight = weights[subsection_idx]
                weighted_similarity += weight * similarity
            
            sim_i2t[img_idx, text_idx] = weighted_similarity
            sim_t2i[text_idx, img_idx] = weighted_similarity
    
    # Apply temperature scaling
    sim_i2t = sim_i2t / temperature
    sim_t2i = sim_t2i / temperature

    # Create labels and calculate loss
    labels = torch.arange(batch_size, device=device)
    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)

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
    return weighted_contrastive_loss_subsections_vectorized(image_features, text_subsections_batch, temperature)

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
    image_features_long = image_embedding
    text_features_short = short_embedding 
    
    # Apply PCA to get compressed image features (assuming PCA function exists)
    from .train import PCA
    image_features_short = PCA(image_features_long, 32)
    image_features_short = image_features_short / image_features_short.norm(dim=1, keepdim=True)
    
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
        F.cross_entropy(sim_i2ts, targets)
        + F.cross_entropy(sim_ts2i, targets)
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

if __name__ == "__main__":
    # Test the weight calculation
    validate_subsection_weights()
    
    print("\nWeight distribution explanation:")
    print("- 1st subsection: Most important (weight = 9)")
    print("- 2nd subsection: Important (weight = 4)")  
    print("- 3rd subsection: Less important (weight = 1)")
    print("- 4th+ subsections: Exponentially decreasing importance")
    print("- All weights are normalized to maintain loss scale")


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


def weighted_contrastive_loss_subsections_vectorized(image_features, text_subsections_batch, temperature=0.07):
    """
    Vectorized version: Calculate contrastive loss using weighted similarity scores between sub-captions and images.
    Creates vectors for each subsection position across all samples, handles missing subsections with zeros.
    
    Args:
        image_features: Image embeddings [batch_size, embed_dim]
        text_subsections_batch: List of lists of text embeddings for each sample
        temperature: Temperature parameter for contrastive loss
    
    Returns:
        Contrastive loss using weighted similarity scores
    """
    batch_size = image_features.shape[0]
    device = image_features.device
    embed_dim = image_features.shape[1]
    
    # Find maximum number of subsections across all samples
    max_subsections = max(len(subsections) for subsections in text_subsections_batch)
    
    if max_subsections == 0:
        # Fallback: return zero loss if no subsections
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # get the weight matrix for each caption
    caption_weights = [calculate_subsection_weights(len(subsections)) for subsections in text_subsections_batch]
    
    # Convert to matrix with zero padding
    caption_weights_matrix = torch.zeros(batch_size, max_subsections, device=device)
    for i, weights in enumerate(caption_weights):
        caption_weights_matrix[i, :len(weights)] = torch.tensor(weights, device=device)


    # Initialize accumulated similarity matrices
    sim_i2t_total = torch.zeros(batch_size, batch_size, device=device)
    sim_t2i_total = torch.zeros(batch_size, batch_size, device=device)
    
    # Process each subsection position
    for subsection_idx in range(max_subsections):
        # Create vector for this subsection position
        subsection_vector = torch.zeros(batch_size, embed_dim, device=device)
        mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        
        # Fill the vector with subsections at this position
        for sample_idx, subsections in enumerate(text_subsections_batch):
            if subsection_idx < len(subsections):
                subsection_vector[sample_idx] = subsections[subsection_idx]
                mask[sample_idx] = True
            # else: remains zero (missing subsection)
        
        # Skip if no samples have subsection at this position
        if not mask.any():
            continue
        
        # Calculate similarity matrices for this subsection position
        # sim_i2t: image_features @ subsection_vector.T
        sim_i2t_sub = torch.matmul(image_features, subsection_vector.T)  # [batch_size, batch_size]
        sim_t2i_sub = sim_i2t_sub.T  # [batch_size, batch_size]
        
        # Apply mask to zero out missing subsections
        # For each sample that doesn't have this subsection, zero out its column/row
        mask_matrix = mask.unsqueeze(0).expand(batch_size, -1)  # [batch_size, batch_size]
        sim_i2t_sub = sim_i2t_sub * mask_matrix  # Zero out columns for missing subsections
        
        mask_matrix_t = mask.unsqueeze(1).expand(-1, batch_size)  # [batch_size, batch_size]
        sim_t2i_sub = sim_t2i_sub * mask_matrix_t  # Zero out rows for missing subsections
        
        # Calculate weight for this subsection position
        # Get average number of subsections to calculate relative importance
        avg_num_subsections = sum(len(subs) for subs in text_subsections_batch) / batch_size
        
        
        # get the weight for this subsection position
        weight = caption_weights_matrix[:, subsection_idx]  # [batch_size]
        
        # Apply weight and accumulate
        sim_i2t_total += weight * sim_i2t_sub

        sim_t2i_total += weight.T * sim_t2i_sub

    # Apply temperature scaling
    sim_i2t_total = sim_i2t_total / temperature
    sim_t2i_total = sim_t2i_total / temperature
    
    # Create labels and calculate loss
    labels = torch.arange(batch_size, device=device)
    loss_i2t = F.cross_entropy(sim_i2t_total, labels)
    loss_t2i = F.cross_entropy(sim_t2i_total, labels)
    
    total_loss = (loss_i2t + loss_t2i) / 2
    
    return total_loss



def process_batch_subsections_vectorized(model, tokenizer, long_splitted_captions, device):
    """
    Vectorized version: Process batch subsections by creating position-based vectors.
    
    Args:
        model: The CLIP model
        tokenizer: Text tokenizer
        long_splitted_captions: List of lists containing text subsections for each sample
        device: Device to run on
    
    Returns:
        List of tensors containing encoded features for each sample's subsections
    """
    if not long_splitted_captions:
        return []
    
    batch_size = len(long_splitted_captions)
    max_subsections = max(len(subsections) for subsections in long_splitted_captions)
    
    if max_subsections == 0:
        return []
    
    # Create position-based vectors and encode them
    subsection_features_per_sample = [[] for _ in range(batch_size)]
    
    for position in range(max_subsections):
        # Collect subsections at this position
        position_texts = []
        position_indices = []  # Track which samples have subsection at this position
        
        for sample_idx, subsections in enumerate(long_splitted_captions):
            if position < len(subsections):
                position_texts.append(subsections[position])
                position_indices.append(sample_idx)
        
        if position_texts:
            # Tokenize and encode subsections at this position
            tokenized = tokenizer(position_texts).to(device)
            encoded_features = model.encode_text(tokenized)
            
            # Distribute back to per-sample structure
            for i, sample_idx in enumerate(position_indices):
                subsection_features_per_sample[sample_idx].append(encoded_features[i])
    
    return subsection_features_per_sample
