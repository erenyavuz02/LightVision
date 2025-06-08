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


import torch
import torch.nn.functional as F

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
    
    print("\n[DEBUG] Original image features:\n", image_features)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    print("\n[DEBUG] Normalized image features:\n", image_features)
    
    sim_i2t = torch.zeros(batch_size, batch_size, device=device)
    sim_t2i = torch.zeros(batch_size, batch_size, device=device)
    
    for text_idx in range(batch_size):
        text_subsections = text_subsections_batch[text_idx]
        num_subsections = len(text_subsections)
        
        print(f"\n[DEBUG] Processing sample {text_idx}, num_subsections: {num_subsections}")
        
        if num_subsections == 0:
            continue
        
        weights = calculate_subsection_weights(num_subsections)
        print(f"[DEBUG] Subsection weights: {weights}")
        
        for img_idx in range(batch_size):
            current_image = image_features[img_idx]
            weighted_similarity = 0.0
            
            for subsection_idx, subsection_features in enumerate(text_subsections):
                subsection_features = subsection_features / subsection_features.norm(dim=-1, keepdim=True)
                similarity = torch.dot(current_image, subsection_features)
                weight = weights[subsection_idx]
                weighted_similarity += weight * similarity

                print(f"[DEBUG] img_idx={img_idx}, text_idx={text_idx}, subsection_idx={subsection_idx}")
                print(f"        similarity={similarity.item()}, weight={weight}, weighted_contribution={weight * similarity}")
            
            sim_i2t[img_idx, text_idx] = weighted_similarity
            sim_t2i[text_idx, img_idx] = weighted_similarity
            print(f"[DEBUG] sim_i2t[{img_idx}, {text_idx}] = {weighted_similarity.item()}")
    
    print("\n[DEBUG] Raw sim_i2t:\n", sim_i2t)
    print("\n[DEBUG] Raw sim_t2i:\n", sim_t2i)

    sim_i2t = sim_i2t / temperature
    sim_t2i = sim_t2i / temperature

    print("\n[DEBUG] Scaled sim_i2t:\n", sim_i2t)
    print("\n[DEBUG] Scaled sim_t2i:\n", sim_t2i)

    labels = torch.arange(batch_size, device=device)
    print("\n[DEBUG] Labels:\n", labels)

    loss_i2t = F.cross_entropy(sim_i2t, labels)
    loss_t2i = F.cross_entropy(sim_t2i, labels)

    print(f"\n[DEBUG] Loss i2t: {loss_i2t.item()}, Loss t2i: {loss_t2i.item()}")

    total_loss = (loss_i2t + loss_t2i) / 2
    print(f"[DEBUG] Total loss: {total_loss.item()}")

    # terminate all code kernel wehatever interrupt
    raise SystemExit

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
    image_features_long = image_embedding / image_embedding.norm(dim=1, keepdim=True)
    text_features_short = short_embedding / short_embedding.norm(dim=1, keepdim=True)
    
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