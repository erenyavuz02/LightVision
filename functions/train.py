import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import torch.nn.functional as F
from functions.mod_77_token_training import mod_77_long_clip_loss, process_batch_subsections_vectorized
import copy

def get_positional_embedding(model, lambda2: int = 4):
    """
    Get modified positional embedding for text encoder based on the given formula.
    """
    pos_embed = model.text_encoder.get_positional_embedding().pos_embed.pos_embed
    if pos_embed is None:
        raise ValueError("Positional embedding not found in text encoder.")

    max_pos, embed_dim = pos_embed.shape[2], pos_embed.shape[3]
    modified_pos_embed = torch.zeros((1, 1, max_pos, embed_dim), device=pos_embed.device)

    for pos in range(max_pos):
        if pos <= 20:
            modified_pos_embed[:, :, pos, :] = pos_embed[:, :, pos, :]
        else:
            lower_idx = pos // lambda2
            upper_idx = min(lower_idx + 1, max_pos - 1)  # Ensure upper_idx is within bounds
            alpha = (pos % lambda2) / lambda2
            modified_pos_embed[:, :, pos, :] = (1 - alpha) * pos_embed[:, :, lower_idx, :] + alpha * pos_embed[:, :, upper_idx, :]
    
    # turn the torch tensor into nn parameter
    modified_pos_embed = torch.nn.Parameter(modified_pos_embed, requires_grad=False)
    return modified_pos_embed

def apply_positional_embedding_modification(model, lambda2=4):
    """
    Apply the positional embedding modification to the model.
    """
    print("Original positional embedding:", model.get_positional_embedding())
    
    # Get modified positional embedding
    new_pos_embed = get_positional_embedding(model, lambda2)
    print("Modified Positional Embedding shape:", new_pos_embed.shape)
    
    # Set the model's pos embedding to the new one
    model.text_encoder.get_positional_embedding().pos_embed.pos_embed = new_pos_embed
    print("Positional embedding successfully modified!")
    
    return model

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    Contrastive loss function for CLIP training.
    """
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Create labels (diagonal should be positive pairs)
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Calculate loss for both directions
    loss_img = nn.CrossEntropyLoss()(logits, labels)
    loss_txt = nn.CrossEntropyLoss()(logits.T, labels)
    
    return (loss_img + loss_txt) / 2

def long_clip_loss(model, image_embedding, long_embedding, short_embedding):
    image_features_long = image_embedding
    text_features_long = long_embedding
    text_features_short = short_embedding

    # Normalize features
    image_features_long = image_features_long / image_features_long.norm(dim=1, keepdim=True)
    text_features_long = text_features_long / text_features_long.norm(dim=1, keepdim=True)
    text_features_short = text_features_short / text_features_short.norm(dim=1, keepdim=True)

    # Apply PCA to get compressed image features
    image_features_short = PCA(image_features_long, 32)
    image_features_short = image_features_short / image_features_short.norm(dim=1, keepdim=True)

    # Since we're not using distributed training, simplify this part
    image_feat_all_long = image_features_long
    image_features_all_short = image_features_short
    text_feat_all_long = text_features_long
    text_feat_all_short = text_features_short

    # Calculate similarity matrices
    sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T)
    
    sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T)
    sim_tl2i = sim_tl2i.T

    sim_i2ts = torch.matmul(image_features_short, text_feat_all_short.T)
    sim_ts2i = torch.matmul(image_features_all_short, text_features_short.T)
    sim_ts2i = sim_ts2i.T

    # Apply temperature scaling
    logit_scale = model.logit_scale if hasattr(model, 'logit_scale') else 1.0

    if isinstance(logit_scale, torch.nn.Parameter):
        sim_i2tl = logit_scale.exp() * sim_i2tl
        sim_tl2i = logit_scale.exp() * sim_tl2i
        sim_i2ts = logit_scale.exp() * sim_i2ts
        sim_ts2i = logit_scale.exp() * sim_ts2i

    # Create targets for loss calculation
    bs = image_embedding.size(0)
    targets = torch.arange(bs, device=image_embedding.device)

    # Calculate losses
    loss_itcl = (
        F.cross_entropy(sim_i2tl, targets, label_smoothing=0.1)
        + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.1)
    ) / 2

    loss_itcs = (
        F.cross_entropy(sim_i2ts, targets, label_smoothing=0.1)
        + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.1)
    ) / 2

    # single loss by combining the two
    total_loss = (loss_itcl + loss_itcs) / 2

    return total_loss


def train_model(model, config, dataset, num_epochs=10, batch_size=32, learning_rate=1e-4, tokenizer=None, training_mode = 'standard', use_mod_77=True):
    """
    Train the model with positional embedding modification and custom dataset.
    
    Args:
        model: The mobile CLIP model to train
        config: Configuration dictionary
        dataset: Dataset object
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        tokenizer: Text tokenizer
        training_mode: Training mode, 'standard' , 'randomized'
        use_mod_77: Whether to use mod 77 token training logic
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # duplicate the model to avoid modifying the original
    # FIXED: Proper way to copy a PyTorch model

    try:
        model = copy.deepcopy(model)
        print("✓ Model successfully deep copied")
    except Exception as e:
        print(f"⚠ Warning: Could not deep copy model ({e}). Using original model.")
        # Use original model directly
    model = model.to(device)
    
    # Initialize tracking lists
    train_losses = []
    val_losses = []
    epochs_list = []
    
    # Step 1: Apply positional embedding modification
    print("Applying positional embedding modification...")
    model = apply_positional_embedding_modification(model, lambda2=4)
    
    # Step 2: Load and split dataset
    train_loader = dataset.get_dataloader("train", batch_size=batch_size, shuffle=True)
    test_loader = dataset.get_dataloader("test" , batch_size=batch_size, shuffle=False)
    
    # Check if dataset supports mod 77 token training
    sample_batch = next(iter(train_loader))
    has_subsections = 'long_splitted_captions' in sample_batch 
    
    
    # Step 3: Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Step 4: Training loop
    model.train()
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):

            images = batch_data['images'].to(device)  
            short_captions = tokenizer(batch_data['short_captions']).to(device)
            long_captions_batch = batch_data['long_captions']
            
            if training_mode == 'randomized':
                # Randomly select subsections for each sample
                long_splitted_captions = batch_data['long_splitted_captions']
                
                for i, subsections in enumerate(long_splitted_captions):
                    num_captions = len(subsections)
                    
                    # Calculate probabilities based on position (first caption gets higher probability)
                    # Using exponential decay: first caption gets highest probability
                    weights = [np.exp(-0.5 * j) for j in range(num_captions)]
                    total_weight = sum(weights)
                    probabilities = [weight / total_weight for weight in weights]
                    
                    # Select based on position-weighted probability
                    selected_idx = np.random.choice(len(subsections), p=probabilities)
                    long_captions_batch[i] = subsections[selected_idx]

            long_captions = tokenizer(long_captions_batch).to(device)

            optimizer.zero_grad()

            # Forward pass
            image_features = model.encode_image(images)
            text_features_short = model.encode_text(short_captions)
            text_features_long = model.encode_text(long_captions)
            
            long_splitted_captions = batch_data['long_splitted_captions']

            loss = long_clip_loss(model, image_features, text_features_long, text_features_short)

            if use_mod_77 :
                subsection_features_per_sample = process_batch_subsections_vectorized(
                    model, tokenizer, long_splitted_captions, device
                )
                sub_caption_loss = mod_77_long_clip_loss(model, image_features, subsection_features_per_sample)
                loss = ((loss * 2) + sub_caption_loss) / 3

            # Backward pass
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            progress_bar.set_postfix({'loss': batch_loss, 'avg_loss': total_loss/(batch_idx+1), 'mode': training_mode})
        
        # Update learning rate
        #scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        epochs_list.append(epoch + 1)
        
        # Validation
        val_loss = validate_model(model, test_loader, device, tokenizer, use_mod_77, training_mode)
        val_losses.append(val_loss)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Training Loss: {avg_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Elapsed Time: {elapsed_time:.2f}s")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, 'best_model.pth')
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
        
        # Plot training progress at the very end
        if epoch == num_epochs - 1:
            plot_training_progress(epochs_list, train_losses, val_losses)
        
        print("-" * 50)
    
    print("Training completed!")
    plot_final_results(epochs_list, train_losses, val_losses, elapsed_time)
    return model

def validate_model(model, test_loader, device, tokenizer=None, use_mod_77=True, training_mode='standard'):
    """
    Validate the model on test dataset.
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            images = batch_data['images'].to(device)  
            short_captions = tokenizer(batch_data['short_captions']).to(device)
            long_captions_batch = batch_data['long_captions']
            
            if training_mode == 'randomized':
                # Randomly select subsections for each sample
                long_splitted_captions = batch_data['long_splitted_captions']
                
                for i, subsections in enumerate(long_splitted_captions):
                    # select the first subsection for validation
                    long_captions_batch[i] = subsections[0]

            long_captions = tokenizer(long_captions_batch).to(device)
            
            # Forward pass
            image_features = model.encode_image(images)
            text_features_short = model.encode_text(short_captions)
            text_features_long = model.encode_text(long_captions)
            
            # Calculate base loss (same as in training)
            loss = long_clip_loss(model, image_features, text_features_long, text_features_short)
            
            # Add mod 77 loss if enabled and data is available
            if use_mod_77 and 'long_splitted_captions' in batch_data:
                long_splitted_captions = batch_data['long_splitted_captions']
                subsection_features_per_sample = process_batch_subsections_vectorized(
                    model, tokenizer, long_splitted_captions, device
                )
                sub_caption_loss = mod_77_long_clip_loss(model, image_features, subsection_features_per_sample)
                loss = ((loss * 2) + sub_caption_loss) / 3
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches

def plot_training_progress(epochs, train_losses, val_losses):
    """
    Plot training and validation losses in real-time.
    """
    
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Recent loss trend (last 10 epochs)
    plt.subplot(1, 2, 2)
    recent_epochs = epochs[-10:] if len(epochs) > 10 else epochs
    recent_train = train_losses[-10:] if len(train_losses) > 10 else train_losses
    recent_val = val_losses[-10:] if len(val_losses) > 10 else val_losses
    
    plt.plot(recent_epochs, recent_train, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    plt.plot(recent_epochs, recent_val, 'r-o', label='Validation Loss', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Recent Training Trend')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_final_results(epochs, train_losses, val_losses, total_time):
    """
    Plot final training results with additional statistics.
    """
    plt.figure(figsize=(15, 5))
    
    # Main loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Complete Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss improvement rate
    plt.subplot(1, 3, 2)
    if len(train_losses) > 1:
        train_diff = np.diff(train_losses)
        val_diff = np.diff(val_losses)
        plt.plot(epochs[1:], train_diff, 'b-', label='Train Loss Change', linewidth=2)
        plt.plot(epochs[1:], val_diff, 'r-', label='Val Loss Change', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change')
        plt.title('Loss Improvement Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Training statistics
    plt.subplot(1, 3, 3)
    stats_text = f"""
    Final Training Loss: {train_losses[-1]:.4f}
    Final Validation Loss: {val_losses[-1]:.4f}
    Best Validation Loss: {min(val_losses):.4f}
    Best Epoch: {np.argmin(val_losses) + 1}
    Total Training Time: {total_time/60:.1f} min
    Avg Time per Epoch: {total_time/len(epochs):.1f}s
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.axis('off')
    plt.title('Training Statistics')
    
    plt.tight_layout()
    plt.show()


#rewrite PCA to avoid inf
def PCA(input_tensor, PCA_dim):
    # 计算均值
    mean = torch.mean(input_tensor, dim=0)
    # 去均值
    X_centered = input_tensor - mean.unsqueeze(0)
    X_centered = X_centered.float()

    # 使用SVD而不是eig来计算主成分
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    principal_components = Vt.T[:, :PCA_dim]

    # 转换到新的维度
    X_transformed = torch.mm(X_centered, principal_components)
    # 恢复到原始空间
    X_reversed = torch.mm(X_transformed, principal_components.T)
    X_reversed += mean

    return X_reversed


