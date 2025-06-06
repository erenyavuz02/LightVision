import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

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

def train_model(model, config, dataset, num_epochs=10, batch_size=32, learning_rate=1e-4):
    """
    Train the model with positional embedding modification and custom dataset.
    
    Args:
        model: The mobile CLIP model to train
        config: Configuration dictionary
        dataset_path: Path to the all_captions.json file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize tracking lists
    train_losses = []
    val_losses = []
    epochs_list = []
    
    # Step 1: Apply positional embedding modification
    print("Applying positional embedding modification...")
    model = apply_positional_embedding_modification(model, lambda2=4)
    
    # Step 2: Load and split dataset
    # Create datasets and dataloaders
    train_loader = dataset.get_dataloader("train")
    test_loader = dataset.get_dataloader("test")
    
    
    # Step 3: Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Step 4: Training loop
    model.train()
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0
        batch_losses = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (images, texts) in enumerate(progress_bar):
            images = images.to(device)
            # texts are already tokenized in the dataset
            
            optimizer.zero_grad()
            
            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            # Calculate contrastive loss
            loss = contrastive_loss(image_features, text_features)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            progress_bar.set_postfix({'loss': batch_loss, 'avg_loss': total_loss/(batch_idx+1)})
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        epochs_list.append(epoch + 1)
        
        # Validation
        val_loss = validate_model(model, test_loader, device)
        val_losses.append(val_loss)
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Training Loss: {avg_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
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
        
        # Plot training progress
        if (epoch + 1) % 2 == 0 or epoch == 0:
            plot_training_progress(epochs_list, train_losses, val_losses)
        
        print("-" * 50)
    
    print("Training completed!")
    plot_final_results(epochs_list, train_losses, val_losses, elapsed_time)
    return model

def validate_model(model, test_loader, device):
    """
    Validate the model on test dataset.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, texts in test_loader:
            images = images.to(device)
            
            # Forward pass
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            # Calculate loss
            loss = contrastive_loss(image_features, text_features)
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(test_loader)

def plot_training_progress(epochs, train_losses, val_losses):
    """
    Plot training and validation losses in real-time.
    """
    clear_output(wait=True)
    
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
