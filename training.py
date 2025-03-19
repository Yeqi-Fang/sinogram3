import torch.optim as optim
import time
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os
import random
import numpy as np

def train_model(model, train_loader, test_loader, num_epochs=50, start_epoch=0, device='cuda', 
                save_path='model.pth', vis_dir='visualizations', optimizer_state=None, 
                scaler_state=None, best_loss=float('inf'), scheduler_state=None, 
                random_state=None, vis_data=None, lr=1e-4):
    """
    Train the model with support for resuming from checkpoints
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        num_epochs: Total number of epochs to train
        start_epoch: Epoch to start training from (for resuming)
        device: Device to train on (cuda or cpu)
        save_path: Path to save the best model
        vis_dir: Directory to save visualizations
        optimizer_state: State dict of optimizer (for resuming)
        scaler_state: State dict of gradient scaler (for resuming)
        best_loss: Best validation loss so far (for resuming)
        scheduler_state: State dict of learning rate scheduler (for resuming)
        random_state: Dictionary of random states (for resuming)
        vis_data: Visualization samples from previous run (for resuming)
        
    Returns:
        Trained model
    """
    # Create directory for visualizations if it doesn't exist
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a checkpoint directory within visualizations
    checkpoint_dir = os.path.join(vis_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Restore random states if resuming
    if random_state:
        torch.set_rng_state(random_state['torch'])
        np.random.set_state(random_state['numpy'])
        random.setstate(random_state['python'])
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Load optimizer state if resuming
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
        # Move optimizer state to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler('cuda')
    
    # Load scaler state if resuming
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.3)
    
    # Load scheduler state if resuming
    if scheduler_state:
        scheduler.__dict__.update(scheduler_state)
    
    # Get visualization samples
    if vis_data:
        # Use the same visualization samples as before
        vis_incomplete, vis_complete = vis_data
    else:
        # Get new samples for visualization
        vis_dataloader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=4, shuffle=True)
        vis_batch = next(iter(vis_dataloader))
        vis_incomplete, vis_complete = vis_batch
    
    # Save initial visualizations for reference if starting fresh
    if start_epoch == 0:
        with torch.no_grad():
            model.eval()
            with autocast(device_type='cuda'):
                vis_outputs = model(vis_incomplete.to(device))
            
            save_visualizations(vis_incomplete, vis_outputs, vis_complete, 
                              os.path.join(vis_dir, 'initial_state.png'),
                              title="Initial Model State")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for incomplete, complete in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)'):
            # Move tensors to device
            incomplete = incomplete.to(device)
            complete = complete.to(device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs = model(incomplete)
                loss = criterion(outputs, complete)
            
            # Backward pass and optimize with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update statistics
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for incomplete, complete in tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Test)'):
                # Move tensors to device
                incomplete = incomplete.to(device)
                complete = complete.to(device)
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda'):
                    outputs = model(incomplete)
                    loss = criterion(outputs, complete)
                
                # Update statistics
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save checkpoint every epoch for safety
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        # Capture current random states
        random_states = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate()
        }
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state': scheduler.__dict__,  # Save full scheduler state
            'scaler': scaler.state_dict(),
            'loss': avg_val_loss,
            'best_loss': best_loss,
            'random_state': random_states,
            'vis_data': (vis_incomplete, vis_complete)  # Save visualization data
        }, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.__dict__,
                'scaler': scaler.state_dict(),
                'loss': best_loss,
                'random_state': random_states,
                'vis_data': (vis_incomplete, vis_complete)
            }, save_path)
            
            # Also save to the vis directory with a clear name
            best_model_path = os.path.join(vis_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state': scheduler.__dict__,
                'scaler': scaler.state_dict(),
                'loss': best_loss,
                'random_state': random_states,
                'vis_data': (vis_incomplete, vis_complete)
            }, best_model_path)
            
            print(f'Model saved at epoch {epoch+1} with validation loss {best_loss:.6f}')
        
        # Generate and save visualizations for this epoch
        with torch.no_grad():
            model.eval()
            with autocast(device_type='cuda'):
                vis_outputs = model(vis_incomplete.to(device))
            
            vis_path = os.path.join(vis_dir, f'epoch_{epoch+1:03d}.png')
            save_visualizations(vis_incomplete, vis_outputs, vis_complete, vis_path,
                              title=f'Epoch {epoch+1} - Validation Loss: {avg_val_loss:.6f}')
            print(f"Visualization saved to {vis_path}")
    
    # Load best model
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def save_visualizations(incomplete, outputs, complete, filepath, title="Visualization"):
    """Helper function to save visualizations"""
    # Create a figure to visualize results
    fig, axes = plt.subplots(min(4, len(incomplete)), 3, figsize=(16, 15), gridspec_kw={"width_ratios": [1, 1, 1.0675]})
    
    # Handle the case where there's only one sample
    if min(4, len(incomplete)) == 1:
        axes = axes.reshape(1, 3)
    
    for i in range(min(4, len(incomplete))):
        # Get the images
        input_img = incomplete[i, 0].cpu().numpy()
        output_img = outputs[i, 0].cpu().numpy()
        target_img = complete[i, 0].cpu().numpy()
        
        # Determine global min and max for consistent colormap scaling
        vmin = min(input_img.min(), output_img.min(), target_img.min())
        vmax = max(input_img.max(), output_img.max(), target_img.max())
        
        # Plot input
        im = axes[i, 0].imshow(input_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title('Input (Incomplete)')
        axes[i, 0].axis('off')
        
        # Plot output
        im = axes[i, 1].imshow(output_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title('Output (Predicted)')
        axes[i, 1].axis('off')
        
        # Plot ground truth
        im = axes[i, 2].imshow(target_img, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title('Ground Truth (Complete)')
        axes[i, 2].axis('off')
        
        # Add a colorbar to the last image in each row
        plt.colorbar(im, ax=axes[i, 2], fraction=0.04, pad=0.02)
    
    # Add title information
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig(filepath, dpi=600)
    plt.close(fig)
