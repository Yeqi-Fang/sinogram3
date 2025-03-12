import torch.optim as optim
import time
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import os
import numpy as np

def train_model(model, train_loader, test_loader, num_epochs=50, device='cuda', save_path='model.pth', vis_dir='visualizations'):
    # Create directory for visualizations if it doesn't exist
    os.makedirs(vis_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Track best model
    best_loss = float('inf')
    
    # Get a few samples from the test set for visualization
    vis_samples = []
    vis_dataloader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=4, shuffle=True)
    vis_batch = next(iter(vis_dataloader))
    vis_incomplete, vis_complete = vis_batch
    
    # Training loop
    for epoch in range(num_epochs):
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
            
            with autocast():
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
                with autocast():
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
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),  # Save scaler state
                'loss': best_loss,
            }, save_path)
            print(f'Model saved at epoch {epoch+1} with validation loss {best_loss:.6f}')
        
        # Generate and save visualizations for this epoch
        with torch.no_grad():
            model.eval()
            with autocast():
                vis_outputs = model(vis_incomplete.to(device))
            
            # Create a figure to visualize results
            fig, axes = plt.subplots(4, 3, figsize=(15, 20))
            
            for i in range(min(4, len(vis_incomplete))):
                # Get the images
                input_img = vis_incomplete[i, 0].cpu().numpy()
                output_img = vis_outputs[i, 0].cpu().numpy()
                target_img = vis_complete[i, 0].cpu().numpy()
                
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
                plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
            
            # Add epoch information to the title
            plt.suptitle(f'Epoch {epoch+1} - Validation Loss: {avg_val_loss:.6f}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure
            vis_path = os.path.join(vis_dir, f'epoch_{epoch+1:03d}.png')
            plt.savefig(vis_path, dpi=200)
            plt.close(fig)
            print(f"Visualization saved to {vis_path}")
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model