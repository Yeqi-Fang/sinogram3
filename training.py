import torch.optim as optim
import time
from tqdm import tqdm

def train_model(model, train_loader, test_loader, num_epochs=50, device='cuda', save_path='model.pth'):
    # Move model to device
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Track best model
    best_loss = float('inf')
    
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
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(incomplete)
            loss = criterion(outputs, complete)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
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
                
                # Forward pass
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
                'loss': best_loss,
            }, save_path)
            print(f'Model saved at epoch {epoch+1} with validation loss {best_loss:.6f}')
    
    # Load best model
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model