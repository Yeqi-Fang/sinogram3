import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class SinogramDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        
        # Determine dataset range based on train/test
        if is_train:
            self.i_range = range(1, 171)  # 1 to 170
        else:
            self.i_range = range(1, 37)   # 1 to 36
            
        self.j_range = range(1, 1765)  # 1 to 1764
        
        # Create all possible (i,j) pairs
        self.pairs = [(i, j) for i in self.i_range for j in self.j_range]
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        
        # Load complete and incomplete sinograms
        incomplete_path = os.path.join(self.data_dir, f"incomplete_{i}_{j}.npy")
        complete_path = os.path.join(self.data_dir, f"complete_{i}_{j}.npy")
        
        incomplete = np.load(incomplete_path).astype(np.float32)
        complete = np.load(complete_path).astype(np.float32)
        
        # Convert to torch tensors
        incomplete = torch.from_numpy(incomplete)
        complete = torch.from_numpy(complete)
        
        # Add channel dimension if not present
        if incomplete.dim() == 2:
            incomplete = incomplete.unsqueeze(0)
        if complete.dim() == 2:
            complete = complete.unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            incomplete = self.transform(incomplete)
            complete = self.transform(complete)
        
        return incomplete, complete


# Example of how to use the dataset
def create_dataloaders(data_dir, batch_size=8, num_workers=4):
    # Define transforms
    transform = None
    
    # Create datasets
    train_dataset = SinogramDataset(os.path.join(data_dir, 'train'), is_train=True, transform=transform)
    test_dataset = SinogramDataset(os.path.join(data_dir, 'test'), is_train=False, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader