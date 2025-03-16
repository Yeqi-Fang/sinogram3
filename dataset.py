import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class SinogramDataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None):
        self.data_dir = data_dir
        self.is_train = is_train
        self.transform = transform
        
        # Determine dataset range based on train/test
        if is_train:
            self.i_range = range(1, 171 - 169)  # 1 to 170
        else:
            self.i_range = range(1, 37 - 35)   # 1 to 36
            
        self.j_range = range(1, 1765)  # 1 to 1764
        
        # Create all possible (i,j) pairs
        self.pairs = [(i, j) for i in self.i_range for j in self.j_range]
        
        # Preload all data into memory
        print(f"Preloading {'training' if is_train else 'testing'} data into memory...")
        self.incomplete_data = {}
        self.complete_data = {}
        
        for i, j in tqdm(self.pairs):
            # Define file paths
            incomplete_path = os.path.join(self.data_dir, f"incomplete_{i}_{j}.npy")
            complete_path = os.path.join(self.data_dir, f"complete_{i}_{j}.npy")
            
            # Load data as float16 to save memory during preloading
            self.incomplete_data[(i, j)] = np.load(incomplete_path).astype(np.float16)
            self.complete_data[(i, j)] = np.load(complete_path).astype(np.float16)
        
        print(f"Successfully preloaded {len(self.pairs)} pairs of sinograms")
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        
        # Get data from memory
        incomplete = self.incomplete_data[(i, j)].astype(np.float32)
        complete = self.complete_data[(i, j)].astype(np.float32)
        
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