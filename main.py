import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our modules (defined above)
from dataset import SinogramDataset, create_dataloaders
from model import UNet
from training import train_model
from evaluation import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Sinogram Restoration')
    parser.add_argument('--data_dir', type=str, default='2e9div', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='sinogram_model.pth', help='Path to save model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    
    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)
    
    # Create model 
    model = UNet(n_channels=1, n_classes=1, bilinear=False)
    
    if args.mode == 'train':
        # Train model
        model = train_model(model, train_loader, test_loader, 
                           num_epochs=args.num_epochs, 
                           device=device, 
                           save_path=args.save_path)
    else:
        # Load model for testing
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    
    # Evaluate model
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()