import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
import shutil
from pathlib import Path

# Import our modules (defined above)
from dataset import SinogramDataset, create_dataloaders
from model import UNet, LighterUNet
from training import train_model
from evaluation import evaluate_model

def get_timestamp():
    """Create a timestamp string for directory naming"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    parser = argparse.ArgumentParser(description='Sinogram Restoration')
    parser.add_argument('--data_dir', type=str, default='2e9div', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='log', help='Base directory for logs and visualizations')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode: train or test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--attention', type=bool, default=False, help='attention')
    parser.add_argument('--pretrain', type=bool, default=False, help='pretrain')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--light', type=bool, default=False, help='light model')
    args = parser.parse_args()
    
    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)
    
    # Create model 
    if not args.light:
        model = UNet(n_channels=1, n_classes=1, bilinear=False, attention=args, pretrain=args.pretrain)
    else:
        model = LighterUNet(n_channels=1, n_classes=1, bilinear=False, attention=args, pretrain=args.pretrain)
    # Create timestamped log directory
    timestamp = get_timestamp()
    run_log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(run_log_dir, exist_ok=True)
    
    # Create models directory if it doesn't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Define model checkpoint path
    model_path = os.path.join(args.models_dir, f"model_{timestamp}.pth")
    
    # Save a copy of the run parameters
    with open(os.path.join(run_log_dir, 'run_params.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    if args.mode == 'train':
        # Initialize epoch and states
        start_epoch = 0
        optimizer_state = None
        scaler_state = None
        scheduler_state = None
        best_loss = float('inf')
        random_state = None
        vis_data = None
        
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"Loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                optimizer_state = checkpoint['optimizer_state_dict']
                scaler_state = checkpoint.get('scaler', None)
                scheduler_state = checkpoint.get('scheduler_state', None)
                best_loss = checkpoint.get('best_loss', checkpoint['loss'])
                random_state = checkpoint.get('random_state', None)
                vis_data = checkpoint.get('vis_data', None)
                print(f"Resuming from epoch {start_epoch} with best loss {best_loss:.6f}")
                
                # Copy the checkpoint to the log directory for reference
                shutil.copy2(args.resume, os.path.join(run_log_dir, 'resume_checkpoint.pth'))
            else:
                print(f"No checkpoint found at '{args.resume}'")
        
        # Train model
        model = train_model(
            model, 
            train_loader, 
            test_loader, 
            num_epochs=args.num_epochs,
            start_epoch=start_epoch,
            device=device, 
            save_path=model_path,
            vis_dir=run_log_dir,
            optimizer_state=optimizer_state,
            scaler_state=scaler_state,
            scheduler_state=scheduler_state,
            best_loss=best_loss,
            random_state=random_state,
            vis_data=vis_data,
            lr=args.lr
        )
        
    else:
        # For testing, load the specified checkpoint
        checkpoint_path = args.resume if args.resume else model_path
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            print(f"Loaded model from {checkpoint_path}")
        else:
            print(f"No checkpoint found at '{checkpoint_path}', using untrained model")
    
    # Evaluate model and save results to log directory
    evaluate_model(model, test_loader, device, output_dir=run_log_dir)
    
    # Save final model
    final_model_path = os.path.join(run_log_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_model_path)
    print(f"Final model saved at {final_model_path}")

if __name__ == '__main__':
    main()