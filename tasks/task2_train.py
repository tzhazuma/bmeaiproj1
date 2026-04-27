"""
Task 2: Baseline U-Net Reconstruction Training
Trains a U-Net to remove aliasing artifacts from undersampled T2 images.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from models.unet import UNet
from utils.logger import TrainingLogger
from utils.visualize import plot_loss_curves


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        aliased = batch['aliased'].to(device)
        gt = batch['gt'].to(device)

        optimizer.zero_grad()
        output = model(aliased)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        aliased = batch['aliased'].to(device)
        gt = batch['gt'].to(device)

        output = model(aliased)
        loss = criterion(output, gt)
        running_loss += loss.item()

    return running_loss / len(dataloader)


def main():
    config = load_config()
    cfg = config['task2']
    output_dir = os.path.join(config['output']['dir'], 'task2')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    train_ds, val_ds, test_ds, mask = create_dataloaders(config)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} slices, Val: {len(val_ds)} slices, Test: {len(test_ds)} slices")

    model = UNet(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        base_channels=cfg['base_channels'],
        depth=cfg['depth']
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])

    scheduler_type = cfg['lr_scheduler']['type']
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg['lr_scheduler']['factor'],
            patience=cfg['lr_scheduler']['patience'],
            min_lr=cfg['lr_scheduler']['min_lr'],
        )

    logger = TrainingLogger(config['output']['dir'], 'task2')
    best_val_loss = float('inf')

    for epoch in range(cfg['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(train_loss, val_loss, current_lr)

        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.save_model(model, optimizer, epoch, best_val_loss)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")

        logger.save_metrics()

    plot_loss_curves(
        logger.metrics['train_loss'],
        logger.metrics['val_loss'],
        os.path.join(output_dir, 'loss_curves.png'),
        title='Task 2: U-Net Baseline'
    )

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'unet_baseline_final.pth'))

    # Save test dataloader info for evaluation
    print(f"\nSaved model to {output_dir}")
    print("Task 2 training completed! Run task2_eval.py next.")


if __name__ == '__main__':
    main()
