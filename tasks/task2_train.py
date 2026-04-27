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


def create_grad_scaler(device, use_amp):
    if not use_amp or device.type != 'cuda':
        return None
    return torch.amp.GradScaler('cuda')


def autocast_context(device, use_amp):
    return torch.amp.autocast(device_type=device.type, enabled=use_amp)


def create_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    base_dataset = getattr(dataset, 'dataset', dataset)
    if getattr(base_dataset, '_sample_cache', None) is not None:
        num_workers = 0
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4
    return DataLoader(dataset, **loader_kwargs)


def move_image_tensor(batch_tensor, device):
    tensor = batch_tensor.to(device, non_blocking=True)
    if device.type == 'cuda':
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('BMEAI_CONFIG')
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config',
            'config.yaml',
        )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        aliased = move_image_tensor(batch['aliased'], device)
        gt = move_image_tensor(batch['gt'], device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            output = model(aliased)
            loss = criterion(output, gt)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        aliased = move_image_tensor(batch['aliased'], device)
        gt = move_image_tensor(batch['gt'], device)

        with autocast_context(device, use_amp):
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
    use_amp = config.get('runtime', {}).get('use_amp', device.type == 'cuda')
    num_workers = config['data'].get('num_workers', 0)
    pin_memory = device.type == 'cuda'
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    print(f"Using device: {device}")
    print(f"AMP enabled: {use_amp}")

    print("Loading dataset...")
    train_ds, val_ds, test_ds, mask = create_dataloaders(config)

    train_loader = create_dataloader(train_ds, cfg['batch_size'], True, num_workers, pin_memory)
    val_loader = create_dataloader(val_ds, cfg['batch_size'], False, num_workers, pin_memory)

    print(f"Train: {len(train_ds)} slices, Val: {len(val_ds)} slices, Test: {len(test_ds)} slices")

    model = UNet(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        base_channels=cfg['base_channels'],
        depth=cfg['depth']
    ).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                                 weight_decay=cfg['weight_decay'])
    scaler = create_grad_scaler(device, use_amp)

    scheduler_type = cfg['lr_scheduler']['type']
    scheduler = None
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg['lr_scheduler']['factor'],
            patience=cfg['lr_scheduler']['patience'],
            min_lr=cfg['lr_scheduler']['min_lr'],
        )

    logger = TrainingLogger(config['output']['dir'], 'task2')
    logger.save_config(config)
    best_val_loss = float('inf')
    start_epoch = 0
    checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
    if config.get('runtime', {}).get('resume', True) and os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = logger.load_model(
            checkpoint_path,
            model,
            optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch += 1
        print(f"Resumed from {checkpoint_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{cfg['num_epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     scaler=scaler, use_amp=use_amp)
        val_loss = validate_one_epoch(model, val_loader, criterion, device, use_amp=use_amp)

        current_lr = optimizer.param_groups[0]['lr']
        logger.log_epoch(train_loss, val_loss, current_lr)

        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.save_model(
                model,
                optimizer,
                epoch,
                best_val_loss,
                scheduler=scheduler,
                scaler=scaler,
            )
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
