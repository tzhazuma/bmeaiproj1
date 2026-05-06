"""
Task 3: Multi-modal Fusion and Advanced Reconstruction Training
Implements unrolled reconstruction network with Data Consistency (DC) layers.
Uses undersampled T2 + fully sampled T1 as input.
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
from models.unrolled_net import UnrolledReconNet
from models.losses import HybridLoss
from utils.config_helpers import normalize_output_dir
from utils.logger import TrainingLogger
from utils.torch_compat import build_adam_optimizer
from utils.visualize import plot_loss_curves


def create_grad_scaler(device, use_amp):
    if not use_amp or device.type != 'cuda':
        return None
    return torch.amp.GradScaler('cuda')


def autocast_context(device, use_amp):
    return torch.amp.autocast(device_type=device.type, enabled=use_amp)


def create_dataloader(
    dataset, batch_size, shuffle, num_workers, pin_memory,
    workers_with_slice_cache=False,
):
    base_dataset = getattr(dataset, 'dataset', dataset)
    if getattr(base_dataset, '_sample_cache', None) is not None and not workers_with_slice_cache:
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
    config_path = os.path.normpath(os.path.abspath(config_path))
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    normalize_output_dir(config, config_path)
    return config


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_amp=False):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        aliased = move_image_tensor(batch['aliased'], device)
        gt = move_image_tensor(batch['gt'], device)
        t1_full = move_image_tensor(batch['t1_full'], device)
        kspace_us = batch['kspace_us'].to(device, non_blocking=True)
        mask = move_image_tensor(batch['mask'], device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            output = model(aliased, t1_full, kspace_us, mask)
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
        t1_full = move_image_tensor(batch['t1_full'], device)
        kspace_us = batch['kspace_us'].to(device, non_blocking=True)
        mask = move_image_tensor(batch['mask'], device)

        with autocast_context(device, use_amp):
            output = model(aliased, t1_full, kspace_us, mask)
            loss = criterion(output, gt)
        running_loss += loss.item()

    return running_loss / len(dataloader)


def main():
    config = load_config()
    cfg = config['task3']
    output_dir = os.path.join(config['output']['dir'], 'task3')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = config.get('runtime', {}).get('use_amp', device.type == 'cuda')
    num_workers = config['data'].get('num_workers', 0)
    workers_with_cache = bool(config['data'].get('dataloader_workers_with_slice_cache', False))
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

    if workers_with_cache and getattr(getattr(train_ds, 'dataset', train_ds), '_sample_cache', None):
        print(
            "DataLoader: dataloader_workers_with_slice_cache=true (multiprocessing may duplicate "
            "RAM per worker on Windows; only enable with enough system memory)."
        )
    eff_workers = num_workers if workers_with_cache else (
        0 if getattr(getattr(train_ds, 'dataset', train_ds), '_sample_cache', None) else num_workers
    )
    if eff_workers == 0 and num_workers > 0:
        print(
            f"DataLoader: num_workers forced to 0 because in-RAM slice cache is enabled "
            f"(set data.dataloader_workers_with_slice_cache: true to use num_workers={num_workers})."
        )

    train_loader = create_dataloader(
        train_ds, cfg['batch_size'], True, num_workers, pin_memory,
        workers_with_slice_cache=workers_with_cache,
    )
    val_loader = create_dataloader(
        val_ds, cfg['batch_size'], False, num_workers, pin_memory,
        workers_with_slice_cache=workers_with_cache,
    )

    print(f"Train: {len(train_ds)} slices, Val: {len(val_ds)} slices, Test: {len(test_ds)} slices")

    model = UnrolledReconNet(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        base_channels=cfg['base_channels'],
        depth=cfg['depth'],
        num_cascades=cfg['num_cascades'],
        dc_weight=cfg['dc_weight'],
    ).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    loss_type = cfg['loss_type']
    if loss_type == 'hybrid':
        criterion = HybridLoss(l1_weight=cfg['l1_weight'])
        print(f"Using Hybrid Loss (L1 weight: {cfg['l1_weight']})")
    elif loss_type == 'l1':
        criterion = nn.L1Loss()
        print("Using L1 Loss")
    else:
        criterion = nn.MSELoss()
        print("Using L2 (MSE) Loss")

    optimizer = build_adam_optimizer(
        model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
    )
    scaler = create_grad_scaler(device, use_amp)

    scheduler_type = cfg['lr_scheduler']['type']
    scheduler = None
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg['lr_scheduler']['factor'],
            patience=cfg['lr_scheduler']['patience'],
            min_lr=cfg['lr_scheduler']['min_lr'],
        )

    logger = TrainingLogger(config['output']['dir'], 'task3')
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

    early_stop_patience = int(cfg.get('early_stop_patience') or 0)
    epochs_without_improve = 0
    if early_stop_patience:
        print(f"Early stopping: stop if val loss does not improve for {early_stop_patience} epochs")

    stopped_early = False
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

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improve = 0
            logger.save_model(
                model,
                optimizer,
                epoch,
                best_val_loss,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
        elif early_stop_patience:
            epochs_without_improve += 1
            print(f"  -> No val improvement ({epochs_without_improve}/{early_stop_patience} epochs)")

        logger.save_metrics()

        if early_stop_patience and epochs_without_improve >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (val loss plateau).")
            stopped_early = True
            break

    plot_loss_curves(
        logger.metrics['train_loss'],
        logger.metrics['val_loss'],
        os.path.join(output_dir, 'loss_curves.png'),
        title='Task 3: Multi-modal Unrolled Network'
    )

    torch.save(model.state_dict(), os.path.join(output_dir, 'unrolled_net_final.pth'))
    print(f"\nSaved model to {output_dir}")
    if stopped_early:
        print("Training finished early (early stopping). Run task3_eval.py next.")
    else:
        print("Task 3 training completed! Run task3_eval.py next.")


if __name__ == '__main__':
    main()
