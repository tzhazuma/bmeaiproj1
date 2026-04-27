import copy
import json
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from models.unet import UNet
from models.unrolled_net import UnrolledReconNet


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('BMEAI_CONFIG')
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config',
            'formal_train.yaml',
        )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f), config_path


def create_dataloader(dataset, batch_size, num_workers, pin_memory):
    base_dataset = getattr(dataset, 'dataset', dataset)
    if getattr(base_dataset, '_sample_cache', None) is not None:
        num_workers = 0
    kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4
    return DataLoader(dataset, **kwargs)


def move_image_tensor(batch_tensor, device):
    tensor = batch_tensor.to(device, non_blocking=True)
    if device.type == 'cuda':
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def benchmark_task2(config, device, max_steps=30):
    train_ds, _, _, _ = create_dataloaders(config)
    loader = create_dataloader(train_ds, config['task2']['batch_size'], config['data'].get('num_workers', 0), device.type == 'cuda')
    model = UNet(
        in_channels=config['task2']['in_channels'],
        out_channels=config['task2']['out_channels'],
        base_channels=config['task2']['base_channels'],
        depth=config['task2']['depth'],
    ).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['task2']['learning_rate'])
    criterion = torch.nn.MSELoss()
    use_amp = config.get('runtime', {}).get('use_amp', device.type == 'cuda')

    steps = 0
    start = time.perf_counter()
    for batch in loader:
        aliased = move_image_tensor(batch['aliased'], device)
        gt = move_image_tensor(batch['gt'], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(aliased)
            loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps >= min(max_steps, len(loader)):
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        'dataset_size': len(train_ds),
        'steps_benchmarked': steps,
        'seconds_per_step': elapsed / max(steps, 1),
        'estimated_epoch_seconds': (elapsed / max(steps, 1)) * len(loader),
    }


def benchmark_task3(config, device, max_steps=30):
    train_ds, _, _, _ = create_dataloaders(config)
    loader = create_dataloader(train_ds, config['task3']['batch_size'], config['data'].get('num_workers', 0), device.type == 'cuda')
    model = UnrolledReconNet(
        in_channels=config['task3']['in_channels'],
        out_channels=config['task3']['out_channels'],
        base_channels=config['task3']['base_channels'],
        depth=config['task3']['depth'],
        num_cascades=config['task3']['num_cascades'],
        dc_weight=config['task3']['dc_weight'],
    ).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['task3']['learning_rate'])
    criterion = torch.nn.L1Loss()
    use_amp = config.get('runtime', {}).get('use_amp', device.type == 'cuda')

    steps = 0
    start = time.perf_counter()
    for batch in loader:
        aliased = move_image_tensor(batch['aliased'], device)
        gt = move_image_tensor(batch['gt'], device)
        t1_full = move_image_tensor(batch['t1_full'], device)
        kspace_us = batch['kspace_us'].to(device, non_blocking=True)
        mask = move_image_tensor(batch['mask'], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(aliased, t1_full, kspace_us, mask)
            loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        steps += 1
        if steps >= min(max_steps, len(loader)):
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        'dataset_size': len(train_ds),
        'steps_benchmarked': steps,
        'seconds_per_step': elapsed / max(steps, 1),
        'estimated_epoch_seconds': (elapsed / max(steps, 1)) * len(loader),
    }


def main():
    config, config_path = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    candidate_env = os.environ.get('BMEAI_BENCH_CANDIDATES', '8,12,16,20,24')
    candidate_slices = [int(value.strip()) for value in candidate_env.split(',') if value.strip()]
    max_steps = int(os.environ.get('BMEAI_BENCH_STEPS', '30'))
    results = []
    output_path = os.path.join(os.path.dirname(config_path), 'formal_speed_benchmark.json')
    for slices_per_patient in candidate_slices:
        cfg = copy.deepcopy(config)
        cfg['data']['max_slices_per_patient'] = slices_per_patient
        task2_stats = benchmark_task2(cfg, device, max_steps=max_steps)
        task3_stats = benchmark_task3(cfg, device, max_steps=max_steps)
        result = {
            'config_path': config_path,
            'max_slices_per_patient': slices_per_patient,
            'task2': task2_stats,
            'task3': task3_stats,
            'estimated_total_train_hours': (
                task2_stats['estimated_epoch_seconds'] * cfg['task2']['num_epochs'] +
                task3_stats['estimated_epoch_seconds'] * cfg['task3']['num_epochs']
            ) / 3600.0,
        }
        results.append(result)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(json.dumps(result, indent=2))
    print(f'Benchmark saved to {output_path}')


if __name__ == '__main__':
    main()
