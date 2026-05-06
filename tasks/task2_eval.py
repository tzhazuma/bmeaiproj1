"""
Task 2: Baseline U-Net Evaluation
Computes PSNR/SSIM on test set and generates reconstruction visualizations.
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from models.unet import UNet
from utils.config_helpers import normalize_output_dir
from utils.metrics import compute_psnr, compute_ssim
from utils.visualize import plot_reconstruction


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


def main():
    config = load_config()
    cfg = config['task2']
    output_dir = os.path.join(config['output']['dir'], 'task2')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = config['data'].get('num_workers', 0)
    pin_memory = device.type == 'cuda'
    print(f"Using device: {device}")

    _, _, test_ds, _ = create_dataloaders(config)
    test_loader = create_dataloader(test_ds, cfg['batch_size'], False, num_workers, pin_memory)

    model = UNet(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        base_channels=cfg['base_channels'],
        depth=cfg['depth']
    ).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    model_path = os.path.join(output_dir, 'unet_baseline_final.pth')
    checkpoint_path = os.path.join(config['output']['dir'], 'task2', 'checkpoint.pth')

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    elif os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No trained model found. Run task2_train.py first.")
        sys.exit(1)

    model.eval()

    results = {
        'psnr_before': [],
        'psnr_after': [],
        'ssim_before': [],
        'ssim_after': [],
    }

    aliased_samples = []
    recon_samples = []
    gt_samples = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            aliased = move_image_tensor(batch['aliased'], device)
            gt = move_image_tensor(batch['gt'], device)

            output = model(aliased)

            for i in range(aliased.size(0)):
                a = aliased[i, 0].cpu().numpy()
                r = output[i, 0].cpu().numpy()
                g = gt[i, 0].cpu().numpy()

                psnr_before = compute_psnr(g, a)
                psnr_after = compute_psnr(g, r)
                ssim_before = compute_ssim(g, a)
                ssim_after = compute_ssim(g, r)

                results['psnr_before'].append(psnr_before)
                results['psnr_after'].append(psnr_after)
                results['ssim_before'].append(ssim_before)
                results['ssim_after'].append(ssim_after)

                if len(gt_samples) < 8:
                    aliased_samples.append(a)
                    recon_samples.append(r)
                    gt_samples.append(g)

    if not results['psnr_after']:
        raise RuntimeError('No test slices were evaluated.')

    # Summary statistics
    summary = {}
    for key, vals in results.items():
        summary[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        }

    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation Results:")
    print(f"  PSNR Before: {summary['psnr_before']['mean']:.2f} ± {summary['psnr_before']['std']:.2f} dB")
    print(f"  PSNR After:  {summary['psnr_after']['mean']:.2f} ± {summary['psnr_after']['std']:.2f} dB")
    print(f"  SSIM Before: {summary['ssim_before']['mean']:.4f} ± {summary['ssim_before']['std']:.4f}")
    print(f"  SSIM After:  {summary['ssim_after']['mean']:.4f} ± {summary['ssim_after']['std']:.4f}")

    # Visualization
    plot_reconstruction(
        aliased_samples, recon_samples, gt_samples,
        os.path.join(output_dir, 'reconstruction_results.png'),
        metrics={
            'psnr_before': [compute_psnr(g, a) for g, a in zip(gt_samples, aliased_samples)],
            'psnr_after': [compute_psnr(g, r) for g, r in zip(gt_samples, recon_samples)],
            'ssim_before': [compute_ssim(g, a) for g, a in zip(gt_samples, aliased_samples)],
            'ssim_after': [compute_ssim(g, r) for g, r in zip(gt_samples, recon_samples)],
        }
    )
    print(f"\nSaved evaluation results to {output_dir}")


if __name__ == '__main__':
    main()
