"""
Task 3: Multi-modal Fusion Evaluation & Error Analysis
Computes PSNR/SSIM, identifies 5 worst reconstruction cases,
compares with baseline (Task 2), and generates report data.
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
from models.unrolled_net import UnrolledReconNet
from models.unet import UNet
from utils.metrics import compute_psnr, compute_ssim
from utils.visualize import plot_reconstruction, plot_error_analysis


def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    cfg = config['task3']
    output_dir = os.path.join(config['output']['dir'], 'task3')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    _, _, test_ds, _ = create_dataloaders(config)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)

    model = UnrolledReconNet(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        base_channels=cfg['base_channels'],
        depth=cfg['depth'],
        num_cascades=cfg['num_cascades'],
        dc_weight=cfg['dc_weight'],
    ).to(device)

    # Load either checkpoint or final model
    model_path = os.path.join(output_dir, 'unrolled_net_final.pth')
    checkpoint_path = os.path.join(config['output']['dir'], 'task3', 'checkpoint.pth')

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint (epoch {ckpt['epoch'] + 1})")
    elif os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded final model")
    else:
        print("No trained model found. Run task3_train.py first.")
        sys.exit(1)

    model.eval()

    all_cases = []
    results = {
        'psnr_before': [],
        'psnr_after': [],
        'ssim_before': [],
        'ssim_after': [],
    }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            aliased = batch['aliased'].to(device)
            gt = batch['gt'].to(device)
            t1_full = batch['t1_full'].to(device)
            kspace_us = batch['kspace_us'].to(device)
            mask = batch['mask'].to(device)

            output = model(aliased, t1_full, kspace_us, mask)

            for i in range(aliased.size(0)):
                a = aliased[i, 0].cpu().numpy()
                r = output[i, 0].cpu().numpy()
                g = gt[i, 0].cpu().numpy()
                t1 = t1_full[i, 0].cpu().numpy()

                psnr_before = compute_psnr(g, a)
                psnr_after = compute_psnr(g, r)
                ssim_before = compute_ssim(g, a)
                ssim_after = compute_ssim(g, r)

                results['psnr_before'].append(psnr_before)
                results['psnr_after'].append(psnr_after)
                results['ssim_before'].append(ssim_before)
                results['ssim_after'].append(ssim_after)

                all_cases.append({
                    'aliased': a.tolist() if isinstance(a, np.ndarray) else a,
                    'recon': r.tolist() if isinstance(r, np.ndarray) else r,
                    'gt': g.tolist() if isinstance(g, np.ndarray) else g,
                    'psnr_before': float(psnr_before),
                    'psnr_after': float(psnr_after),
                    'ssim_before': float(ssim_before),
                    'ssim_after': float(ssim_after),
                })

    # Summary statistics
    summary = {}
    for key, vals in results.items():
        summary[key] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        }

    print("\n=== Task 3 Evaluation Results ===")
    print(f"  PSNR Before: {summary['psnr_before']['mean']:.2f} ± {summary['psnr_before']['std']:.2f} dB")
    print(f"  PSNR After:  {summary['psnr_after']['mean']:.2f} ± {summary['psnr_after']['std']:.2f} dB")
    print(f"  SSIM Before: {summary['ssim_before']['mean']:.4f} ± {summary['ssim_before']['std']:.4f}")
    print(f"  SSIM After:  {summary['ssim_after']['mean']:.4f} ± {summary['ssim_after']['std']:.4f}")

    # Compare with Task 2 baseline
    task2_summary_path = os.path.join(config['output']['dir'], 'task2', 'evaluation_summary.json')
    if os.path.exists(task2_summary_path):
        with open(task2_summary_path, 'r') as f:
            task2_summary = json.load(f)
        print("\n=== Comparison with Baseline (Task 2) ===")
        print(f"  PSNR Improvement: {summary['psnr_after']['mean'] - task2_summary['psnr_after']['mean']:.2f} dB")
        print(f"  SSIM Improvement: {summary['ssim_after']['mean'] - task2_summary['ssim_after']['mean']:.4f}")

    # Save summary
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Error analysis: find 5 worst cases (lowest PSNR after reconstruction)
    all_cases.sort(key=lambda x: x['psnr_after'])
    worst_cases = all_cases[:5]

    print("\n=== 5 Worst Reconstruction Cases ===")
    for i, case in enumerate(worst_cases):
        print(f"  Case {i+1}: PSNR={case['psnr_after']:.2f} dB, SSIM={case['ssim']:.4f} "
              f"(Before: PSNR={case['psnr_before']:.2f} dB, SSIM={case['ssim_before']:.4f})")

    # Save worst cases data
    worst_cases_path = os.path.join(output_dir, 'worst_cases.json')
    with open(worst_cases_path, 'w') as f:
        json.dump(worst_cases, f, indent=2)

    # Visualize worst cases
    worst_viz = []
    for case in worst_cases:
        worst_viz.append({
            'aliased': np.array(case['aliased']),
            'recon': np.array(case['recon']),
            'gt': np.array(case['gt']),
            'psnr_before': case['psnr_before'],
            'psnr_after': case['psnr_after'],
            'ssim': case['ssim_after'],
        })

    plot_error_analysis(worst_viz, os.path.join(output_dir, 'error_analysis.png'))
    print(f"\nSaved error analysis to {output_dir}")

    # Also generate some good reconstructions for comparison
    all_cases.sort(key=lambda x: x['psnr_after'], reverse=True)
    best_cases = all_cases[:4]

    best_aliased = [np.array(c['aliased']) for c in best_cases]
    best_recon = [np.array(c['recon']) for c in best_cases]
    best_gt = [np.array(c['gt']) for c in best_cases]

    plot_reconstruction(
        best_aliased, best_recon, best_gt,
        os.path.join(output_dir, 'best_reconstructions.png'),
        metrics={
            'psnr_before': [c['psnr_before'] for c in best_cases],
            'psnr_after': [c['psnr_after'] for c in best_cases],
            'ssim_before': [c['ssim_before'] for c in best_cases],
            'ssim_after': [c['ssim_after'] for c in best_cases],
        }
    )

    print("\nTask 3 evaluation completed! All results saved.")


if __name__ == '__main__':
    main()
