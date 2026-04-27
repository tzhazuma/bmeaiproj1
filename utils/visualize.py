import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_undersampling(mask, full_img, aliased_img, save_path, slice_names=None):
    if slice_names is None:
        slice_names = [f"Slice {i+1}" for i in range(len(full_img))]

    n = len(full_img)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        axes[i, 0].imshow(mask[i] if len(mask.shape) == 3 else mask,
                          cmap='gray', interpolation='nearest')
        axes[i, 0].set_title(f"Sampling Mask\n{slice_names[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(full_img[i], cmap='gray')
        axes[i, 1].set_title(f"Fully Sampled\n{slice_names[i]}")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(aliased_img[i], cmap='gray')
        axes[i, 2].set_title(f"Aliased (AF=5)\n{slice_names[i]}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction(aliased, recon, gt, save_path, metrics=None, n_examples=4):
    n = min(n_examples, len(aliased))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        axes[i, 0].imshow(aliased[i], cmap='gray')
        title = "Aliased"
        if metrics and 'psnr_before' in metrics and i < len(metrics['psnr_before']):
            title += f"\nPSNR: {metrics['psnr_before'][i]:.2f} | SSIM: {metrics['ssim_before'][i]:.4f}"
        axes[i, 0].set_title(title, fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(recon[i], cmap='gray')
        title = "Reconstructed"
        if metrics and 'psnr_after' in metrics and i < len(metrics['psnr_after']):
            title += f"\nPSNR: {metrics['psnr_after'][i]:.2f} | SSIM: {metrics['ssim_after'][i]:.4f}"
        axes[i, 1].set_title(title, fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(gt[i], cmap='gray')
        axes[i, 2].set_title("Ground Truth", fontsize=9)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_losses, val_losses, save_path, title="Training Curves"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', linewidth=1)
    ax1.plot(val_losses, label='Val Loss', linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(train_losses, label='Train Loss', linewidth=1)
    ax2.semilogy(val_losses, label='Val Loss', linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title(f'{title} - Log Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_analysis(worst_cases, save_path):
    n = len(worst_cases)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, case in enumerate(worst_cases):
        axes[i, 0].imshow(case['aliased'], cmap='gray')
        axes[i, 0].set_title(f"Aliased\nPSNR: {case.get('psnr_before', 0):.2f}",
                             fontsize=9)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(case['recon'], cmap='gray')
        axes[i, 1].set_title(f"Reconstructed\nPSNR: {case['psnr_after']:.2f} | SSIM: {case['ssim']:.4f}",
                             fontsize=9)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(case['gt'], cmap='gray')
        axes[i, 2].set_title("Ground Truth", fontsize=9)
        axes[i, 2].axis('off')

        diff = np.abs(case['gt'] - case['recon'])
        axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f"Error Map\nMax: {diff.max():.4f}", fontsize=9)
        axes[i, 3].axis('off')

    plt.suptitle("5 Worst Reconstruction Cases", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
