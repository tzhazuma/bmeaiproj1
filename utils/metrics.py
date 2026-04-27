import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(gt, pred, data_range=None):
    gt = gt.squeeze()
    pred = pred.squeeze()
    if data_range is None:
        data_range = max(gt.max(), pred.max()) - min(gt.min(), pred.min())
        if data_range == 0:
            data_range = 1.0
    return peak_signal_noise_ratio(gt, pred, data_range=data_range)


def compute_ssim(gt, pred, data_range=None):
    gt = gt.squeeze()
    pred = pred.squeeze()
    if data_range is None:
        data_range = max(gt.max(), pred.max()) - min(gt.min(), pred.min())
        if data_range == 0:
            data_range = 1.0
    return structural_similarity(gt, pred, data_range=data_range)
