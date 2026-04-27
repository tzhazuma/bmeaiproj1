import numpy as np


def generate_undersampling_mask(shape, acceleration_factor=5, sigma=0.3):
    h, w = shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt(((y - center_y) / (h / 2)) ** 2 +
                   ((x - center_x) / (w / 2)) ** 2)

    density = np.exp(-dist ** 2 / (2 * sigma ** 2))
    density = density / density.max()
    density = density * (1 - 1 / acceleration_factor) + (1 / acceleration_factor)

    prob = density / density.sum()
    n_samples = int(np.prod(shape) / acceleration_factor)

    rng = np.random.RandomState(42)
    indices = rng.choice(np.prod(shape), size=n_samples, replace=False, p=prob.ravel())
    mask = np.zeros(shape, dtype=np.float32)
    mask.flat[indices] = 1.0

    mask[center_y - 2:center_y + 3, center_x - 2:center_x + 3] = 1.0
    return mask


def fft2(image):
    return np.fft.fftshift(np.fft.fft2(image))


def ifft2(kspace):
    return np.fft.ifft2(np.fft.ifftshift(kspace)).real


def undersample_kspace(image, mask):
    kspace = fft2(image)
    kspace_us = kspace * mask
    aliased = ifft2(kspace_us)
    return kspace_us, aliased
