import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class BraTSDataset(Dataset):
    def __init__(self, root_dir, modalities=('T1', 'T2'), slice_axis=2,
                 undersample_mask=None, acceleration_factor=5, sigma=0.3):
        self.root_dir = root_dir
        self.modalities = list(modalities)
        self.slice_axis = slice_axis
        self.undersample_mask = undersample_mask
        self.acceleration_factor = acceleration_factor
        self.sigma = sigma

        self.patient_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.slices = self._build_slice_index()

    def _find_nifti(self, patient_dir, modality):
        for f in os.listdir(patient_dir):
            if modality.lower() in f.lower() and f.endswith('.nii.gz'):
                return os.path.join(patient_dir, f)
        return None

    def _build_slice_index(self):
        indices = []
        for pid, pdir in enumerate(self.patient_dirs):
            full_path = os.path.join(self.root_dir, pdir)
            t2_path = self._find_nifti(full_path, 't2')
            if t2_path is None:
                continue
            vol = nib.load(t2_path).get_fdata()
            n_slices = vol.shape[self.slice_axis]
            for s in range(n_slices):
                indices.append((pid, pdir, s))
        return indices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        pid, pdir, slice_idx = self.slices[idx]
        full_path = os.path.join(self.root_dir, pdir)

        volumes = {}
        for mod in self.modalities:
            nii_path = self._find_nifti(full_path, mod)
            if nii_path is None:
                raise FileNotFoundError(f"Missing {mod} for {pdir}")
            vol = nib.load(nii_path).get_fdata()
            slc = np.take(vol, slice_idx, axis=self.slice_axis)
            slc = slc.astype(np.float32)

            m = slc > 0
            if m.sum() > 0:
                slc[m] = (slc[m] - slc[m].mean()) / (slc[m].std() + 1e-8)
            volumes[mod.lower()] = slc

        t2 = volumes['t2']
        t2_tensor = torch.from_numpy(t2).unsqueeze(0)

        t1 = None
        for mod in self.modalities:
            if mod.lower() == 't1':
                t1 = volumes['t1']
                t1_tensor = torch.from_numpy(t1).unsqueeze(0)
                break

        if self.undersample_mask is None:
            mask = generate_undersampling_mask_cpu(
                t2.shape, self.acceleration_factor, self.sigma
            )
        else:
            mask = self.undersample_mask

        kspace = np.fft.fftshift(np.fft.fft2(t2))
        kspace_us = kspace * mask
        t2_us = np.fft.ifft2(np.fft.ifftshift(kspace_us)).real
        t2_us_tensor = torch.from_numpy(t2_us.astype(np.float32)).unsqueeze(0)

        result = {
            'aliased': t2_us_tensor,
            'gt': t2_tensor,
            'mask': torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
            'kspace_us': torch.from_numpy(
                np.stack([kspace_us.real, kspace_us.imag], axis=0)
            ).float(),
            'slice_idx': slice_idx,
            'patient': pdir,
        }

        if t1 is not None:
            result['t1_full'] = t1_tensor

        return result


def generate_undersampling_mask_cpu(shape, acceleration_factor=5, sigma=0.3):
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


def create_dataloaders(config):
    from .dataset import BraTSDataset

    dataset_path = os.path.expanduser(config['data']['dataset_path'])
    modalities = config['data']['modalities']
    slice_axis = config['data']['slice_axis']
    seed = config['data']['seed']

    # Pre-generate mask
    h, w = 240, 240
    mask = generate_undersampling_mask_cpu(
        (h, w),
        config['task1']['acceleration_factor'],
        config['task1']['sigma']
    )

    full_dataset = BraTSDataset(
        root_dir=dataset_path,
        modalities=modalities,
        slice_axis=slice_axis,
        undersample_mask=mask,
    )

    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    train_ratio = 1 - val_ratio - test_ratio

    total = len(full_dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    return train_ds, val_ds, test_ds, mask
