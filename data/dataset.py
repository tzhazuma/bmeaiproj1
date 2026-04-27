import os
import random
from functools import lru_cache

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


NIFTI_EXTENSIONS = ('.nii.gz', '.nii')
MODALITY_ALIASES = {
    't1': ('t1n', 't1'),
    't2': ('t2w', 't2'),
    't1ce': ('t1ce', 't1c'),
    'flair': ('flair', 't2f'),
}


def strip_nifti_extension(filename):
    lower = filename.lower()
    for ext in NIFTI_EXTENSIONS:
        if lower.endswith(ext):
            return filename[:-len(ext)]
    return os.path.splitext(filename)[0]


def resolve_dataset_path(dataset_path):
    code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    project_root = os.path.abspath(os.path.join(code_root, '..'))

    candidates = [
        os.path.abspath(os.path.expanduser(dataset_path)),
        os.path.abspath(os.path.join(code_root, dataset_path)),
        os.path.abspath(os.path.join(project_root, dataset_path)),
        os.path.join(project_root, 'bmeaidataset'),
    ]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find dataset directory from '{dataset_path}'. "
        f"Checked: {', '.join(seen)}"
    )


def find_modality_path(patient_dir, modality):
    aliases = MODALITY_ALIASES.get(modality.lower(), (modality.lower(),))
    files = sorted(os.listdir(patient_dir))

    for alias in aliases:
        for filename in files:
            lower = filename.lower()
            if not lower.endswith(NIFTI_EXTENSIONS):
                continue

            stem = strip_nifti_extension(lower)
            tokens = stem.replace('_', '-').split('-')
            if alias in tokens:
                return os.path.join(patient_dir, filename)

    return None


@lru_cache(maxsize=128)
def load_nifti_image(path):
    return nib.load(path)


def load_slice_from_image(image, slice_axis, slice_idx):
    slicer = [slice(None)] * len(image.shape)
    slicer[slice_axis] = slice_idx
    return np.asarray(image.dataobj[tuple(slicer)], dtype=np.float32).copy()


class BraTSDataset(Dataset):
    def __init__(self, root_dir, modalities=('T1', 'T2'), slice_axis=2,
                 undersample_mask=None, acceleration_factor=5, sigma=0.3,
                 max_patients=None, max_slices_per_patient=None,
                 preload_volumes=False):
        self.root_dir = resolve_dataset_path(root_dir)
        self.modalities = list(modalities)
        self.slice_axis = slice_axis
        self.undersample_mask = undersample_mask
        self.acceleration_factor = acceleration_factor
        self.sigma = sigma
        self.max_slices_per_patient = max_slices_per_patient
        self.preload_volumes = preload_volumes
        self.image_shape = None
        self.patient_records = {}
        self._mask_tensor = None
        self._sample_cache = None

        self.patient_dirs = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        if max_patients is not None:
            self.patient_dirs = self.patient_dirs[:max_patients]

        self.slices = self._build_slice_index()

    def _find_nifti(self, patient_dir, modality):
        return find_modality_path(patient_dir, modality)

    def _select_slice_indices(self, n_slices):
        if self.max_slices_per_patient is None or self.max_slices_per_patient >= n_slices:
            return range(n_slices)

        start = max((n_slices - self.max_slices_per_patient) // 2, 0)
        end = start + self.max_slices_per_patient
        return range(start, end)

    def _build_slice_index(self):
        indices = []
        for pid, pdir in enumerate(self.patient_dirs):
            full_path = os.path.join(self.root_dir, pdir)
            modality_paths = {}
            missing_modalities = False
            for modality in self.modalities:
                modality_path = self._find_nifti(full_path, modality)
                if modality_path is None:
                    missing_modalities = True
                    break
                modality_paths[modality.lower()] = modality_path

            if missing_modalities or 't2' not in modality_paths:
                continue

            t2_image = load_nifti_image(modality_paths['t2'])
            volume_shape = t2_image.shape
            n_slices = volume_shape[self.slice_axis]
            if self.image_shape is None:
                self.image_shape = tuple(
                    dim for axis, dim in enumerate(volume_shape) if axis != self.slice_axis
                )

            self.patient_records[pdir] = {
                'paths': modality_paths,
                'shape': volume_shape,
            }

            selected_indices = list(self._select_slice_indices(n_slices))

            if self.preload_volumes:
                slice_cache = {}
                for modality_name, modality_path in modality_paths.items():
                    image = load_nifti_image(modality_path)
                    modality_cache = {}
                    for slice_idx in selected_indices:
                        slc = load_slice_from_image(image, self.slice_axis, slice_idx)
                        m = slc > 0
                        if m.sum() > 0:
                            slc[m] = (slc[m] - slc[m].mean()) / (slc[m].std() + 1e-8)
                        modality_cache[slice_idx] = slc
                    slice_cache[modality_name] = modality_cache
                self.patient_records[pdir]['slice_cache'] = slice_cache

            for s in selected_indices:
                indices.append((pid, pdir, s))
        return indices

    def set_undersample_mask(self, mask):
        self.undersample_mask = mask
        self._mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        if self.preload_volumes:
            self._build_sample_cache()

    def _build_sample_cache(self):
        self._sample_cache = []
        if self._mask_tensor is None:
            return

        for _, pdir, slice_idx in self.slices:
            patient_record = self.patient_records[pdir]
            volumes = {
                mod.lower(): patient_record['slice_cache'][mod.lower()][slice_idx]
                for mod in self.modalities
            }
            self._sample_cache.append(
                self._build_sample_result(pdir, slice_idx, volumes, self.undersample_mask)
            )

    def _build_sample_result(self, pdir, slice_idx, volumes, mask):
        t2 = volumes['t2']
        t2_tensor = torch.from_numpy(t2).unsqueeze(0)

        t1 = None
        for mod in self.modalities:
            if mod.lower() == 't1':
                t1 = volumes['t1']
                t1_tensor = torch.from_numpy(t1).unsqueeze(0)
                break

        kspace = np.fft.fftshift(np.fft.fft2(t2))
        kspace_us = kspace * mask
        t2_us = np.fft.ifft2(np.fft.ifftshift(kspace_us)).real
        t2_us_tensor = torch.from_numpy(t2_us.astype(np.float32)).unsqueeze(0)

        result = {
            'aliased': t2_us_tensor,
            'gt': t2_tensor,
            'mask': self._mask_tensor,
            'kspace_us': torch.from_numpy(
                np.stack([kspace_us.real, kspace_us.imag], axis=0)
            ).float(),
            'slice_idx': slice_idx,
            'patient': pdir,
        }

        if t1 is not None:
            result['t1_full'] = t1_tensor

        return result

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        if self._sample_cache is not None:
            return self._sample_cache[idx]

        pid, pdir, slice_idx = self.slices[idx]
        patient_record = self.patient_records[pdir]

        volumes = {}
        for mod in self.modalities:
            nii_path = patient_record['paths'].get(mod.lower())
            if nii_path is None:
                raise FileNotFoundError(f"Missing {mod} for {pdir}")
            if self.preload_volumes:
                slc = patient_record['slice_cache'][mod.lower()][slice_idx]
            else:
                image = load_nifti_image(nii_path)
                slc = load_slice_from_image(image, self.slice_axis, slice_idx)

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

        if self.undersample_mask is None or self.undersample_mask.shape != t2.shape:
            mask = generate_undersampling_mask_cpu(
                t2.shape, self.acceleration_factor, self.sigma
            )
            self._mask_tensor = None
        else:
            mask = self.undersample_mask

        if self._mask_tensor is None or self._mask_tensor.shape[-2:] != mask.shape:
            self._mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return self._build_sample_result(pdir, slice_idx, volumes, mask)


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
    data_cfg = config['data']
    dataset_path = data_cfg['dataset_path']
    modalities = config['data']['modalities']
    slice_axis = config['data']['slice_axis']
    seed = config['data']['seed']

    full_dataset = BraTSDataset(
        root_dir=dataset_path,
        modalities=modalities,
        slice_axis=slice_axis,
        max_patients=data_cfg.get('max_patients'),
        max_slices_per_patient=data_cfg.get('max_slices_per_patient'),
        preload_volumes=data_cfg.get('preload_volumes', False),
    )

    if len(full_dataset) == 0 or full_dataset.image_shape is None:
        raise RuntimeError(f"No valid BraTS slices found under {full_dataset.root_dir}")

    mask = generate_undersampling_mask_cpu(
        full_dataset.image_shape,
        config['task1']['acceleration_factor'],
        config['task1']['sigma']
    )
    full_dataset.set_undersample_mask(mask)

    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    patient_to_indices = {}
    for idx, (_, patient_id, _) in enumerate(full_dataset.slices):
        patient_to_indices.setdefault(patient_id, []).append(idx)

    patient_ids = list(patient_to_indices)
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    num_patients = len(patient_ids)
    val_patients = int(round(val_ratio * num_patients))
    test_patients = int(round(test_ratio * num_patients))

    if val_ratio > 0 and val_patients == 0 and num_patients >= 3:
        val_patients = 1
    if test_ratio > 0 and test_patients == 0 and num_patients - val_patients >= 2:
        test_patients = 1

    while num_patients - val_patients - test_patients <= 0:
        if val_patients >= test_patients and val_patients > 0:
            val_patients -= 1
        elif test_patients > 0:
            test_patients -= 1
        else:
            break

    train_patients = num_patients - val_patients - test_patients
    if train_patients <= 0:
        raise ValueError('Train/val/test split left no patients for training.')

    train_ids = set(patient_ids[:train_patients])
    val_ids = set(patient_ids[train_patients:train_patients + val_patients])
    test_ids = set(patient_ids[train_patients + val_patients:])

    train_indices = [idx for idx, (_, patient_id, _) in enumerate(full_dataset.slices) if patient_id in train_ids]
    val_indices = [idx for idx, (_, patient_id, _) in enumerate(full_dataset.slices) if patient_id in val_ids]
    test_indices = [idx for idx, (_, patient_id, _) in enumerate(full_dataset.slices) if patient_id in test_ids]

    if not train_indices or not val_indices or not test_indices:
        raise RuntimeError(
            'Train/val/test split produced an empty subset. '
            'Increase available patients or adjust split ratios.'
        )

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)

    return train_ds, val_ds, test_ds, mask
