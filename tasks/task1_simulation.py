"""
Task 1: Synthesis of Undersampled MRI and Visualization
Generates 2D random variable-density undersampling mask (AF=5),
applies it to T2 images in k-space, reconstructs aliased images,
and visualizes results side-by-side.
"""
import os
import sys
import yaml
import numpy as np
import nibabel as nib
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.transforms import generate_undersampling_mask, undersample_kspace
from data.dataset import find_modality_path, resolve_dataset_path
from utils.visualize import plot_undersampling


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


def main():
    config = load_config()
    cfg = config['task1']
    output_dir = os.path.join(config['output']['dir'], 'task1')
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = resolve_dataset_path(config['data']['dataset_path'])
    slice_axis = config['data']['slice_axis']

    patient_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    if not patient_dirs:
        print(f"No patient directories found in {dataset_path}")
        print("Please ensure dataset is downloaded to the correct path.")
        print("You can also modify dataset_path in config/config.yaml")
        sys.exit(1)

    print(f"Found {len(patient_dirs)} patient directories")

    # Load first patient's T2 for demonstration
    demo_patient = patient_dirs[0]
    patient_path = os.path.join(dataset_path, demo_patient)

    t2_path = find_modality_path(patient_path, 't2w')

    if t2_path is None:
        print(f"No T2 file found for {demo_patient}")
        sys.exit(1)

    print(f"Loading T2: {t2_path}")
    t2_vol = nib.load(t2_path).get_fdata().astype(np.float32)

    n_slices = t2_vol.shape[slice_axis]
    middle = n_slices // 2
    num_slices_to_show = max(int(cfg.get('num_slices_to_show', 4)), 1)
    start = max(middle - 8, 0)
    end = min(middle + 8, n_slices - 1)
    indices = np.linspace(start, end, num_slices_to_show, dtype=int).tolist()

    h, w = t2_vol.shape[0], t2_vol.shape[1]
    mask = generate_undersampling_mask(
        (h, w), cfg['acceleration_factor'], cfg['sigma']
    )

    full_slices = []
    aliased_slices = []

    for idx in tqdm(indices, desc="Processing slices"):
        if idx < 0 or idx >= n_slices:
            continue
        slc = np.take(t2_vol, idx, axis=slice_axis)
        m = slc > 0
        if m.sum() > 0:
            slc[m] = (slc[m] - slc[m].mean()) / (slc[m].std() + 1e-8)

        _, aliased = undersample_kspace(slc, mask)
        full_slices.append(slc)
        aliased_slices.append(aliased)

    save_path = os.path.join(output_dir, 'undersampling_visualization.png')
    plot_undersampling(
        mask, full_slices, aliased_slices, save_path,
        slice_names=[f"Slice {i}" for i in indices[:len(full_slices)]]
    )
    print(f"Saved visualization to {save_path}")

    mask_save = os.path.join(output_dir, 'mask.npy')
    np.save(mask_save, mask)
    print(f"Saved mask to {mask_save}")

    aliased_save = os.path.join(output_dir, 'aliased_examples.npy')
    np.save(aliased_save, np.stack(aliased_slices))
    print(f"Saved aliased examples to {aliased_save}")

    print("\nTask 1 completed successfully!")


if __name__ == '__main__':
    main()
