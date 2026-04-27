# BraTS MRI Reconstruction

Multi-contrast MRI Reconstruction from Undersampled k-space Data using the BraTS dataset.

## Project Structure

```
bmeaiproj1/
├── config/config.yaml     # All hyperparameters and paths
├── data/                  # Dataset loading and transforms
├── models/                # U-Net, Unrolled Network, Losses
├── tasks/                 # Task 1, 2, 3 scripts
├── utils/                 # Metrics, visualization, logging
├── outputs/               # Saved models, figures, logs
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

BraTS dataset in NIfTI format (`.nii.gz`). Place patient folders under the path specified in `config/config.yaml` (default: `Home/Downloads/dataset`).

Each patient folder should contain `T1.nii.gz` and `T2.nii.gz` files.

## Usage

### Task 1: Undersampling & Visualization
```bash
python tasks/task1_simulation.py
```
Generates variable-density undersampling mask (AF=5), creates aliased images, and visualizes results.

### Task 2: Baseline U-Net Reconstruction
```bash
python tasks/task2_train.py    # Train U-Net baseline
python tasks/task2_eval.py     # Evaluate PSNR/SSIM
```

### Task 3: Multi-modal Advanced Reconstruction
```bash
python tasks/task3_train.py    # Train unrolled network with DC layers
python tasks/task3_eval.py     # Evaluate + error analysis
```

### Run All Tasks
```bash
python tasks/task1_simulation.py
python tasks/task2_train.py && python tasks/task2_eval.py
python tasks/task3_train.py && python tasks/task3_eval.py
```

## Configuration

Edit `config/config.yaml` to adjust:
- Dataset path (`data.dataset_path`)
- Model hyperparameters (channels, depth, batch size, epochs)
- Learning rate and scheduler settings
- Loss function type (l1, l2, hybrid)

## Output

All results are saved to `outputs/`:
- `task1/`: Undersampling visualization, mask
- `task2/`: Trained U-Net model, loss curves, evaluation metrics, reconstruction examples
- `task3/`: Trained unrolled model, error analysis, best/worst reconstructions
