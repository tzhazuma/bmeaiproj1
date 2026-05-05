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

### Using uv (recommended)

```bash
uv venv .venv
uv pip install -r requirements.txt
```

### Using pip

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### GPU Support

This project supports both NVIDIA (CUDA) and AMD (ROCm) GPUs:
- **NVIDIA**: Install `torch>=2.0` with CUDA support
- **AMD**: Install `torch` with ROCm support (e.g., `torch==2.11.0+rocm7.2`)

Verify GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Dataset

### Option 1: Automatic Download (Recommended)

```bash
python scripts/download_brats.py --output /mnt/d/brats2023
```

This script uses the Kaggle API with multi-threaded download (aria2c) and automatic extraction.

**Prerequisites:**
1. Get your Kaggle API token from [kaggle.com/settings](https://www.kaggle.com/settings/account)
2. Place it at `~/.kaggle/kaggle.json`

### Option 2: Manual Download

Download [BraTS2023 Part 1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1) and extract to your desired path (e.g., `/mnt/d/brats2023`).

### Dataset Format

BraTS dataset in NIfTI format (`.nii` or `.nii.gz`). The loader looks for BraTS-style modality names such as `*-t1n.nii` and `*-t2w.nii`, and also supports plain `t1` / `t2` filenames.

Edit `config/config.yaml` to set your dataset path:
```yaml
data:
  dataset_path: "/mnt/d/brats2023"
```

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

### Quick Smoke Test
```bash
BMEAI_CONFIG=config/smoke_test.yaml python tasks/task1_simulation.py
BMEAI_CONFIG=config/smoke_test.yaml python tasks/task2_train.py && BMEAI_CONFIG=config/smoke_test.yaml python tasks/task2_eval.py
BMEAI_CONFIG=config/smoke_test.yaml python tasks/task3_train.py && BMEAI_CONFIG=config/smoke_test.yaml python tasks/task3_eval.py
```

### Medium Validation
```bash
BMEAI_CONFIG=config/medium_test.yaml python tasks/task2_train.py && BMEAI_CONFIG=config/medium_test.yaml python tasks/task2_eval.py
BMEAI_CONFIG=config/medium_test.yaml python tasks/task3_train.py && BMEAI_CONFIG=config/medium_test.yaml python tasks/task3_eval.py
```

### Formal Training
```bash
BMEAI_CONFIG=config/formal_train.yaml python tasks/task2_train.py && BMEAI_CONFIG=config/formal_train.yaml python tasks/task2_eval.py
BMEAI_CONFIG=config/formal_train.yaml python tasks/task3_train.py && BMEAI_CONFIG=config/formal_train.yaml python tasks/task3_eval.py
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
