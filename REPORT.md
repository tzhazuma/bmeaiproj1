# BraTS MRI Reconstruction Project

Generated from config `config/formal_train.yaml` on 2026-04-28 04:50:12.

## Group Information

- Course: BME AI Project 1
- Group: 第4组
- Chinese names: 唐志昊 2022533131
- Note: Metadata has been filled automatically. Review the final report before submission.

## AI Usage Declaration

This project used AI tools for code debugging, experiment automation, and report drafting. Manual review and result verification were performed before submission.

Presentation reminder: Remember to claim AI usage in the presentation slides as required by the assignment.

## Project Objective

This project reconstructs fully sampled T2 brain MRI slices from AF=5 undersampled k-space using the BraTS dataset. The assignment requires three tasks: undersampling simulation, a baseline reconstruction model, and a multi-modal unrolled model with data consistency.

## Dataset and Preprocessing

- Dataset path: `../bmeaidataset`
- Modalities used: T1, T2
- Slice axis: 2
- Intensity normalization: z-score on non-zero voxels
- Split strategy: patient-level train/validation/test
- Slice selection for this run: 16 central slices per patient
- Volume loading mode: preloaded in RAM to reduce I/O stalls
- Split counts: train=6992, validation=1504, test=1504

## Methods

### Task 1

A 2D random variable-density sampling mask with acceleration factor 5 is generated in k-space. Fully sampled T2 slices are transformed with FFT, masked, and reconstructed with inverse FFT to obtain aliased images.

### Task 2

Task 2 uses a U-Net baseline with base channels 24, depth 4, batch size 8, MSE loss, and learning rate 0.0005. `ReduceLROnPlateau` is used for learning rate decay.
During training, `channels_last`, pinned memory, non-blocking GPU transfers, TF32, and multi-worker prefetching are enabled to reduce GPU idle time.

### Task 3

Task 3 uses an unrolled U-Net with 2 cascades, data consistency layers, and multi-modal input consisting of aliased T2 plus fully sampled T1. The loss is `hybrid` with L1 weight 0.7.

## Division of Labor

- 唐志昊：完成项目整合、训练实验执行、结果核查、报告整理与提交。

## Quantitative Results

- Task 2 improved PSNR from 25.76 dB to 35.18 dB and SSIM from 0.3628 to 0.9271.
- Task 3 improved PSNR from 25.76 dB to 38.12 dB and SSIM from 0.3628 to 0.9660.
- Compared with Task 2, Task 3 changed PSNR by 2.93 dB and SSIM by 0.0389.

## Figures and Tables

### Dataset Split Chart

![Dataset Split Chart](outputs_formal/report_assets/dataset_split.png)

### Metric Comparison Chart

![Metric Comparison Chart](outputs_formal/report_assets/metric_comparison.png)

### Summary Tables

## Dataset Split

| Split | Slice Count |
| --- | ---: |
| Train | 6992 |
| Validation | 1504 |
| Test | 1504 |

## Hyperparameters

| Setting | Task 2 | Task 3 |
| --- | --- | --- |
| Input | Aliased T2 | Aliased T2 + full T1 |
| Backbone | U-Net | Unrolled U-Net + DC |
| Base channels | 24 | 16 |
| Depth | 4 | 3 |
| Batch size | 8 | 4 |
| Epochs | 10 | 8 |
| Learning rate | 0.0005 | 0.0001 |
| Loss | MSE | hybrid |

## Quantitative Results

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Aliased input | 25.76 | 0.3628 |
| Task 2 baseline | 35.18 | 0.9271 |
| Task 3 multi-modal | 38.12 | 0.9660 |

## Improvements Over Aliased Input

| Method | PSNR Gain (dB) | SSIM Gain |
| --- | ---: | ---: |
| Task 2 baseline | 9.43 | 0.5643 |
| Task 3 multi-modal | 12.36 | 0.6032 |

## Task 3 Over Task 2

| Comparison | Value |
| --- | ---: |
| PSNR gain | 2.93 dB |
| SSIM gain | 0.0389 |

### Worst-case Table

| Rank | Patient | Slice | PSNR Before | PSNR After | SSIM Before | SSIM After |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BraTS-GLI-00077-000 | 76 | 23.26 | 34.69 | 0.2865 | 0.9443 |
| 2 | BraTS-GLI-00077-000 | 82 | 23.14 | 34.74 | 0.2875 | 0.9502 |
| 3 | BraTS-GLI-00088-001 | 82 | 24.33 | 34.82 | 0.3160 | 0.9439 |
| 4 | BraTS-GLI-00077-000 | 78 | 23.23 | 34.82 | 0.2942 | 0.9481 |
| 5 | BraTS-GLI-00077-000 | 79 | 23.22 | 34.84 | 0.2921 | 0.9484 |

### Task 1 Visualization

![Task 1 Visualization](outputs_formal/task1/undersampling_visualization.png)

### Task 2 Loss Curve

![Task 2 Loss Curve](outputs_formal/task2/loss_curves.png)

### Task 2 Reconstructions

![Task 2 Reconstructions](outputs_formal/task2/reconstruction_results.png)

### Task 3 Loss Curve

![Task 3 Loss Curve](outputs_formal/task3/loss_curves.png)

### Task 3 Best Reconstructions

![Task 3 Best Reconstructions](outputs_formal/task3/best_reconstructions.png)

### Task 3 Error Analysis

![Task 3 Error Analysis](outputs_formal/task3/error_analysis.png)

## Error Analysis

The 5 worst-performing Task 3 cases are summarized below.

| Rank | Patient | Slice | PSNR After | SSIM After |
| --- | --- | ---: | ---: | ---: |
| 1 | BraTS-GLI-00077-000 | 76 | 34.69 | 0.9443 |
| 2 | BraTS-GLI-00077-000 | 82 | 34.74 | 0.9502 |
| 3 | BraTS-GLI-00088-001 | 82 | 34.82 | 0.9439 |
| 4 | BraTS-GLI-00077-000 | 78 | 34.82 | 0.9481 |
| 5 | BraTS-GLI-00077-000 | 79 | 34.84 | 0.9484 |

Potential improvements for these cases include stronger edge-preserving loss terms, more cascades if runtime allows, and targeted inspection of slices with complex tumor boundaries.

## Execution Status

- Current run status: Success
- Output directory: `outputs_formal`
- Total runtime: 24m 42s
- Last completed stage: assets
- Log file: `outputs_formal/logs/formal_pipeline.log`

## Discussion

The current results support the expected conclusion of the project: the baseline network substantially reduces aliasing artifacts, and the multi-modal unrolled model provides additional measurable improvement while using 16 central slices per patient across all available patients.

## Submission Checklist

- Code: prepared
- Report draft: `REPORT.md`
- LaTeX report: `REPORT.tex`
- PDF report: `REPORT.pdf`
- Presentation slides: still need to be prepared manually
