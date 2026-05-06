# BraTS MRI Reconstruction Project

Generated from config `config/formal_train.yaml` on 2026-05-06 04:45:11.

## Group Information

- Course: BME AI Project 1
- Group: 第4组
- Chinese names: 唐志昊 2022533131
- Note: This report was generated from the final verified experiment outputs and then reviewed before submission.

## AI Usage Declaration

AI tools were used only as development assistance for code debugging, experiment orchestration, and draft editing. All implementation decisions, experimental validation, result interpretation, and final submitted materials were manually reviewed and confirmed by the student before submission.

Presentation reminder: AI assistance is explicitly declared in the generated presentation slides, in accordance with the assignment requirement.

## Project Objective

The objective of this project is to reconstruct high-fidelity T2-weighted brain MRI slices from AF=5 undersampled k-space using the BraTS dataset. The study is organized into three stages: undersampling simulation, a supervised baseline reconstruction model, and a multi-modal unrolled reconstruction model with explicit data consistency.

## Dataset and Preprocessing

- Dataset path: `/mnt/d/brats2023`
- Modalities used: T1, T2
- Slice axis: 2
- Intensity normalization: z-score on non-zero voxels
- Split strategy: patient-level train/validation/test
- Slice selection for this run: 16 central slices per patient
- Data loading strategy: slice-level preloading in RAM to reduce I/O stalls
- Split counts: train=6272, validation=1344, test=1344
- Data split unit: patient-level split to avoid leakage across adjacent slices from the same subject

## Methods

### Task 1

A 2D random variable-density sampling mask with acceleration factor 5 is generated in k-space. Fully sampled T2 slices are transformed with FFT, masked, and reconstructed with inverse FFT to obtain aliased images. This stage establishes the artifact characteristics that must subsequently be removed by the learning-based reconstruction models.

### Task 2

Task 2 uses a U-Net baseline with base channels 24, depth 6, batch size 8, MSE loss, and initial learning rate 0.001. `ReduceLROnPlateau` is used for learning rate scheduling, with training configured for up to 150 epochs and early stopping (patience 15).
To improve runtime efficiency on the available RTX 4060/5070Ti laptop GPU, the implementation uses slice caching, pinned memory, non-blocking GPU transfer, `channels_last`, TF32, and prefetch-friendly dataloading to reduce GPU idle time.

### Task 3

Task 3 uses an unrolled U-Net with 5 cascades, data-consistency layers, and multi-modal input consisting of aliased T2 together with fully sampled T1. The loss is `hybrid` with L1 weight 0.5 (same initial LR 0.001 and `ReduceLROnPlateau` schedule as in Task 2; up to 150 epochs with early stopping).
The underlying rationale is that T1 provides stable anatomical structure, while the data-consistency layer constrains the network output to remain faithful to the measured undersampled k-space.

## Division of Labor

- 唐志昊：完成项目整合、训练实验执行、结果核查、报告整理与提交。

## Quantitative Results

- Task 2 improved mean PSNR from 25.76 dB to 36.11 dB and mean SSIM from 0.3628 to 0.9404.
- Task 3 improved mean PSNR from 25.76 dB to 47.08 dB and mean SSIM from 0.3628 to 0.9944.
- Compared with Task 2, Task 3 improved mean PSNR by 10.97 dB and mean SSIM by 0.0540.

## Figures and Tables

### Dataset Split Chart

![Dataset Split Chart](outputs_formal/report_assets/dataset_split.png)

### Metric Comparison Chart

![Metric Comparison Chart](outputs_formal/report_assets/metric_comparison.png)

### Summary Tables

## Dataset Split

| Split | Slice Count |
| --- | ---: |
| Train | 6272 |
| Validation | 1344 |
| Test | 1344 |

## Hyperparameters

| Setting | Task 2 | Task 3 |
| --- | --- | --- |
| Input | Aliased T2 | Aliased T2 + full T1 |
| Backbone | U-Net | Unrolled U-Net + DC |
| Base channels | 24 | 16 |
| Depth | 6 | 6 |
| Cascades | — | 5 |
| Batch size | 8 | 8 |
| Max epochs | 150 (early stop) | 150 (early stop) |
| Initial learning rate | 0.001 | 0.001 |
| Loss | MSE | hybrid (L1 weight 0.5) |

## Quantitative Results

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Aliased input | 25.76 | 0.3628 |
| Task 2 baseline | 36.11 | 0.9404 |
| Task 3 multi-modal | 47.08 | 0.9944 |

## Improvements Over Aliased Input

| Method | PSNR Gain (dB) | SSIM Gain |
| --- | ---: | ---: |
| Task 2 baseline | 10.35 | 0.5776 |
| Task 3 multi-modal | 21.32 | 0.6315 |

## Task 3 Over Task 2

| Comparison | Value |
| --- | ---: |
| PSNR gain | 10.97 dB |
| SSIM gain | 0.0540 |

### Worst-case Table

Five lowest-PSNR Task 3 test slices (per evaluation log).

| Rank | Patient | Slice | PSNR Before | PSNR After | SSIM Before | SSIM After |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BraTS-GLI-00429-000 | 74 | 25.00 | 43.10 | 0.3065 | 0.9872 |
| 2 | BraTS-GLI-00429-000 | 77 | 25.07 | 43.14 | 0.3255 | 0.9867 |
| 3 | BraTS-GLI-00446-000 | 73 | 25.26 | 43.17 | 0.3505 | 0.9920 |
| 4 | BraTS-GLI-00429-000 | 73 | 25.08 | 43.17 | 0.3102 | 0.9881 |
| 5 | BraTS-GLI-00734-000 | 69 | 25.15 | 43.19 | 0.3509 | 0.9891 |

### Task 1 Visualization

Pending generation.

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

The five lowest-PSNR Task 3 test slices are summarized below. Even these relative worst cases reach about 43.1 dB PSNR and SSIM near 0.99, with large gains over the folded inputs (about +18 dB PSNR). The dominant pattern is mild performance variation across slices rather than reconstruction failure.

| Rank | Patient | Slice | PSNR After | SSIM After |
| --- | --- | ---: | ---: | ---: |
| 1 | BraTS-GLI-00429-000 | 74 | 43.10 | 0.9872 |
| 2 | BraTS-GLI-00429-000 | 77 | 43.14 | 0.9867 |
| 3 | BraTS-GLI-00446-000 | 73 | 43.17 | 0.9920 |
| 4 | BraTS-GLI-00429-000 | 73 | 43.17 | 0.9881 |
| 5 | BraTS-GLI-00734-000 | 69 | 43.19 | 0.9891 |

Potential improvements for the hardest slices include stronger edge-preserving loss terms, additional cascades if runtime permits, and targeted inspection of regions with complex tumor boundaries.

## Execution Status

- Current run status: Task 2 and Task 3 evaluation artifacts present under `outputs_formal`
- Output directory: `outputs_formal`
- Total runtime: N/A (not logged in this report)

## Discussion

The final experiment used 16 central slices per patient across all available BraTS patients and produced consistent quantitative improvements over the aliased baseline in both reconstruction settings.

The Task 2 baseline recovered a substantial proportion of the missing image fidelity, improving mean PSNR by 10.35 dB and mean SSIM by 0.5776 over the aliased input, which confirms that the supervised reconstruction pipeline converged as expected.

Task 3 further improved mean PSNR by 21.32 dB and mean SSIM by 0.6315 relative to the aliased input, and outperformed Task 2 by 10.97 dB PSNR and 0.0540 SSIM on the same test split.

These findings support the central conclusion of the project: incorporating fully sampled T1 structural guidance together with unrolled data-consistency reconstruction yields sharper boundaries, fewer residual artifacts, and stronger quantitative fidelity than a single-modality baseline.

The remaining challenging cases are concentrated in slices with more complex local structure, suggesting that future improvements may be obtained through stronger edge-aware loss functions, a deeper unrolled architecture, or targeted sampling and training strategies for anatomically difficult regions.

## Submission Checklist

- Code: prepared
- Report draft: `REPORT.md`
- LaTeX report: `REPORT.tex`
- PDF report: `REPORT.pdf`
- Presentation slides: generated as `slides.tex` and `slides.pdf`
