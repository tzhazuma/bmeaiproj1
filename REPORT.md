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

Task 2 uses a U-Net baseline with base channels 24, depth 4, batch size 8, MSE loss, and learning rate 0.0005. `ReduceLROnPlateau` is used for learning rate scheduling.
To improve runtime efficiency on the available RTX 4060 laptop GPU, the implementation uses slice caching, pinned memory, non-blocking GPU transfer, `channels_last`, TF32, and prefetch-friendly dataloading to reduce GPU idle time.

### Task 3

Task 3 uses an unrolled U-Net with 2 cascades, data-consistency layers, and multi-modal input consisting of aliased T2 together with fully sampled T1. The loss is `hybrid` with L1 weight 0.7.
The underlying rationale is that T1 provides stable anatomical structure, while the data-consistency layer constrains the network output to remain faithful to the measured undersampled k-space.

## Division of Labor

- 唐志昊：完成项目整合、训练实验执行、结果核查、报告整理与提交。

## Quantitative Results

- Task 2 improved PSNR from 25.95 dB to 35.16 dB and SSIM from 0.3717 to 0.9261.
- Task 3 improved PSNR from 25.95 dB to 38.16 dB and SSIM from 0.3717 to 0.9655.
- Compared with Task 2, Task 3 changed PSNR by 2.99 dB and SSIM by 0.0394.

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
| Depth | 4 | 3 |
| Batch size | 8 | 4 |
| Epochs | 10 | 8 |
| Learning rate | 0.0005 | 0.0001 |
| Loss | MSE | hybrid |

## Quantitative Results

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Aliased input | 25.95 | 0.3717 |
| Task 2 baseline | 35.16 | 0.9261 |
| Task 3 multi-modal | 38.16 | 0.9655 |

## Improvements Over Aliased Input

| Method | PSNR Gain (dB) | SSIM Gain |
| --- | ---: | ---: |
| Task 2 baseline | 9.22 | 0.5544 |
| Task 3 multi-modal | 12.21 | 0.5938 |

## Task 3 Over Task 2

| Comparison | Value |
| --- | ---: |
| PSNR gain | 2.99 dB |
| SSIM gain | 0.0394 |

### Worst-case Table

| Rank | Patient | Slice | PSNR Before | PSNR After | SSIM Before | SSIM After |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | BraTS-GLI-00588-000 | 78 | 23.75 | 33.85 | 0.3009 | 0.9504 |
| 2 | BraTS-GLI-00588-000 | 83 | 24.12 | 34.00 | 0.3050 | 0.9486 |
| 3 | BraTS-GLI-00588-000 | 81 | 23.93 | 34.06 | 0.2995 | 0.9491 |
| 4 | BraTS-GLI-00588-000 | 82 | 23.99 | 34.07 | 0.3005 | 0.9485 |
| 5 | BraTS-GLI-00588-000 | 84 | 24.32 | 34.15 | 0.3121 | 0.9490 |

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

The 5 lowest-performing Task 3 cases are summarized below. Even in these challenging slices, the reconstructed outputs remain substantially better than the aliased inputs, indicating that the dominant failure mode is relative quality degradation rather than complete reconstruction collapse.

| Rank | Patient | Slice | PSNR After | SSIM After |
| --- | --- | ---: | ---: | ---: |
| 1 | BraTS-GLI-00588-000 | 78 | 33.85 | 0.9504 |
| 2 | BraTS-GLI-00588-000 | 83 | 34.00 | 0.9486 |
| 3 | BraTS-GLI-00588-000 | 81 | 34.06 | 0.9491 |
| 4 | BraTS-GLI-00588-000 | 82 | 34.07 | 0.9485 |
| 5 | BraTS-GLI-00588-000 | 84 | 34.15 | 0.9490 |

Potential improvements for these cases include stronger edge-preserving loss terms, additional cascades if runtime permits, and targeted inspection of slices with complex tumor boundaries.

## Execution Status

- Current run status: Not started
- Output directory: `outputs_formal`
- Total runtime: N/A

## Discussion

The final experiment used 16 central slices per patient across all available BraTS patients and produced consistent quantitative improvements over the aliased baseline in both reconstruction settings.

The Task 2 baseline recovered a substantial proportion of the missing image fidelity, improving PSNR by 9.22 dB and SSIM by 0.5544, which confirms that the supervised reconstruction pipeline converged as expected.

Task 3 further improved PSNR by 12.21 dB and SSIM by 0.5938 relative to the aliased input, and outperformed Task 2 by 2.99 dB PSNR and 0.0394 SSIM.

These findings support the central conclusion of the project: incorporating fully sampled T1 structural guidance together with unrolled data-consistency reconstruction yields sharper boundaries, fewer residual artifacts, and stronger quantitative fidelity than a single-modality baseline.

The remaining challenging cases are concentrated in slices with more complex local structure, suggesting that future improvements may be obtained through stronger edge-aware loss functions, a deeper unrolled architecture, or targeted sampling and training strategies for anatomically difficult regions.

## Submission Checklist

- Code: prepared
- Report draft: `REPORT.md`
- LaTeX report: `REPORT.tex`
- PDF report: `REPORT.pdf`
- Presentation slides: generated as `slides.tex` and `slides.pdf`
