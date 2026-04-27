# BraTS MRI Reconstruction Project

Generated from config `config/formal_train.yaml` on 2026-04-28 01:22:47.

## Group Information

- Course: BME AI Project 1
- Group: <fill group name>
- Chinese names: <fill Chinese names>
- Note: Replace placeholders before final PDF export.

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
- Split counts: train=3496, validation=752, test=752

## Methods

### Task 1

A 2D random variable-density sampling mask with acceleration factor 5 is generated in k-space. Fully sampled T2 slices are transformed with FFT, masked, and reconstructed with inverse FFT to obtain aliased images.

### Task 2

Task 2 uses a U-Net baseline with base channels 24, depth 4, batch size 8, MSE loss, and learning rate 0.0005. `ReduceLROnPlateau` is used for learning rate decay.

### Task 3

Task 3 uses an unrolled U-Net with 2 cascades, data consistency layers, and multi-modal input consisting of aliased T2 plus fully sampled T1. The loss is `hybrid` with L1 weight 0.7.

## Division of Labor

- <fill division of labor>

## Quantitative Results

- Task 2 results: pending formal execution.
- Task 3 results: pending formal execution.

## Figures and Tables

- Dataset split chart: `outputs_formal/report_assets/dataset_split.png`
- Metric comparison chart: pending generation
- Summary tables: `outputs_formal/report_assets/summary_tables.md`
- Worst-case table: pending generation
- Task 1 visualization: pending generation
- Task 2 loss curve: pending generation
- Task 2 reconstructions: pending generation
- Task 3 loss curve: pending generation
- Task 3 best reconstructions: pending generation
- Task 3 error analysis: pending generation

## Error Analysis

Task 3 worst-case analysis is pending because `worst_cases.json` is not available yet.

## Execution Status

- Current run status: Not started
- Output directory: `outputs_formal`
- Total runtime: N/A

## Discussion

Formal quantitative conclusions are pending because the final training run has not completed yet. The automation scripts are prepared so this report will be refreshed automatically after the formal run finishes.

## Submission Checklist

- Code: prepared
- Report draft: `REPORT.md`
- Presentation slides: still need to be prepared manually
- Final PDF export: still needs manual export after placeholders are reviewed
