# Presentation Outline

1. Motivation: accelerated MRI reduces scan time but introduces aliasing artifacts under k-space undersampling.
2. Dataset and protocol: BraTS T1/T2, patient-level split, 16 central slices per patient, AF=5 undersampling.
3. Task 1: simulate k-space undersampling and visualize the resulting aliasing pattern.
4. Task 2: train a U-Net baseline to reconstruct T2 from undersampled input.
5. Task 3: use a multi-modal unrolled network with undersampled T2, fully sampled T1, and data consistency.
6. Main quantitative result: Task 2 reaches PSNR 35.18 / SSIM 0.9271, while Task 3 reaches PSNR 38.12 / SSIM 0.9660.
7. Improvement summary: Task 3 exceeds Task 2 by 2.93 dB PSNR and 0.0389 SSIM.
8. Error analysis: the most difficult cases occur on structurally complex slices, but still remain much better than the aliased inputs.
9. AI declaration: AI was used only for debugging, automation, and draft assistance; all final decisions and checks were manual.
