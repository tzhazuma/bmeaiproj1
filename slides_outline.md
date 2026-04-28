# Presentation Outline

1. Project motivation: MRI acceleration reduces scan time but creates aliasing artifacts.
2. Dataset and setup: BraTS T1/T2, patient-level split, 16 central slices per patient, AF=5 undersampling.
3. Task 1: simulate undersampling in k-space and visualize aliasing.
4. Task 2: U-Net baseline trained to reconstruct T2 from undersampled input.
5. Task 3: multi-modal unrolled network using undersampled T2 + full T1 + data consistency.
6. Key result: Task 2 reaches PSNR 35.18 / SSIM 0.9271, while Task 3 reaches PSNR 38.12 / SSIM 0.9660.
7. Improvement summary: Task 3 beats Task 2 by 2.93 dB PSNR and 0.0389 SSIM.
8. Error analysis: difficult cases still occur on structurally complex slices, but remain much better than aliased inputs.
9. AI declaration: AI was used only for debugging, automation, and draft assistance; all final decisions and checks were manual.
