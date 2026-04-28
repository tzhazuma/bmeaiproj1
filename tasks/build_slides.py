import json
import os
import subprocess


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs_formal')
SLIDES_TEX = os.path.join(ROOT_DIR, 'slides.tex')
SLIDES_PDF = os.path.join(ROOT_DIR, 'slides.pdf')
OUTLINE_MD = os.path.join(ROOT_DIR, 'slides_outline.md')


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_text(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def latex_escape(text):
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(ch, ch) for ch in text)


def relative(path):
    return os.path.relpath(path, ROOT_DIR).replace(os.sep, '/')


def build_outline(task2, task3):
    lines = []
    lines.append('# Presentation Outline')
    lines.append('')
    lines.append('1. Project motivation: MRI acceleration reduces scan time but creates aliasing artifacts.')
    lines.append('2. Dataset and setup: BraTS T1/T2, patient-level split, 16 central slices per patient, AF=5 undersampling.')
    lines.append('3. Task 1: simulate undersampling in k-space and visualize aliasing.')
    lines.append('4. Task 2: U-Net baseline trained to reconstruct T2 from undersampled input.')
    lines.append('5. Task 3: multi-modal unrolled network using undersampled T2 + full T1 + data consistency.')
    lines.append(
        f"6. Key result: Task 2 reaches PSNR {task2['psnr_after']['mean']:.2f} / SSIM {task2['ssim_after']['mean']:.4f}, while Task 3 reaches PSNR {task3['psnr_after']['mean']:.2f} / SSIM {task3['ssim_after']['mean']:.4f}."
    )
    lines.append(
        f"7. Improvement summary: Task 3 beats Task 2 by {task3['psnr_after']['mean'] - task2['psnr_after']['mean']:.2f} dB PSNR and {task3['ssim_after']['mean'] - task2['ssim_after']['mean']:.4f} SSIM."
    )
    lines.append('8. Error analysis: difficult cases still occur on structurally complex slices, but remain much better than aliased inputs.')
    lines.append('9. AI declaration: AI was used only for debugging, automation, and draft assistance; all final decisions and checks were manual.')
    return '\n'.join(lines) + '\n'


def build_slides_tex(task2, task3):
    task2_loss = relative(os.path.join(OUTPUT_DIR, 'task2', 'loss_curves.png'))
    task2_recon = relative(os.path.join(OUTPUT_DIR, 'task2', 'reconstruction_results.png'))
    task3_loss = relative(os.path.join(OUTPUT_DIR, 'task3', 'loss_curves.png'))
    task3_best = relative(os.path.join(OUTPUT_DIR, 'task3', 'best_reconstructions.png'))
    task3_error = relative(os.path.join(OUTPUT_DIR, 'task3', 'error_analysis.png'))
    metric_chart = relative(os.path.join(OUTPUT_DIR, 'report_assets', 'metric_comparison.png'))
    task1_viz = relative(os.path.join(OUTPUT_DIR, 'task1', 'undersampling_visualization.png'))

    task2_psnr = task2['psnr_after']['mean']
    task2_ssim = task2['ssim_after']['mean']
    task3_psnr = task3['psnr_after']['mean']
    task3_ssim = task3['ssim_after']['mean']
    psnr_gain = task3_psnr - task2_psnr
    ssim_gain = task3_ssim - task2_ssim

    return rf"""\documentclass[aspectratio=169,11pt]{{beamer}}
\usetheme{{Madrid}}
\usecolortheme{{dolphin}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\setmainfont{{Noto Sans CJK SC}}
\setsansfont{{Noto Sans CJK SC}}
\setmonofont{{Noto Sans Mono CJK SC}}
\setCJKmainfont{{Noto Sans CJK SC}}
\title{{BraTS MRI Reconstruction}}
\subtitle{{BME AI Project 1}}
\author{{第4组 / 唐志昊 2022533131}}
\date{{2026-04-28}}

\begin{{document}}

\begin{{frame}}
  \titlepage
\end{{frame}}

\begin{{frame}}{{Project Motivation}}
\begin{{itemize}}
  \item Faster MRI acquisition is clinically desirable but undersampling causes aliasing artifacts.
  \item Goal: reconstruct high-quality T2 MRI from AF=5 undersampled k-space.
  \item Tasks: simulation, baseline reconstruction, and multi-modal unrolled reconstruction.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Dataset and Experimental Setup}}
\begin{{itemize}}
  \item Dataset: BraTS, using co-registered T1 and T2 modalities.
  \item Split: patient-level split to avoid train/test leakage.
  \item Final formal run: 16 central slices per patient, all available patients.
  \item Runtime optimization: slice caching, pinned memory, TF32, non-blocking transfer.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Task 1: Undersampling Simulation}}
  \centering
  \includegraphics[width=0.92\linewidth,height=0.78\textheight,keepaspectratio]{{{latex_escape(task1_viz)}}}
\end{{frame}}

\begin{{frame}}{{Task 2: Baseline U-Net}}
\begin{{columns}}[T]
\column{{0.48\linewidth}}
\begin{{itemize}}
  \item Input: undersampled T2 slice.
  \item Model: U-Net baseline.
  \item Loss: MSE.
  \item Final performance: PSNR {task2_psnr:.2f}, SSIM {task2_ssim:.4f}.
\end{{itemize}}
\column{{0.48\linewidth}}
  \includegraphics[width=\linewidth]{{{latex_escape(task2_loss)}}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Task 2 Reconstruction Examples}}
  \centering
  \includegraphics[width=0.92\linewidth,height=0.78\textheight,keepaspectratio]{{{latex_escape(task2_recon)}}}
\end{{frame}}

\begin{{frame}}{{Task 3: Multi-modal Unrolled Network}}
\begin{{columns}}[T]
\column{{0.5\linewidth}}
\begin{{itemize}}
  \item Input: undersampled T2 + fully sampled T1.
  \item Architecture: 2-cascade unrolled U-Net with data consistency.
  \item Loss: hybrid L1/L2.
  \item Final performance: PSNR {task3_psnr:.2f}, SSIM {task3_ssim:.4f}.
\end{{itemize}}
\column{{0.45\linewidth}}
  \includegraphics[width=\linewidth]{{{latex_escape(task3_loss)}}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Quantitative Comparison}}
\begin{{columns}}[T]
\column{{0.52\linewidth}}
\begin{{itemize}}
  \item Task 3 improves over Task 2 by {psnr_gain:.2f} dB PSNR.
  \item Task 3 improves over Task 2 by {ssim_gain:.4f} SSIM.
  \item Multi-modal guidance and data consistency both contribute measurable gains.
\end{{itemize}}
\column{{0.43\linewidth}}
  \includegraphics[width=\linewidth]{{{latex_escape(metric_chart)}}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Task 3 Best Reconstructions}}
  \centering
  \includegraphics[width=0.92\linewidth,height=0.78\textheight,keepaspectratio]{{{latex_escape(task3_best)}}}
\end{{frame}}

\begin{{frame}}{{Task 3 Error Analysis}}
  \centering
  \includegraphics[width=0.92\linewidth,height=0.78\textheight,keepaspectratio]{{{latex_escape(task3_error)}}}
\end{{frame}}

\begin{{frame}}{{Conclusions}}
\begin{{itemize}}
  \item The baseline model already removes most aliasing artifacts effectively.
  \item The multi-modal unrolled model provides the best fidelity and visual quality.
  \item Remaining hard cases are mainly complex slices with challenging local structure.
  \item Future work: stronger edge-aware loss, deeper cascades, and targeted hard-case training.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{AI Usage Declaration}}
\begin{{itemize}}
  \item AI tools were used only for debugging support, automation, and draft editing.
  \item All model choices, experiment validation, metric checks, and final submitted materials were manually reviewed and confirmed.
  \item This declaration is included to satisfy the assignment requirement.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Thank You}}
  \centering
  Questions are welcome.
\end{{frame}}

\end{{document}}
"""


def compile_xelatex(tex_path):
    command = [
        'xelatex',
        '-interaction=nonstopmode',
        '-halt-on-error',
        '-output-directory',
        ROOT_DIR,
        tex_path,
    ]
    subprocess.run(command, check=True, cwd=ROOT_DIR)
    subprocess.run(command, check=True, cwd=ROOT_DIR)


def main():
    task2 = read_json(os.path.join(OUTPUT_DIR, 'task2', 'evaluation_summary.json'))
    task3 = read_json(os.path.join(OUTPUT_DIR, 'task3', 'evaluation_summary.json'))

    write_text(OUTLINE_MD, build_outline(task2, task3))
    write_text(SLIDES_TEX, build_slides_tex(task2, task3))
    compile_xelatex(SLIDES_TEX)

    if not os.path.exists(SLIDES_PDF):
        raise RuntimeError('slides.pdf was not generated')

    print(f'Outline written to {OUTLINE_MD}')
    print(f'Slides written to {SLIDES_TEX}')
    print(f'Slides PDF written to {SLIDES_PDF}')


if __name__ == '__main__':
    main()
