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
    lines.append('1. Motivation and setup: accelerated MRI, BraTS T1/T2, patient-level split, 16 central slices per patient, AF=5 undersampling.')
    lines.append('2. Task 1: simulate k-space undersampling and visualize the resulting aliasing pattern.')
    lines.append('3. Task 2: train a U-Net baseline to reconstruct T2 from undersampled input.')
    lines.append('4. Task 3: use a multi-modal unrolled network with undersampled T2, fully sampled T1, and data consistency.')
    lines.append(
        f"5. Main quantitative result: Task 2 reaches PSNR {task2['psnr_after']['mean']:.2f} / SSIM {task2['ssim_after']['mean']:.4f}, while Task 3 reaches PSNR {task3['psnr_after']['mean']:.2f} / SSIM {task3['ssim_after']['mean']:.4f}."
    )
    lines.append(
        f"6. Improvement summary: Task 3 exceeds Task 2 by {task3['psnr_after']['mean'] - task2['psnr_after']['mean']:.2f} dB PSNR and {task3['ssim_after']['mean'] - task2['ssim_after']['mean']:.4f} SSIM."
    )
    lines.append('7. Error analysis: the most difficult cases occur on structurally complex slices, but still remain much better than the aliased inputs.')
    lines.append('8. Conclusions and AI declaration: summarize the gain from multi-modal reconstruction and state that AI was used only for debugging, automation, and draft assistance.')
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
\setCJKsansfont{{Noto Sans CJK SC}}
\setCJKmonofont{{Noto Sans Mono CJK SC}}
\title{{BraTS MRI Reconstruction}}
\subtitle{{Multi-contrast MRI Reconstruction from Undersampled Data}}
\author{{第4组 / 唐志昊 2022533131}}
\date{{2026-04-28}}

\begin{{document}}

\begin{{frame}}
  \titlepage
\end{{frame}}

\begin{{frame}}{{Motivation and Experimental Setup}}
\begin{{itemize}}
  \item Accelerated MRI acquisition is clinically desirable, but k-space undersampling introduces aliasing artifacts.
  \item Objective: reconstruct high-fidelity T2 MRI slices from AF=5 undersampled data.
  \item Dataset: BraTS, using co-registered T1 and T2 modalities.
  \item Split strategy: patient-level partition to prevent train/test leakage.
  \item Final formal run: 16 central slices per patient across all available patients.
  \item Runtime optimization: slice caching, pinned memory, TF32, and non-blocking GPU transfer.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Methods Overview}}
\begin{{columns}}[T]
\column{{0.48\linewidth}}
\textbf{{Task 1: Undersampling Simulation}}\\
\includegraphics[width=\linewidth,height=0.34\textheight,keepaspectratio]{{{latex_escape(task1_viz)}}}
\\[0.3em]
\textbf{{Task 2: Baseline U-Net}}\\
\begin{{itemize}}
  \item Input: undersampled T2 slice.
  \item Model: U-Net baseline.
  \item Loss: MSE.
\end{{itemize}}
\column{{0.48\linewidth}}
\textbf{{Task 3: Multi-modal Unrolled Model}}\\
\begin{{itemize}}
  \item Input: undersampled T2 + fully sampled T1.
  \item Model: 2-cascade unrolled U-Net with data consistency.
  \item Loss: hybrid L1/L2.
\end{{itemize}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Training Behaviour}}
\begin{{columns}}[T]
\column{{0.48\linewidth}}
  \centering
  \includegraphics[width=\linewidth]{{{latex_escape(task2_loss)}}}
  \\
  {{\footnotesize Task 2 final performance: PSNR {task2_psnr:.2f}, SSIM {task2_ssim:.4f}}}
\column{{0.48\linewidth}}
  \centering
  \includegraphics[width=\linewidth]{{{latex_escape(task3_loss)}}}
  \\
  {{\footnotesize Task 3 final performance: PSNR {task3_psnr:.2f}, SSIM {task3_ssim:.4f}}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Quantitative Comparison}}
\begin{{columns}}[T]
\column{{0.5\linewidth}}
\begin{{itemize}}
  \item Task 2 achieves PSNR {task2_psnr:.2f} and SSIM {task2_ssim:.4f}.
  \item Task 3 achieves PSNR {task3_psnr:.2f} and SSIM {task3_ssim:.4f}.
  \item Task 3 exceeds Task 2 by {psnr_gain:.2f} dB PSNR.
  \item Task 3 exceeds Task 2 by {ssim_gain:.4f} SSIM.
\end{{itemize}}
\column{{0.45\linewidth}}
  \includegraphics[width=\linewidth]{{{latex_escape(metric_chart)}}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Representative Reconstructions}}
\begin{{columns}}[T]
\column{{0.48\linewidth}}
  \includegraphics[width=\linewidth,height=0.73\textheight,keepaspectratio]{{{latex_escape(task2_recon)}}}
\column{{0.48\linewidth}}
  \includegraphics[width=\linewidth,height=0.73\textheight,keepaspectratio]{{{latex_escape(task3_best)}}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Task 3 Error Analysis}}
\begin{{columns}}[T]
\column{{0.42\linewidth}}
\begin{{itemize}}
  \item The lowest-performing cases are concentrated in structurally complex slices.
  \item Even these cases remain substantially better than the aliased input.
  \item Likely improvements: stronger edge-aware losses, deeper cascades, and hard-case-focused training.
\end{{itemize}}
\column{{0.54\linewidth}}
  \includegraphics[width=\linewidth,height=0.72\textheight,keepaspectratio]{{{latex_escape(task3_error)}}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Conclusions and AI Declaration}}
\begin{{itemize}}
  \item The baseline model removes most of the aliasing artifacts effectively.
  \item The multi-modal unrolled model provides the best quantitative fidelity and visual quality.
  \item T1 structural guidance and data consistency together yield measurable gains over the baseline.
  \item Future work includes stronger edge-aware objectives, deeper cascades, and targeted hard-case training.
  \item AI tools were used only for debugging support, experiment automation, and draft editing.
  \item All modelling choices, metric verification, result interpretation, and submitted materials were manually reviewed and confirmed.
\end{{itemize}}
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
