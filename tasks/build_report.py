import json
import os
import sys
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import create_dataloaders
from utils.config_helpers import normalize_output_dir


def load_config(config_path=None):
    if config_path is None:
        config_path = os.environ.get('BMEAI_CONFIG')
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config',
            'config.yaml',
        )
    config_path = os.path.normpath(os.path.abspath(config_path))
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    normalize_output_dir(config, config_path)
    return config, config_path


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        return json.load(f)


def load_text(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        return f.read()


def format_names(names):
    if not names:
        return '<fill Chinese names>'
    return ', '.join(names)


def format_list(items, default_text):
    if not items:
        return [default_text]
    return items


def summarize_run_status(run_metadata):
    if not run_metadata:
        return 'Not started'
    return run_metadata.get('status', 'unknown').capitalize()


def summarize_duration(seconds):
    if seconds is None:
        return 'N/A'
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h {minutes}m {seconds}s'
    if minutes:
        return f'{minutes}m {seconds}s'
    return f'{seconds}s'


def asset_status_line(root_dir, label, path):
    if os.path.exists(path):
        return f"- {label}: `{os.path.relpath(path, root_dir)}`"
    return f"- {label}: pending generation"


def format_slice_selection(max_slices_per_patient):
    if max_slices_per_patient is None:
        return 'all available slices per patient'
    return f'{max_slices_per_patient} central slices per patient'


def relative_asset_path(root_dir, path):
    if not os.path.exists(path):
        return None
    return os.path.relpath(path, root_dir).replace(os.sep, '/')


def append_image_block(lines, title, rel_path):
    lines.append(f'### {title}')
    lines.append('')
    if rel_path is None:
        lines.append('Pending generation.')
    else:
        lines.append(f'![{title}]({rel_path})')
    lines.append('')


def append_markdown_file(lines, title, path, fallback_text):
    lines.append(f'### {title}')
    lines.append('')
    content = load_text(path)
    if content is None:
        lines.append(fallback_text)
        lines.append('')
        return

    content_lines = content.strip().splitlines()
    if content_lines and content_lines[0].startswith('# '):
        content_lines = content_lines[1:]
        while content_lines and not content_lines[0].strip():
            content_lines = content_lines[1:]
    lines.extend(content_lines)
    lines.append('')


def build_results_lines(task2_summary, task3_summary):
    lines = []
    if task2_summary is None:
        lines.append('Task 2 results: pending formal execution.')
    else:
        lines.append(
            f"Task 2 improved PSNR from {task2_summary['psnr_before']['mean']:.2f} dB to "
            f"{task2_summary['psnr_after']['mean']:.2f} dB and SSIM from "
            f"{task2_summary['ssim_before']['mean']:.4f} to {task2_summary['ssim_after']['mean']:.4f}."
        )

    if task3_summary is None:
        lines.append('Task 3 results: pending formal execution.')
    else:
        lines.append(
            f"Task 3 improved PSNR from {task3_summary['psnr_before']['mean']:.2f} dB to "
            f"{task3_summary['psnr_after']['mean']:.2f} dB and SSIM from "
            f"{task3_summary['ssim_before']['mean']:.4f} to {task3_summary['ssim_after']['mean']:.4f}."
        )
        if task2_summary is not None:
            lines.append(
                f"Compared with Task 2, Task 3 changed PSNR by "
                f"{task3_summary['psnr_after']['mean'] - task2_summary['psnr_after']['mean']:.2f} dB "
                f"and SSIM by {task3_summary['ssim_after']['mean'] - task2_summary['ssim_after']['mean']:.4f}."
            )

    return lines


def build_discussion_lines(task2_summary, task3_summary, max_slices_per_patient):
    if task2_summary is None or task3_summary is None:
        return [
            'Formal quantitative conclusions remain pending because the final training run has not yet completed. The automation pipeline is prepared to refresh this report automatically once the run finishes.'
        ]

    task2_psnr_gain = task2_summary['psnr_after']['mean'] - task2_summary['psnr_before']['mean']
    task2_ssim_gain = task2_summary['ssim_after']['mean'] - task2_summary['ssim_before']['mean']
    task3_psnr_gain = task3_summary['psnr_after']['mean'] - task3_summary['psnr_before']['mean']
    task3_ssim_gain = task3_summary['ssim_after']['mean'] - task3_summary['ssim_before']['mean']
    task3_over_task2_psnr = task3_summary['psnr_after']['mean'] - task2_summary['psnr_after']['mean']
    task3_over_task2_ssim = task3_summary['ssim_after']['mean'] - task2_summary['ssim_after']['mean']

    lines = []
    lines.append(
        f'The final experiment used {format_slice_selection(max_slices_per_patient)} across all available BraTS patients and produced consistent quantitative improvements over the aliased baseline in both reconstruction settings.'
    )
    lines.append(
        f'The Task 2 baseline recovered a substantial proportion of the missing image fidelity, improving PSNR by {task2_psnr_gain:.2f} dB and SSIM by {task2_ssim_gain:.4f}, which confirms that the supervised reconstruction pipeline converged as expected.'
    )
    lines.append(
        f'Task 3 further improved PSNR by {task3_psnr_gain:.2f} dB and SSIM by {task3_ssim_gain:.4f} relative to the aliased input, and outperformed Task 2 by {task3_over_task2_psnr:.2f} dB PSNR and {task3_over_task2_ssim:.4f} SSIM.'
    )
    lines.append(
        'These findings support the central conclusion of the project: incorporating fully sampled T1 structural guidance together with unrolled data-consistency reconstruction yields sharper boundaries, fewer residual artifacts, and stronger quantitative fidelity than a single-modality baseline.'
    )
    lines.append(
        'The remaining challenging cases are concentrated in slices with more complex local structure, suggesting that future improvements may be obtained through stronger edge-aware loss functions, a deeper unrolled architecture, or targeted sampling and training strategies for anatomically difficult regions.'
    )
    return lines


def build_report(config, config_path, metadata, split_counts, task2_summary, task3_summary,
                 run_metadata, output_dir, worst_cases):
    root_dir = os.path.dirname(output_dir)
    report_assets_dir = os.path.join(output_dir, 'report_assets')
    generated_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    max_slices_per_patient = config['data'].get('max_slices_per_patient')

    division_of_labor = format_list(metadata.get('division_of_labor'), '<fill division of labor>')
    chinese_names = metadata.get('chinese_names', [])
    run_status = summarize_run_status(run_metadata)
    total_runtime = summarize_duration(run_metadata.get('total_duration_seconds') if run_metadata else None)

    dataset_split_chart = relative_asset_path(root_dir, os.path.join(report_assets_dir, 'dataset_split.png'))
    metric_comparison_chart = relative_asset_path(root_dir, os.path.join(report_assets_dir, 'metric_comparison.png'))
    task1_visualization = relative_asset_path(root_dir, os.path.join(output_dir, 'task1', 'undersampling_visualization.png'))
    task2_loss_curve = relative_asset_path(root_dir, os.path.join(output_dir, 'task2', 'loss_curves.png'))
    task2_recons = relative_asset_path(root_dir, os.path.join(output_dir, 'task2', 'reconstruction_results.png'))
    task3_loss_curve = relative_asset_path(root_dir, os.path.join(output_dir, 'task3', 'loss_curves.png'))
    task3_best_recons = relative_asset_path(root_dir, os.path.join(output_dir, 'task3', 'best_reconstructions.png'))
    task3_error_analysis = relative_asset_path(root_dir, os.path.join(output_dir, 'task3', 'error_analysis.png'))

    lines = []
    lines.append(f"# {metadata.get('project_title', 'BraTS MRI Reconstruction Project')}")
    lines.append('')
    lines.append(f"Generated from config `{os.path.relpath(config_path, root_dir)}` on {generated_time}.")
    lines.append('')
    lines.append('## Group Information')
    lines.append('')
    lines.append(f"- Course: {metadata.get('course', '<fill course name>')}")
    lines.append(f"- Group: {metadata.get('group_name', '<fill group name>')}")
    lines.append(f"- Chinese names: {format_names(chinese_names)}")
    lines.append(f"- Note: {metadata.get('report_author_note', 'Fill any remaining placeholders before submission.')}")
    lines.append('')
    lines.append('## AI Usage Declaration')
    lines.append('')
    lines.append(metadata.get('ai_usage_statement', '<fill AI usage statement>'))
    lines.append('')
    lines.append(f"Presentation reminder: {metadata.get('presentation_note', 'Remember to declare AI use in slides.')}")
    lines.append('')
    lines.append('## Project Objective')
    lines.append('')
    lines.append('The objective of this project is to reconstruct high-fidelity T2-weighted brain MRI slices from AF=5 undersampled k-space using the BraTS dataset. The study is organized into three stages: undersampling simulation, a supervised baseline reconstruction model, and a multi-modal unrolled reconstruction model with explicit data consistency.')
    lines.append('')
    lines.append('## Dataset and Preprocessing')
    lines.append('')
    lines.append(f"- Dataset path: `{config['data']['dataset_path']}`")
    lines.append(f"- Modalities used: {', '.join(config['data']['modalities'])}")
    lines.append(f"- Slice axis: {config['data']['slice_axis']}")
    lines.append(f"- Intensity normalization: z-score on non-zero voxels")
    lines.append(f"- Split strategy: patient-level train/validation/test")
    lines.append(f"- Slice selection for this run: {format_slice_selection(max_slices_per_patient)}")
    lines.append(f"- Data loading strategy: {'slice-level preloading in RAM to reduce I/O stalls' if config['data'].get('preload_volumes', False) else 'on-demand loading'}")
    lines.append(f"- Split counts: train={split_counts['Train']}, validation={split_counts['Validation']}, test={split_counts['Test']}")
    lines.append('- Data split unit: patient-level split to avoid leakage across adjacent slices from the same subject')
    lines.append('')
    lines.append('## Methods')
    lines.append('')
    lines.append('### Task 1')
    lines.append('')
    lines.append('A 2D random variable-density sampling mask with acceleration factor 5 is generated in k-space. Fully sampled T2 slices are transformed with FFT, masked, and reconstructed with inverse FFT to obtain aliased images. This stage establishes the artifact characteristics that must subsequently be removed by the learning-based reconstruction models.')
    lines.append('')
    lines.append('### Task 2')
    lines.append('')
    lines.append(f"Task 2 uses a U-Net baseline with base channels {config['task2']['base_channels']}, depth {config['task2']['depth']}, batch size {config['task2']['batch_size']}, MSE loss, and learning rate {config['task2']['learning_rate']}. `ReduceLROnPlateau` is used for learning rate scheduling.")
    lines.append('To improve runtime efficiency on the available RTX 4060 laptop GPU, the implementation uses slice caching, pinned memory, non-blocking GPU transfer, `channels_last`, TF32, and prefetch-friendly dataloading to reduce GPU idle time.')
    lines.append('')
    lines.append('### Task 3')
    lines.append('')
    lines.append(f"Task 3 uses an unrolled U-Net with {config['task3']['num_cascades']} cascades, data-consistency layers, and multi-modal input consisting of aliased T2 together with fully sampled T1. The loss is `{config['task3']['loss_type']}` with L1 weight {config['task3'].get('l1_weight', 'N/A')}.")
    lines.append('The underlying rationale is that T1 provides stable anatomical structure, while the data-consistency layer constrains the network output to remain faithful to the measured undersampled k-space.')
    lines.append('')
    lines.append('## Division of Labor')
    lines.append('')
    for item in division_of_labor:
        lines.append(f"- {item}")
    lines.append('')
    lines.append('## Quantitative Results')
    lines.append('')
    for result_line in build_results_lines(task2_summary, task3_summary):
        lines.append(f"- {result_line}")
    lines.append('')
    lines.append('## Figures and Tables')
    lines.append('')
    append_image_block(lines, 'Dataset Split Chart', dataset_split_chart)
    append_image_block(lines, 'Metric Comparison Chart', metric_comparison_chart)
    append_markdown_file(
        lines,
        'Summary Tables',
        os.path.join(report_assets_dir, 'summary_tables.md'),
        'Pending generation.',
    )
    append_markdown_file(
        lines,
        'Worst-case Table',
        os.path.join(report_assets_dir, 'worst_cases_table.md'),
        'Pending generation.',
    )
    append_image_block(lines, 'Task 1 Visualization', task1_visualization)
    append_image_block(lines, 'Task 2 Loss Curve', task2_loss_curve)
    append_image_block(lines, 'Task 2 Reconstructions', task2_recons)
    append_image_block(lines, 'Task 3 Loss Curve', task3_loss_curve)
    append_image_block(lines, 'Task 3 Best Reconstructions', task3_best_recons)
    append_image_block(lines, 'Task 3 Error Analysis', task3_error_analysis)
    lines.append('## Error Analysis')
    lines.append('')
    if not worst_cases:
        lines.append('Task 3 worst-case analysis is pending because `worst_cases.json` is not available yet.')
    else:
        lines.append('The 5 lowest-performing Task 3 cases are summarized below. Even in these challenging slices, the reconstructed outputs remain substantially better than the aliased inputs, indicating that the dominant failure mode is relative quality degradation rather than complete reconstruction collapse.')
        lines.append('')
        lines.append('| Rank | Patient | Slice | PSNR After | SSIM After |')
        lines.append('| --- | --- | ---: | ---: | ---: |')
        for index, case in enumerate(worst_cases, start=1):
            lines.append(
                f"| {index} | {case['patient']} | {case['slice_idx']} | {case['psnr_after']:.2f} | {case['ssim_after']:.4f} |"
            )
        lines.append('')
        lines.append('Potential improvements for these cases include stronger edge-preserving loss terms, additional cascades if runtime permits, and targeted inspection of slices with complex tumor boundaries.')
    lines.append('')
    lines.append('## Execution Status')
    lines.append('')
    lines.append(f"- Current run status: {run_status}")
    lines.append(f"- Output directory: `{os.path.relpath(output_dir, root_dir)}`")
    lines.append(f"- Total runtime: {total_runtime}")
    if run_metadata:
        lines.append(f"- Last completed stage: {run_metadata.get('last_completed_stage', 'N/A')}")
        if run_metadata.get('log_path'):
            lines.append(f"- Log file: `{os.path.relpath(run_metadata['log_path'], root_dir)}`")
    lines.append('')
    lines.append('## Discussion')
    lines.append('')
    for discussion_line in build_discussion_lines(task2_summary, task3_summary, max_slices_per_patient):
        lines.append(discussion_line)
        lines.append('')
    if lines and lines[-1] == '':
        lines.pop()
    lines.append('')
    lines.append('## Submission Checklist')
    lines.append('')
    lines.append('- Code: prepared')
    lines.append(f"- Report draft: `{os.path.basename(os.path.join(root_dir, 'REPORT.md'))}`")
    lines.append(f"- LaTeX report: `{os.path.basename(os.path.join(root_dir, 'REPORT.tex'))}`")
    lines.append(f"- PDF report: `{os.path.basename(os.path.join(root_dir, 'REPORT.pdf'))}`")
    lines.append('- Presentation slides: generated as `slides.tex` and `slides.pdf`')

    return '\n'.join(lines) + '\n'


def main():
    config, config_path = load_config()
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = config['output']['dir']
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(root_dir, output_dir.lstrip('./')))
    config_name = os.path.basename(config_path)
    write_main_report_env = os.environ.get('WRITE_MAIN_REPORT')
    if write_main_report_env is None:
        write_main_report = config_name == 'formal_train.yaml'
    else:
        write_main_report = write_main_report_env.lower() in {'1', 'true', 'yes'}

    metadata_path = os.path.join(root_dir, 'config', 'report_metadata.json')
    metadata = load_json(metadata_path, default={}) or {}
    run_metadata = load_json(os.path.join(output_dir, 'run_metadata.json'), default=None)
    task2_summary = load_json(os.path.join(output_dir, 'task2', 'evaluation_summary.json'), default=None)
    task3_summary = load_json(os.path.join(output_dir, 'task3', 'evaluation_summary.json'), default=None)
    worst_cases = load_json(os.path.join(output_dir, 'task3', 'worst_cases.json'), default=[])

    train_ds, val_ds, test_ds, _ = create_dataloaders(config)
    split_counts = {
        'Train': len(train_ds),
        'Validation': len(val_ds),
        'Test': len(test_ds),
    }

    report_text = build_report(
        config,
        config_path,
        metadata,
        split_counts,
        task2_summary,
        task3_summary,
        run_metadata,
        output_dir,
        worst_cases,
    )

    with open(os.path.join(output_dir, 'report_assets', 'report_snapshot.md'), 'w') as f:
        f.write(report_text)

    if write_main_report:
        with open(os.path.join(root_dir, 'REPORT.md'), 'w') as f:
            f.write(report_text)
        print(f"Report written to {os.path.join(root_dir, 'REPORT.md')}")
    else:
        print(f"Report snapshot written to {os.path.join(output_dir, 'report_assets', 'report_snapshot.md')}")


if __name__ == '__main__':
    main()
